{ self
, nixpkgs
, flake-utils
, gradle2nix
, treefmt-nix
, git-hooks
, ...
}@inputs:
flake-utils.lib.eachDefaultSystem (
  system:
  let
    pkgs = import nixpkgs {
      inherit system;
    };

    gradle2nixBuilders = gradle2nix.builders.${system};

    # ── Babylon JDK (OpenJDK + Code Reflection) ─────────────────────
    babylon-jdk = import ./packages/babylon-jdk { inherit pkgs; };

    # ── HAT artifacts (JARs + native libs) ───────────────────────────
    hat-artifacts = import ./packages/hat { inherit pkgs babylon-jdk; };

    # ── llama3-java-hat offline Gradle build ─────────────────────────
    llama3-java-hat = import ./packages/llama3-java-hat {
      inherit pkgs gradle2nixBuilders babylon-jdk hat-artifacts;
    };

    # ── Formatting ───────────────────────────────────────────────────
    treeFmt = treefmt-nix.lib.evalModule pkgs ./treefmt.nix;

    # ── Pre-commit hooks ─────────────────────────────────────────────
    preCommitCheck = import ./git-hooks.nix {
      inherit git-hooks system babylon-jdk;
      treeFmtWrapper = treeFmt.config.build.wrapper;
    };

    # ── Common dev tools ─────────────────────────────────────────────
    commonDevTools = with pkgs; [
      git
      jq
      ripgrep
      fd
      bat
      direnv
      nixd
    ];
  in
  {
    packages = {
      default = llama3-java-hat;
      inherit babylon-jdk hat-artifacts llama3-java-hat;
    };

    checks = {
      formatting = treeFmt.config.build.check self;
    };

    formatter = treeFmt.config.build.wrapper;

    devShells.default = pkgs.mkShell {
      packages = [
        pkgs.cmake
        treeFmt.config.build.wrapper
      ] ++ commonDevTools;

      JAVA_HOME = "${babylon-jdk}/lib/openjdk";
      JAVA_BABYLON_ROOT = "${hat-artifacts}";

      shellHook = ''
        # Ensure Gradle toolchain resolution finds the Babylon JDK.
        # Without this, the foojay resolver tries to download JDK 26.
        export GRADLE_OPTS="''${GRADLE_OPTS:-} -Dorg.gradle.java.installations.paths=${babylon-jdk}/lib/openjdk"

        # Add Babylon JDK to PATH so `java` resolves correctly
        export PATH="${babylon-jdk}/lib/openjdk/bin:$PATH"
      '' + preCommitCheck.shellHook;
    };
  }
)
