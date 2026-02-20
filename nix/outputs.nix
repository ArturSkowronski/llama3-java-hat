{ self
, nixpkgs
, flake-utils
, ...
}@inputs:
flake-utils.lib.eachDefaultSystem (
  system:
  let
    pkgs = import nixpkgs {
      inherit system;
    };

    # ── Babylon JDK (OpenJDK + Code Reflection) ─────────────────────
    babylon-jdk = import ./packages/babylon-jdk { inherit pkgs; };

    # ── HAT artifacts (JARs + native libs) ───────────────────────────
    hat-artifacts = import ./packages/hat { inherit pkgs babylon-jdk; };
  in
  {
    packages = {
      default = hat-artifacts;
      inherit babylon-jdk hat-artifacts;
    };

    devShells.default = pkgs.mkShell {
      packages = with pkgs; [
        cmake
        git
      ];

      JAVA_HOME = "${babylon-jdk}/lib/openjdk";
      JAVA_BABYLON_ROOT = "${hat-artifacts}";

      shellHook = ''
        # Ensure Gradle toolchain resolution finds the Babylon JDK.
        # Without this, the foojay resolver tries to download JDK 26.
        export GRADLE_OPTS="''${GRADLE_OPTS:-} -Dorg.gradle.java.installations.paths=${babylon-jdk}/lib/openjdk"

        # Add Babylon JDK to PATH so `java` resolves correctly
        export PATH="${babylon-jdk}/lib/openjdk/bin:$PATH"
      '';
    };
  }
)
