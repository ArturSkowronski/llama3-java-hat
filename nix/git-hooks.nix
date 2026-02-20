# Pre-commit hooks via cachix/git-hooks.nix
{ git-hooks, system, treeFmtWrapper, babylon-jdk, ... }:

git-hooks.lib.${system}.run {
  src = ../.;

  hooks = {
    # ── Formatting ──────────────────────────────────────────────────
    treefmt = {
      enable = true;
      package = treeFmtWrapper;
    };

    # ── Gradle compilation check ──────────────────────────────────
    gradle-check = {
      enable = true;
      name = "gradle check (compile only)";
      entry = "bash -c './gradlew compileJava -x test 2>&1'";
      files = "\\.java$";
      pass_filenames = false;
      stages = [ "pre-push" ];
    };
  };
}
