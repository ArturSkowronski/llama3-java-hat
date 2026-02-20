{ pkgs, ... }:
{
  projectRootFile = "flake.nix";

  programs = {
    nixpkgs-fmt.enable = true;
    shfmt.enable = true;
    shellcheck.enable = true;
    google-java-format.enable = true;
  };

  settings = {
    global.excludes = [
      "*.patch"
      "*.json"
      "*.md"
      ".gitignore"
      "flake.lock"
      "treefmt.toml"
      "*.gguf"
      "gradle/wrapper/*"
    ];
  };
}
