{
  description = "llama3-java-hat — Llama 3 inference on Project Babylon HAT";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems.url = "github:nix-systems/x86_64-linux";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
    flake-compat.url = "github:lix-project/flake-compat";
  };

  outputs = inputs: import ./nix/outputs.nix inputs;
}
