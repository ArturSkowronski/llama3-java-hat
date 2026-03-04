# Babylon JDK — OpenJDK with Code Reflection support (Project Babylon).
#
# Built by overriding nixpkgs' openjdk25 with the Babylon `code-reflection`
# branch source. The Babylon JDK builds identically to mainline OpenJDK
# (configure && make images), so we reuse the full nixpkgs build infrastructure.
#
# If the override approach causes too many patch conflicts, fall back to a
# bespoke FOD derivation (see plan notes).
#
# To recompute the source hash after a branch update:
#   1. Set hash = lib.fakeHash below
#   2. Run: nix build .#babylon-jdk (will fail with hash mismatch)
#   3. Copy the "got:" hash into the fetchFromGitHub call
{ pkgs
,
}:

let
  babylonSrc = pkgs.fetchFromGitHub {
    owner = "openjdk";
    repo = "babylon";
    # Pin to a specific commit for reproducibility.
    # Update this rev + hash when tracking upstream.
    rev = "code-reflection";
    hash = "sha256-w02MsEFXiQiuIryJIeHkJmYd3w8v6xVKcnwmOXesNH0=";
  };
in
pkgs.openjdk25.overrideAttrs (oldAttrs: {
  pname = "babylon-jdk";
  version = "26-babylon";

  src = babylonSrc;

  # Babylon branch diverges from mainline OpenJDK 25; nixpkgs patches
  # (fix-java-home, read-truststore-from-env, make-4.4.1, etc.) likely
  # won't apply cleanly. Clear them — we can re-add selectively if needed.
  patches = [ ];

  # Adjust version string in configure flags.
  # The original flags from openjdk25 include --with-version-string=<version>.
  # We replace version-related flags for Babylon.
  configureFlags =
    let
      dropFlag = prefix: flag:
        (builtins.match "${prefix}.*" flag) != null;
    in
    (builtins.filter (flag:
      !(dropFlag "--with-version-string" flag)
      && !(dropFlag "--with-vendor-version-string" flag)
    ) (oldAttrs.configureFlags or [ ])) ++ [
      "--with-version-string=26-babylon"
      "--with-vendor-version-string=(nix-babylon)"
    ];
})
