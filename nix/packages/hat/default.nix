# HAT (Hardware Accelerator Toolkit) build artifacts.
#
# Builds HAT from the Babylon source tree using `java @hat/bld` (a Java
# source-launch build script, JEP 458). Produces JARs and native .so
# libraries in $out/hat/build/.
#
# The output layout matches what build.gradle.kts expects from
# JAVA_BABYLON_ROOT: $out/hat/build/hat-core-1.0.jar, etc.
{ pkgs
, babylon-jdk
,
}:

let
  babylonSrc = pkgs.fetchFromGitHub {
    owner = "openjdk";
    repo = "babylon";
    # Must match the rev used in babylon-jdk/default.nix.
    rev = "code-reflection";
    hash = "sha256-w02MsEFXiQiuIryJIeHkJmYd3w8v6xVKcnwmOXesNH0=";
  };
in
pkgs.stdenv.mkDerivation {
  pname = "hat-artifacts";
  version = "1.0";

  src = babylonSrc;

  # Patch upstream bld.java:
  # - Add missing optkl module build (new dependency of hat-core)
  # - Remove dead reference to nonexistent 'tools' directory
  # - Wire optkl as transitive classpath dep for all downstream modules
  patches = [ ./fix-bld-optkl-and-tools.patch ];

  nativeBuildInputs = [
    babylon-jdk
    pkgs.cmake
    pkgs.patchelf
    # OpenCL headers/ICD for FFI backends (optional but useful)
    pkgs.opencl-headers
    pkgs.ocl-icd
  ];

  # HAT uses java as its build system; cmake is only invoked internally
  # by bld.java for native backends. Prevent stdenv from auto-running cmake.
  dontUseCmakeConfigure = true;
  dontConfigure = true;

  # Point JAVA_HOME at the Babylon JDK so `java` is on PATH
  JAVA_HOME = "${babylon-jdk}/lib/openjdk";

  buildPhase = ''
    runHook preBuild

    export HOME=$PWD
    cd hat

    # Use the patched upstream bld.java via the @hat/bld argfile
    ${babylon-jdk}/lib/openjdk/bin/java \
      --add-modules jdk.incubator.code \
      --enable-preview \
      --source 26 \
      hat/bld.java

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    # Reproduce the layout expected by JAVA_BABYLON_ROOT:
    #   $out/hat/build/*.jar  $out/hat/build/*.so
    mkdir -p $out/hat/build

    # Copy JARs
    cp build/*.jar $out/hat/build/

    # Copy native shared libraries and fix their RPATHs
    for f in build/*.so build/*.dylib; do
      if [ -e "$f" ]; then
        cp "$f" $out/hat/build/
        # Strip the /build/ RPATH reference left by cmake
        patchelf --shrink-rpath "$out/hat/build/$(basename $f)" || true
        patchelf --remove-rpath "$out/hat/build/$(basename $f)" || true
      fi
    done

    runHook postInstall
  '';
}
