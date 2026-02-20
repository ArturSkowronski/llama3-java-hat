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

    # ── Write a minimal build script ────────────────────────────────
    # The upstream bld.java is out of sync with the source tree (missing
    # optkl build, references nonexistent 'tools' dir, tests fail to
    # compile). We write a trimmed script that builds only the JARs the
    # llama3-java-hat project needs: optkl, core, and the backends.
    cat > hat/bldnix.java << 'NIXBLD'
    import static java.lang.IO.println;
    void main(String[] args) {
        var dir = Script.DirEntry.current();
        var buildDir = Script.BuildDir.of(dir.path("build")).create();

        // optkl must be built first — core depends on it
        var optkl = buildDir.mavenStyleBuild(
                dir.existingDir("optkl"), "hat-optkl-1.0.jar"
        );
        println("Created hat-optkl-1.0.jar");

        var core = buildDir.mavenStyleBuild(
                dir.existingDir("core"), "hat-core-1.0.jar", optkl
        );
        println("Created hat-core-1.0.jar");

        // Java backends (no native code needed)
        var backendsDir = dir.existingDir("backends");
        var javaBackendsDir = backendsDir.existingDir("java");
        buildDir.mavenStyleBuild(javaBackendsDir.existingDir("mt"),
                "hat-backend-java-mt-1.0.jar", optkl, core
        );
        println("Created hat-backend-java-mt-1.0.jar");

        buildDir.mavenStyleBuild(javaBackendsDir.existingDir("seq"),
                "hat-backend-java-seq-1.0.jar", optkl, core
        );
        println("Created hat-backend-java-seq-1.0.jar");

        // FFI backends (shared + opencl)
        var ffiBackendsDir = backendsDir.existingDir("ffi");
        var ffiShared = buildDir.mavenStyleBuild(
                ffiBackendsDir.existingDir("shared"), "hat-backend-ffi-shared-1.0.jar", optkl, core
        );
        println("Created hat-backend-ffi-shared-1.0.jar");

        if (ffiBackendsDir.optionalDir("opencl") instanceof Script.DirEntry ffiBackendDir) {
            buildDir.mavenStyleBuild(
                    ffiBackendDir, "hat-backend-ffi-opencl-1.0.jar", optkl, core, ffiShared
            );
            println("Created hat-backend-ffi-opencl-1.0.jar");
        }

        // Native FFI backends (cmake)
        var cmakeBuildDir = buildDir.buildDir("cmake-build-debug");
        if (!cmakeBuildDir.exists()) {
            Script.cmake($ -> $ .verbose(false) .source_dir(ffiBackendsDir) .build_dir(cmakeBuildDir) .copy_to(buildDir));
        }
        Script.cmake($ -> $ .build(cmakeBuildDir));
        println("Native backends built");
    }
    NIXBLD

    # ── Build HAT ───────────────────────────────────────────────────
    ${babylon-jdk}/lib/openjdk/bin/java \
      --add-modules jdk.incubator.code \
      --enable-preview \
      --source 26 \
      hat/bldnix.java

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
