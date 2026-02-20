# Nix checks that run the project's Gradle test tasks.
#
# - unit-tests: runs `./gradlew test` (no model needed)
# - plain-integration-tests: runs `./gradlew plainIntegrationTest`
# - hat-integration-tests: runs `./gradlew hatIntegrationTest`
#
# The Llama model is fetched as a fixed-output derivation so all checks
# run in a pure sandbox — no --impure flag needed.
{ pkgs
, gradle2nixBuilders
, babylon-jdk
, hat-artifacts
,
}:

let
  lockRepo = gradle2nixBuilders.buildMavenRepo {
    lockFile = ../../gradle.lock;
  };

  # Llama 3.2 1B Instruct FP16 GGUF (~2.5 GB)
  llamaModel = pkgs.fetchurl {
    url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf";
    hash = "sha256-HzOtQ9K4W5CP8G/nACtpgGpXNZubJhfKJ9e96kKK4UY=";
    name = "Llama-3.2-1B-Instruct-f16.gguf";
  };

  gradleDist = pkgs.fetchurl {
    url = "https://services.gradle.org/distributions/gradle-9.3.0-bin.zip";
    hash = "sha256-DVhfadoJH8Wyvs7Yd/6rVaMGTUO4odRq6weZawkV4OA=";
  };

  initScript = pkgs.writeText "offline-repos.gradle" ''
    import org.gradle.util.GradleVersion

    static void configureRepos(RepositoryHandler repositories) {
      repositories.configureEach { ArtifactRepository repo ->
        if (repo instanceof MavenArtifactRepository) {
          repo.setArtifactUrls(new HashSet<URI>())
          repo.url 'file:${lockRepo}'
          repo.metadataSources {
            gradleMetadata()
            mavenPom()
            artifact()
          }
        } else if (repo instanceof IvyArtifactRepository) {
          repo.url 'file:${lockRepo}'
          repo.layout('maven')
          repo.metadataSources {
            gradleMetadata()
            ivyDescriptor()
            artifact()
          }
        } else if (repo instanceof UrlArtifactRepository) {
          repo.url 'file:/homeless-shelter'
        }
      }
    }

    beforeSettings { settings ->
      configureRepos(settings.pluginManagement.repositories)
      configureRepos(settings.buildscript.repositories)
      if (GradleVersion.current() >= GradleVersion.version("6.8")) {
        configureRepos(settings.dependencyResolutionManagement.repositories)
      }
    }

    beforeProject { project ->
      configureRepos(project.buildscript.repositories)
      configureRepos(project.repositories)
    }
  '';

  projectSrc = pkgs.lib.cleanSourceWith {
    src = ../..;
    filter = path: type:
      let baseName = baseNameOf path; in
      !(builtins.elem baseName [ ".direnv" "result" ".git" ".gradle" "build" ".idea" ".claude" ".junie" "results" ])
      && !(pkgs.lib.hasSuffix ".gguf" path);
  };

  # Shared builder for running Gradle test tasks in the sandbox
  mkGradleCheck =
    { pname
    , gradleTask
    , extraEnv ? { }
    , impure ? false
    }:
    pkgs.stdenv.mkDerivation ({
      inherit pname;
      version = "0.1.0";

      src = projectSrc;

      nativeBuildInputs = [
        babylon-jdk
        pkgs.unzip
      ];

      dontConfigure = true;

      JAVA_HOME = "${babylon-jdk}/lib/openjdk";
      JAVA_BABYLON_ROOT = "${hat-artifacts}";

      buildPhase = ''
        runHook preBuild

        export HOME=$PWD
        export GRADLE_USER_HOME=$PWD/.gradle
        mkdir -p $GRADLE_USER_HOME

        # Set up offline Gradle wrapper
        local wrapper_props=gradle/wrapper/gradle-wrapper.properties
        local dist_dir=$GRADLE_USER_HOME/wrapper/dists/gradle-9.3.0-bin/nix
        mkdir -p "$dist_dir"
        cp ${gradleDist} "$dist_dir/gradle-9.3.0-bin.zip"
        touch "$dist_dir/gradle-9.3.0-bin.zip.ok"
        unzip -q "$dist_dir/gradle-9.3.0-bin.zip" -d "$dist_dir"
        sed -i "s|distributionUrl=.*|distributionUrl=file\\\\:$dist_dir/gradle-9.3.0-bin.zip|" "$wrapper_props"
        chmod +x gradlew

        ./gradlew \
          --no-daemon \
          --no-build-cache \
          --no-configuration-cache \
          --init-script ${initScript} \
          -Dorg.gradle.java.installations.paths=${babylon-jdk}/lib/openjdk \
          ${gradleTask}

        runHook postBuild
      '';

      installPhase = ''
        # Checks just need a marker output
        mkdir -p $out
        touch $out/passed
        cp -r build/reports $out/reports 2>/dev/null || true
      '';
    } // extraEnv);

in
{
  # Unit tests — pure, no model needed
  unit-tests = mkGradleCheck {
    pname = "llama3-java-hat-unit-tests";
    gradleTask = "test";
  };

  # Plain integration tests (no HAT) — pure, model fetched by Nix
  plain-integration-tests = mkGradleCheck {
    pname = "llama3-java-hat-plain-integration-tests";
    gradleTask = "plainIntegrationTest";
    extraEnv = { LLAMA_FP16_PATH = "${llamaModel}"; };
  };

  # HAT integration tests — pure, model fetched by Nix
  hat-integration-tests = mkGradleCheck {
    pname = "llama3-java-hat-hat-integration-tests";
    gradleTask = "hatIntegrationTest";
    extraEnv = { LLAMA_FP16_PATH = "${llamaModel}"; };
  };
}
