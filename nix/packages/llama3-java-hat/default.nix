# Offline Gradle build of llama3-java-hat using gradle2nix V2.
#
# Produces the built application JARs in $out/lib/.
# Uses a pre-fetched Gradle 9.3 distribution and an offline Maven repo
# generated from gradle.lock by gradle2nix.
{ pkgs
, gradle2nixBuilders
, babylon-jdk
, hat-artifacts
,
}:

let
  # Build the offline Maven repo from the gradle.lock
  lockRepo = gradle2nixBuilders.buildMavenRepo {
    lockFile = ../../../gradle.lock;
  };

  # Pre-fetch Gradle 9.3.0 distribution (gradlew needs this)
  gradleDist = pkgs.fetchurl {
    url = "https://services.gradle.org/distributions/gradle-9.3.0-bin.zip";
    hash = "sha256-DVhfadoJH8Wyvs7Yd/6rVaMGTUO4odRq6weZawkV4OA=";
  };

  # Gradle init script that redirects ALL repositories to the offline Maven repo
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
    src = ../../..;
    filter = path: type:
      let baseName = baseNameOf path; in
      # Include project source files, exclude Nix/IDE artifacts
      !(builtins.elem baseName [ ".direnv" "result" ".git" ".gradle" "build" ".idea" ".claude" ".junie" "results" ])
      && !(pkgs.lib.hasSuffix ".gguf" path);
  };

in
pkgs.stdenv.mkDerivation {
  pname = "llama3-java-hat";
  version = "0.1.0";

  src = projectSrc;

  nativeBuildInputs = [
    babylon-jdk
    pkgs.unzip
  ];

  # Disable default phases — we handle everything in buildPhase
  dontConfigure = true;

  JAVA_HOME = "${babylon-jdk}/lib/openjdk";
  JAVA_BABYLON_ROOT = "${hat-artifacts}";

  buildPhase = ''
    runHook preBuild

    export HOME=$PWD
    export GRADLE_USER_HOME=$PWD/.gradle
    mkdir -p $GRADLE_USER_HOME

    # Set up the Gradle wrapper to use the pre-fetched distribution
    # instead of downloading it
    local wrapper_props=gradle/wrapper/gradle-wrapper.properties
    local dist_dir=$GRADLE_USER_HOME/wrapper/dists/gradle-9.3.0-bin/nix
    mkdir -p "$dist_dir"
    cp ${gradleDist} "$dist_dir/gradle-9.3.0-bin.zip"
    # Gradle wrapper checks for a marker file to know the dist is ready
    touch "$dist_dir/gradle-9.3.0-bin.zip.ok"
    # Unzip the distribution
    unzip -q "$dist_dir/gradle-9.3.0-bin.zip" -d "$dist_dir"

    # Point the wrapper properties at our local distribution
    sed -i "s|distributionUrl=.*|distributionUrl=file\\\\:$dist_dir/gradle-9.3.0-bin.zip|" "$wrapper_props"

    chmod +x gradlew

    # Build the project with offline repos
    ./gradlew \
      --no-daemon \
      --no-build-cache \
      --no-configuration-cache \
      --init-script ${initScript} \
      -Dorg.gradle.java.installations.paths=${babylon-jdk}/lib/openjdk \
      build -x test

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/lib
    cp build/libs/*.jar $out/lib/ 2>/dev/null || true

    # Also copy distribution archives if created
    mkdir -p $out/dist
    cp build/distributions/*.tar $out/dist/ 2>/dev/null || true
    cp build/distributions/*.zip $out/dist/ 2>/dev/null || true

    runHook postInstall
  '';
}
