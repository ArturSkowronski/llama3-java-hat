plugins {
    application
    java
}

val babylonRoot = System.getenv("JAVA_BABYLON_ROOT") ?: throw GradleException("JAVA_BABYLON_ROOT environment variable is not set")
val babylonHome = if (System.getProperty("os.name").contains("Mac")) {
    val macJDK = "$babylonRoot/Contents/Home"
    if (File(macJDK).exists()) macJDK else babylonRoot
} else {
    babylonRoot
}

repositories {
    mavenCentral()
    flatDir {
        dirs("$babylonHome/hat/build")
    }
}

dependencies {
    implementation(files("$babylonHome/hat/build/hat-core-1.0.jar"))
    implementation(files("$babylonHome/hat/build/hat-optkl-1.0.jar"))
    implementation(files("$babylonHome/hat/build/hat-backend-java-seq-1.0.jar"))
    implementation(files("$babylonHome/hat/build/hat-backend-java-mt-1.0.jar"))
    implementation(files("$babylonHome/hat/build/hat-backend-ffi-opencl-1.0.jar"))
    implementation(files("$babylonHome/hat/build/hat-backend-ffi-shared-1.0.jar"))

    testImplementation("org.junit.jupiter:junit-jupiter:6.0.0")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform {
        excludeTags("integration")
    }
    jvmArgs(application.applicationDefaultJvmArgs)
}

fun registerIntegrationTest(name: String, description: String, vararg tags: String) =
    tasks.register<Test>(name) {
        this.description = description
        group = "verification"

        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath

        useJUnitPlatform {
            tags.forEach { includeTags(it) }
            excludeTags("regression")
        }
        jvmArgs(application.applicationDefaultJvmArgs)
        jvmArgs("-Xmx5g")
        maxParallelForks = 1 // Avoid running multiple inference backends at once (GPU/FFI/large heap).
        forkEvery = 1 // Fork a new JVM per test class to prevent OOM from repeated model loads
        failFast = false

        // Forward test output to Gradle console so CI sees activity
        // (prevents GitHub Actions no-output timeout during long inference)
        testLogging {
            showStandardStreams = true
            events("started", "passed", "failed")
        }

        System.getenv("TINY_LLAMA_PATH")?.let { environment("TINY_LLAMA_PATH", it) }
        System.getenv("LLAMA_FP16_PATH")?.let { environment("LLAMA_FP16_PATH", it) }
    }

registerIntegrationTest("integrationTest",
    "Runs all integration tests.", "integration")

registerIntegrationTest("plainJavaIntegrationTest",
    "Runs plain Java integration tests (no HAT).", "plain-java")

registerIntegrationTest("hatSequentialIntegrationTest",
    "Runs HAT Sequential backend integration tests.", "hat-sequential")

registerIntegrationTest("hatGpuIntegrationTest",
    "Runs HAT GPU (OpenCL/JavaMT) integration tests.", "hat-gpu")

registerIntegrationTest("benchmark",
    "Runs inference benchmark across all backends.", "benchmark")

registerIntegrationTest("benchmarkPlain",
    "Runs inference benchmark: Plain Java only.", "benchmark-plain")

registerIntegrationTest("benchmarkHatSeq",
    "Runs inference benchmark: HAT Java Sequential only.", "benchmark-hat-seq")

registerIntegrationTest("benchmarkHatMt",
    "Runs inference benchmark: HAT Java MT only.", "benchmark-hat-mt")

registerIntegrationTest("benchmarkOpencl",
    "Runs inference benchmark: HAT OpenCL only (requires RUN_OPENCL_BENCHMARKS=true).", "benchmark-opencl")

tasks.register("benchmarkAll") {
    description = "Runs all benchmark shards (Plain, HAT seq, HAT mt, OpenCL)."
    group = "verification"
    dependsOn("benchmarkPlain", "benchmarkHatSeq", "benchmarkHatMt", "benchmarkOpencl")
}

registerIntegrationTest("openclBugTest",
    "Reproduces OpenCL FFI IllegalAccessError on ComputeEntrypoint.lowered.", "opencl-bug")

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(26))
    }
}

tasks.withType<JavaCompile> {
    options.compilerArgs.addAll(listOf(
        "--enable-preview",
        "--add-modules", "jdk.incubator.code",
        "--add-exports", "java.base/jdk.internal.vm.annotation=ALL-UNNAMED"
    ))
}

application {
    mainClass.set("com.arturskowronski.llama3babylon.hat.GGUFReader")
    applicationDefaultJvmArgs = listOf(
        "--enable-preview",
        "--add-modules=jdk.incubator.code",
        "--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED",
        "--enable-native-access=ALL-UNNAMED",
        "-Djava.library.path=$babylonHome/hat/build"
    )
}

tasks.named<JavaExec>("run") {
    jvmArgs(application.applicationDefaultJvmArgs)
}
