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
        // Keep `test` unit-only. Integration/benchmark/regression tests are run by dedicated tasks/workflows.
        excludeTags("plain-integration", "hat-integration", "benchmark", "regression")
    }
    jvmArgs(application.applicationDefaultJvmArgs)
}

fun Test.configureBaseTestTask(descriptionText: String) {
    description = descriptionText
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    jvmArgs(application.applicationDefaultJvmArgs)
}

fun Test.configureLongRunningJvm() {
    jvmArgs("-Xmx5g")
    maxParallelForks = 1 // Avoid running multiple inference backends at once (GPU/FFI/large heap).
    forkEvery = 1 // Fork a new JVM per test class to prevent OOM from repeated model loads.
    failFast = false
}

fun Test.configureVerboseTestLogging() {
    // Forward test output to Gradle console so CI sees activity.
    // Prevents GitHub Actions no-output timeout during long inference.
    testLogging {
        showStandardStreams = true
        events("started", "passed", "failed")
    }
}

fun registerIntegrationTest(name: String, description: String, vararg tags: String) =
    tasks.register<Test>(name) {
        configureBaseTestTask(description)

        useJUnitPlatform {
            tags.forEach { includeTags(it) }
            excludeTags("regression")
        }
        configureLongRunningJvm()
        configureVerboseTestLogging()

        System.getenv("TINY_LLAMA_PATH")?.let { environment("TINY_LLAMA_PATH", it) }
        System.getenv("LLAMA_FP16_PATH")?.let { environment("LLAMA_FP16_PATH", it) }
    }

registerIntegrationTest("integrationTest",
    "Runs all integration tests.", "plain-integration", "hat-integration")

registerIntegrationTest("plainIntegrationTest",
    "Runs plain Java integration tests (no HAT).", "plain-integration")

registerIntegrationTest("hatIntegrationTest",
    "Runs HAT backend integration tests (JavaMT/OpenCL + backend-dispatch smoke).", "hat-integration")

fun registerBenchmarkTestByPattern(name: String, description: String, testPattern: String) =
    tasks.register<Test>(name) {
        configureBaseTestTask(description)

        useJUnitPlatform {
            includeTags("benchmark")
            excludeTags("regression")
        }
        filter {
            includeTestsMatching(testPattern)
        }

        configureLongRunningJvm()
        configureVerboseTestLogging()

        System.getenv("LLAMA_FP16_PATH")?.let { environment("LLAMA_FP16_PATH", it) }
        System.getenv("RUN_OPENCL_BENCHMARKS")?.let { environment("RUN_OPENCL_BENCHMARKS", it) }
    }

registerBenchmarkTestByPattern(
    "benchmarkKernelGEMV",
    "Runs per-kernel benchmark comparison for GEMV.",
    "com.arturskowronski.llama3babylon.hat.benchmark.KernelModeInferenceBenchmarkTest.benchmarkGEMVAcrossModes"
)

registerBenchmarkTestByPattern(
    "benchmarkKernelRMSNorm",
    "Runs per-kernel benchmark comparison for RMSNorm.",
    "com.arturskowronski.llama3babylon.hat.benchmark.KernelModeInferenceBenchmarkTest.benchmarkRMSNormAcrossModes"
)

registerBenchmarkTestByPattern(
    "benchmarkKernelRoPE",
    "Runs per-kernel benchmark comparison for RoPE.",
    "com.arturskowronski.llama3babylon.hat.benchmark.KernelModeInferenceBenchmarkTest.benchmarkRoPEAcrossModes"
)

registerBenchmarkTestByPattern(
    "benchmarkKernelSiLU",
    "Runs per-kernel benchmark comparison for SiLU.",
    "com.arturskowronski.llama3babylon.hat.benchmark.KernelModeInferenceBenchmarkTest.benchmarkSiLUAcrossModes"
)

registerBenchmarkTestByPattern(
    "benchmarkKernelSoftmax",
    "Runs per-kernel benchmark comparison for Softmax.",
    "com.arturskowronski.llama3babylon.hat.benchmark.KernelModeInferenceBenchmarkTest.benchmarkSoftmaxAcrossModes"
)

registerBenchmarkTestByPattern(
    "benchmarkKernelAttention",
    "Runs per-kernel benchmark comparison for Attention.",
    "com.arturskowronski.llama3babylon.hat.benchmark.KernelModeInferenceBenchmarkTest.benchmarkAttentionAcrossModes"
)

registerBenchmarkTestByPattern(
    "benchmarkMicroGEMV",
    "Runs micro-benchmark for GEMV across Plain/Seq/MT/OpenCL.",
    "com.arturskowronski.llama3babylon.hat.benchmark.KernelMicroBenchmarkTest.benchmarkMicroGEMV"
)

registerBenchmarkTestByPattern(
    "benchmarkMicroRMSNorm",
    "Runs micro-benchmark for RMSNorm across Plain/Seq/MT/OpenCL.",
    "com.arturskowronski.llama3babylon.hat.benchmark.KernelMicroBenchmarkTest.benchmarkMicroRMSNorm"
)

registerBenchmarkTestByPattern(
    "benchmarkMicroRoPE",
    "Runs micro-benchmark for RoPE across Plain/Seq/MT/OpenCL.",
    "com.arturskowronski.llama3babylon.hat.benchmark.KernelMicroBenchmarkTest.benchmarkMicroRoPE"
)

registerBenchmarkTestByPattern(
    "benchmarkMicroSiLU",
    "Runs micro-benchmark for SiLU across Plain/Seq/MT/OpenCL.",
    "com.arturskowronski.llama3babylon.hat.benchmark.KernelMicroBenchmarkTest.benchmarkMicroSiLU"
)

registerBenchmarkTestByPattern(
    "benchmarkMicroSoftmax",
    "Runs micro-benchmark for Softmax across Plain/Seq/MT/OpenCL.",
    "com.arturskowronski.llama3babylon.hat.benchmark.KernelMicroBenchmarkTest.benchmarkMicroSoftmax"
)

registerBenchmarkTestByPattern(
    "benchmarkMicroAttention",
    "Runs micro-benchmark for Attention across Plain/Seq/MT/OpenCL.",
    "com.arturskowronski.llama3babylon.hat.benchmark.KernelMicroBenchmarkTest.benchmarkMicroAttention"
)

registerBenchmarkTestByPattern(
    "benchmarkInferencePlainJava",
    "Runs inference benchmark for Plain Java.",
    "com.arturskowronski.llama3babylon.hat.benchmark.PlainJavaInferenceBenchmarkTest.benchmarkPlainJava"
)

registerBenchmarkTestByPattern(
    "benchmarkInferenceHatSeq",
    "Runs inference benchmark for HAT Java Sequential.",
    "com.arturskowronski.llama3babylon.hat.benchmark.HatJavaSequentialInferenceBenchmarkTest.benchmarkHatJavaSequential"
)

registerBenchmarkTestByPattern(
    "benchmarkInferenceHatMt",
    "Runs inference benchmark for HAT Java Multi-Threaded.",
    "com.arturskowronski.llama3babylon.hat.benchmark.HatJavaMtInferenceBenchmarkTest.benchmarkHatJavaMt"
)

registerBenchmarkTestByPattern(
    "benchmarkInferenceHatOpencl",
    "Runs inference benchmark for HAT OpenCL GPU.",
    "com.arturskowronski.llama3babylon.hat.benchmark.HatOpenclInferenceBenchmarkTest.benchmarkHatOpencl"
)

tasks.register("benchmarkInference") {
    description = "Runs all 4 inference benchmarks (Plain Java + HAT backends)."
    group = "verification"
    dependsOn(
        "benchmarkInferencePlainJava",
        "benchmarkInferenceHatSeq",
        "benchmarkInferenceHatMt",
        "benchmarkInferenceHatOpencl"
    )
}

tasks.register("benchmarkAll") {
    description = "Runs all benchmark suites used in CI."
    group = "verification"
    dependsOn("benchmarkDaily", "benchmarkMicroAll")
}

tasks.register("benchmarkKernelAll") {
    description = "Runs per-kernel benchmark comparisons for all kernels."
    group = "verification"
    dependsOn(
        "benchmarkKernelGEMV",
        "benchmarkKernelRMSNorm",
        "benchmarkKernelRoPE",
        "benchmarkKernelSiLU",
        "benchmarkKernelSoftmax",
        "benchmarkKernelAttention"
    )
}

tasks.register("benchmarkDaily") {
    description = "Runs the daily benchmark suite used in CI/nightly."
    group = "verification"
    dependsOn("benchmarkKernelAll")
}

tasks.register("benchmarkMicroAll") {
    description = "Runs all kernel micro-benchmarks (math-only)."
    group = "verification"
    dependsOn(
        "benchmarkMicroGEMV",
        "benchmarkMicroRMSNorm",
        "benchmarkMicroRoPE",
        "benchmarkMicroSiLU",
        "benchmarkMicroSoftmax",
        "benchmarkMicroAttention"
    )
}

tasks.register<Test>("regressionTest") {
    configureBaseTestTask("Runs regression-only tests.")

    useJUnitPlatform {
        includeTags("regression")
    }
    configureLongRunningJvm()
    configureVerboseTestLogging()

    System.getenv("LLAMA_FP16_PATH")?.let { environment("LLAMA_FP16_PATH", it) }
}

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
