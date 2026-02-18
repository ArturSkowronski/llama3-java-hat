package com.arturskowronski.llama3babylon.hat.integration.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.LlamaInference;
import com.arturskowronski.llama3babylon.hat.kernels.HybridKernelFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.List;
import java.util.Set;

final class InferenceBenchmarkSupport {
    private InferenceBenchmarkSupport() {}

    static final String SYSTEM_PROMPT = "You are a helpful assistant.";
    static final String USER_PROMPT = "Tell a joke about programming";
    static final int MAX_TOKENS = 32;

    private static final Path RESULTS_DIR = Paths.get(System.getProperty("user.dir"), "build", "benchmark-results");
    private static final Path RESULTS_TSV = RESULTS_DIR.resolve("results.tsv");

    private static final Set<HybridKernelFactory.KernelType> ALL_KERNELS = Set.of(
            HybridKernelFactory.KernelType.GEMV,
            HybridKernelFactory.KernelType.RMSNORM,
            HybridKernelFactory.KernelType.ROPE,
            HybridKernelFactory.KernelType.SILU,
            HybridKernelFactory.KernelType.SOFTMAX,
            HybridKernelFactory.KernelType.ATTENTION
    );

    static Path modelPathFromEnv() {
        String p = System.getenv("LLAMA_FP16_PATH");
        if (p == null || p.isBlank()) {
            throw new IllegalStateException("LLAMA_FP16_PATH is not set");
        }
        return Paths.get(p);
    }

    static BenchmarkResult skipped(String name, String reason) {
        return new BenchmarkResult(name, -1, -1, -1, "SKIPPED: " + sanitize(reason));
    }

    static BenchmarkResult runPlainJava(Path modelPath) {
        return runBenchmark("Plain Java", () -> new LlamaInference(modelPath));
    }

    static BenchmarkResult runHat(Path modelPath, BackendType backendType, String label) {
        return runBenchmark(label, () -> new LlamaInference(modelPath, new HybridKernelFactory(ALL_KERNELS), backendType));
    }

    static BenchmarkResult runHatSingleKernel(Path modelPath, BackendType backendType, HybridKernelFactory.KernelType kernel, String label) {
        return runBenchmark(label, () -> new LlamaInference(modelPath, new HybridKernelFactory(Set.of(kernel)), backendType));
    }

    static List<BenchmarkResult> runKernelModeComparison(Path modelPath, HybridKernelFactory.KernelType kernel) {
        String kernelName = kernel.name();
        return List.of(
                runPlainJava(modelPath),
                runHatSingleKernel(modelPath, BackendType.JAVA_SEQ, kernel, "HAT Java Sequential (" + kernelName + ")"),
                runHatSingleKernel(modelPath, BackendType.JAVA_MT, kernel, "HAT Java MT (" + kernelName + ")"),
                runOpenclSingleKernel(modelPath, kernel, "HAT OpenCL GPU (" + kernelName + ")")
        );
    }

    static BenchmarkResult runOpenclSingleKernel(Path modelPath, HybridKernelFactory.KernelType kernel, String label) {
        String run = System.getenv("RUN_OPENCL_BENCHMARKS");
        if (run == null || !run.matches("(?i)true|1|yes")) {
            return skipped(label, "Set RUN_OPENCL_BENCHMARKS=true to enable");
        }
        return runHatSingleKernel(modelPath, BackendType.OPENCL, kernel, label);
    }

    static BenchmarkResult runBenchmark(String name, InferenceSupplier supplier) {
        System.out.println("\n>>> Starting benchmark: " + name);
        System.out.flush();

        long loadStart = System.nanoTime();
        LlamaInference inference;
        try {
            inference = supplier.create();
        } catch (Throwable e) {
            System.out.println(">>> FAILED to initialize " + name + ": " + e);
            return new BenchmarkResult(name, -1, -1, -1, e.getClass().getSimpleName());
        }
        long loadEnd = System.nanoTime();
        double loadTimeSec = (loadEnd - loadStart) / 1_000_000_000.0;

        long inferStart = System.nanoTime();
        String response;
        try {
            response = inference.chat(SYSTEM_PROMPT, USER_PROMPT, MAX_TOKENS);
        } catch (Throwable e) {
            System.out.println(">>> FAILED during inference " + name + ": " + e);
            return new BenchmarkResult(name, loadTimeSec, -1, -1, e.getClass().getSimpleName());
        }
        long inferEnd = System.nanoTime();
        double inferTimeSec = (inferEnd - inferStart) / 1_000_000_000.0;

        // Upper bound; generation may stop earlier (EOS).
        int tokenCount = MAX_TOKENS;
        double tokPerSec = tokenCount / inferTimeSec;

        System.out.println(">>> " + name + " response: " + response.substring(0, Math.min(80, response.length())) + "...");
        System.out.flush();

        return new BenchmarkResult(name, loadTimeSec, inferTimeSec, tokPerSec, null);
    }

    static void recordResult(BenchmarkResult r) {
        // Always print; file output is best-effort.
        printResults(List.of(r));
        writeTsvLine(r);
        System.out.println(">>> Benchmark results TSV: " + RESULTS_TSV);
    }

    static void recordResults(List<BenchmarkResult> results) {
        printResults(results);
        for (BenchmarkResult result : results) {
            writeTsvLine(result);
        }
        System.out.println(">>> Benchmark results TSV: " + RESULTS_TSV);
    }

    static void printResults(List<BenchmarkResult> results) {
        System.out.println("\n");
        System.out.println("╔══════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                    INFERENCE BENCHMARK RESULTS                      ║");
        System.out.println("╠══════════════════════╦════════════╦═══════════════╦═════════════════╣");
        System.out.println("║ Backend              ║ Model Load ║ Inference     ║ Tokens/sec      ║");
        System.out.println("╠══════════════════════╬════════════╬═══════════════╬═════════════════╣");

        for (BenchmarkResult r : results) {
            if (r.error != null) {
                System.out.printf("║ %-20s ║ %-10s ║ %-13s ║ %-15s ║%n",
                        r.name,
                        r.loadTimeSec >= 0 ? String.format("%.2fs", r.loadTimeSec) : "FAILED",
                        "FAILED",
                        r.error.length() > 15 ? r.error.substring(0, 12) + "..." : r.error);
            } else {
                System.out.printf("║ %-20s ║ %8.2fs  ║ %10.2fs   ║ %10.2f tok/s ║%n",
                        r.name, r.loadTimeSec, r.inferTimeSec, r.tokPerSec);
            }
        }

        System.out.println("╚══════════════════════╩════════════╩═══════════════╩═════════════════╝");
        System.out.println("  Prompt: \"" + USER_PROMPT + "\" | Max tokens: " + MAX_TOKENS);
        System.out.println();
    }

    private static void writeTsvLine(BenchmarkResult r) {
        try {
            Files.createDirectories(RESULTS_DIR);
            boolean needsHeader = Files.notExists(RESULTS_TSV);
            String ts = Instant.now().toString();
            String line = String.join("\t",
                    ts,
                    sanitize(r.name),
                    Double.toString(r.loadTimeSec),
                    Double.toString(r.inferTimeSec),
                    Double.toString(r.tokPerSec),
                    r.error == null ? "" : sanitize(r.error)
            ) + "\n";

            if (needsHeader) {
                String header = "timestamp\tbackend\tload_sec\tinfer_sec\ttok_per_sec\terror\n";
                Files.writeString(RESULTS_TSV, header, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            }
            Files.writeString(RESULTS_TSV, line, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            System.out.println(">>> WARN: failed to write benchmark results to " + RESULTS_TSV + ": " + e);
        }
    }

    private static String sanitize(String s) {
        if (s == null) return "";
        // Keep TSV single-line.
        return s.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ');
    }

    @FunctionalInterface
    interface InferenceSupplier {
        LlamaInference create() throws Exception;
    }

    static void gcPause() {
        System.gc();
        try {
            Thread.sleep(500);
        } catch (InterruptedException ignored) {
        }
        System.gc();
    }

    record BenchmarkResult(String name, double loadTimeSec, double inferTimeSec, double tokPerSec, String error) {}
}
