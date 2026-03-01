package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.LlamaInference;
import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import com.arturskowronski.llama3babylon.hat.kernels.HybridKernelFactory;
import com.arturskowronski.llama3babylon.hat.kernels.PlainJavaKernelFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

final class InferenceBenchmarkSupport {
    private InferenceBenchmarkSupport() {}

    static final String SYSTEM_PROMPT = "You are a helpful assistant.";
    static final String USER_PROMPT = System.getenv().getOrDefault("BENCHMARK_USER_PROMPT", "Say hi");
    static final int MAX_TOKENS = parsePositiveInt(System.getenv("BENCHMARK_MAX_TOKENS"), 8);

    private static final Path RESULTS_DIR = Paths.get(System.getProperty("user.dir"), "build", "benchmark-results");
    private static final Path RESULTS_TSV = RESULTS_DIR.resolve("results.tsv");
    private static final ConcurrentMap<String, BenchmarkResult> PLAIN_BASELINE_CACHE = new ConcurrentHashMap<>();

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

    static BenchmarkResult runWeightMode(Path modelPath, WeightStorageMode mode, String label) {
        return runBenchmark(label, () -> new LlamaInference(
                modelPath, new PlainJavaKernelFactory(), BackendType.JAVA_SEQ, mode));
    }

    static BenchmarkResult runPlainJavaCached(Path modelPath) {
        String key = modelPath.toAbsolutePath() + "|tokens=" + MAX_TOKENS + "|prompt=" + USER_PROMPT;
        return PLAIN_BASELINE_CACHE.computeIfAbsent(key, ignored -> runPlainJava(modelPath));
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
                runPlainJavaCached(modelPath),
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
        } catch (java.util.ServiceConfigurationError e) {
            String msg = "No runtime available";
            System.out.println(">>> SKIPPED " + name + ": " + msg + " (" + e.getMessage() + ")");
            return new BenchmarkResult(name, -1, -1, -1, "SKIPPED: " + msg);
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
        boolean microBenchmark = results.stream().allMatch(r -> r.name != null && r.name.startsWith("Kernel Micro "));
        String throughputLabel = microBenchmark ? "Ops/sec" : "Tokens/sec";
        int backendWidth = 44;
        int loadWidth = 10;
        int inferWidth = 13;
        int thrWidth = 15;

        System.out.println("\n");
        System.out.println("╔" + "═".repeat(backendWidth + 2) + "╦" + "═".repeat(loadWidth + 2) + "╦" + "═".repeat(inferWidth + 2) + "╦" + "═".repeat(thrWidth + 2) + "╗");
        System.out.printf("║ %-" + backendWidth + "s ║ %-" + loadWidth + "s ║ %-" + inferWidth + "s ║ %-" + thrWidth + "s ║%n",
                "Backend", "Model Load", "Inference", throughputLabel);
        System.out.println("╠" + "═".repeat(backendWidth + 2) + "╬" + "═".repeat(loadWidth + 2) + "╬" + "═".repeat(inferWidth + 2) + "╬" + "═".repeat(thrWidth + 2) + "╣");

        for (BenchmarkResult r : results) {
            String backend = fit(r.name, backendWidth);
            if (r.error != null) {
                System.out.printf("║ %-" + backendWidth + "s ║ %-" + loadWidth + "s ║ %-" + inferWidth + "s ║ %-" + thrWidth + "s ║%n",
                        backend,
                        r.loadTimeSec >= 0 ? String.format("%.2fs", r.loadTimeSec) : "FAILED",
                        "FAILED",
                        fit(r.error, thrWidth));
            } else {
                String throughput = microBenchmark
                        ? String.format("%.2f ops/s", r.tokPerSec)
                        : String.format("%.2f tok/s", r.tokPerSec);
                System.out.printf("║ %-" + backendWidth + "s ║ %" + loadWidth + "s ║ %" + inferWidth + "s ║ %" + thrWidth + "s ║%n",
                        backend,
                        String.format("%.2fs", r.loadTimeSec),
                        String.format("%.2fs", r.inferTimeSec),
                        throughput);
            }
        }

        System.out.println("╚" + "═".repeat(backendWidth + 2) + "╩" + "═".repeat(loadWidth + 2) + "╩" + "═".repeat(inferWidth + 2) + "╩" + "═".repeat(thrWidth + 2) + "╝");
        if (!microBenchmark) {
            System.out.println("  Prompt: \"" + USER_PROMPT + "\" | Max tokens: " + MAX_TOKENS);
        }
        System.out.println();
    }

    private static String fit(String value, int width) {
        if (value == null) return "";
        if (value.length() <= width) return value;
        return value.substring(0, Math.max(0, width - 3)) + "...";
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

    private static int parsePositiveInt(String raw, int fallback) {
        if (raw == null || raw.isBlank()) return fallback;
        try {
            int value = Integer.parseInt(raw.trim());
            return value > 0 ? value : fallback;
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    @FunctionalInterface
    interface InferenceSupplier {
        LlamaInference create() throws Exception;
    }
    record BenchmarkResult(String name, double loadTimeSec, double inferTimeSec, double tokPerSec, String error) {}
}
