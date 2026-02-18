package com.arturskowronski.llama3babylon.hat.integration.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.LlamaInference;
import com.arturskowronski.llama3babylon.hat.kernels.HybridKernelFactory;
import com.arturskowronski.llama3babylon.hat.kernels.IKernelFactory;
import com.arturskowronski.llama3babylon.hat.kernels.PlainJavaKernelFactory;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Benchmark comparing all four inference backends:
 * 1. Plain Java (no HAT)
 * 2. HAT Java Sequential (single-threaded @Reflect dispatch)
 * 3. HAT Java Multi-Threaded (parallel @Reflect dispatch)
 * 4. HAT OpenCL (GPU via FFI)
 *
 * Each backend runs the same chat prompt and reports:
 * - Total wall-clock time (model load + inference)
 * - Inference-only time (excluding model load)
 * - Tokens per second
 */
@Tag("integration")
@Tag("benchmark")
public class InferenceBenchmark {

    private static final String SYSTEM_PROMPT = "You are a helpful assistant.";
    private static final String USER_PROMPT = "Tell a joke about programming";
    private static final int MAX_TOKENS = 32;

    private static final Set<HybridKernelFactory.KernelType> ALL_KERNELS = Set.of(
            HybridKernelFactory.KernelType.GEMV,
            HybridKernelFactory.KernelType.RMSNORM,
            HybridKernelFactory.KernelType.ROPE,
            HybridKernelFactory.KernelType.SILU,
            HybridKernelFactory.KernelType.SOFTMAX,
            HybridKernelFactory.KernelType.ATTENTION
    );

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void benchmarkAllBackends() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));

        List<BenchmarkResult> results = new ArrayList<>();

        // 1. Plain Java
        results.add(runBenchmark("Plain Java", modelPath,
                () -> new LlamaInference(modelPath)));

        // 2. HAT Java Sequential
        results.add(runBenchmark("HAT Java Sequential", modelPath,
                () -> new LlamaInference(modelPath, new HybridKernelFactory(ALL_KERNELS), BackendType.JAVA_SEQ)));

        // 3. HAT Java Multi-Threaded
        results.add(runBenchmark("HAT Java MT", modelPath,
                () -> new LlamaInference(modelPath, new HybridKernelFactory(ALL_KERNELS), BackendType.JAVA_MT)));

        // 4. HAT OpenCL GPU
        results.add(runBenchmark("HAT OpenCL GPU", modelPath,
                () -> new LlamaInference(modelPath, new HybridKernelFactory(ALL_KERNELS), BackendType.OPENCL)));

        printResults(results);
    }

    private BenchmarkResult runBenchmark(String name, Path modelPath, InferenceSupplier supplier) {
        System.out.println("\n>>> Starting benchmark: " + name);
        System.out.flush();

        long loadStart = System.nanoTime();
        LlamaInference inference;
        try {
            inference = supplier.create();
        } catch (Throwable e) {
            System.out.println(">>> FAILED to initialize " + name + ": " + e.getMessage());
            return new BenchmarkResult(name, -1, -1, -1, e.getClass().getSimpleName());
        }
        long loadEnd = System.nanoTime();
        double loadTimeSec = (loadEnd - loadStart) / 1_000_000_000.0;

        long inferStart = System.nanoTime();
        String response;
        try {
            response = inference.chat(SYSTEM_PROMPT, USER_PROMPT, MAX_TOKENS);
        } catch (Throwable e) {
            System.out.println(">>> FAILED during inference " + name + ": " + e.getMessage());
            return new BenchmarkResult(name, loadTimeSec, -1, -1, e.getClass().getSimpleName());
        }
        long inferEnd = System.nanoTime();
        double inferTimeSec = (inferEnd - inferStart) / 1_000_000_000.0;

        // Count generated tokens (approximate from response words — actual token count is MAX_TOKENS or less)
        int tokenCount = MAX_TOKENS; // upper bound, generation stops at EOS or max
        double tokPerSec = tokenCount / inferTimeSec;

        System.out.println(">>> " + name + " response: " + response.substring(0, Math.min(80, response.length())) + "...");
        System.out.flush();

        return new BenchmarkResult(name, loadTimeSec, inferTimeSec, tokPerSec, null);
    }

    private void printResults(List<BenchmarkResult> results) {
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

    @FunctionalInterface
    private interface InferenceSupplier {
        LlamaInference create() throws Exception;
    }

    private record BenchmarkResult(String name, double loadTimeSec, double inferTimeSec, double tokPerSec, String error) {}
}
