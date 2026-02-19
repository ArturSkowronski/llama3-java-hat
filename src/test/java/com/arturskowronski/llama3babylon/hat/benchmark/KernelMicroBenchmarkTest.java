package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.kernels.HybridKernelFactory;
import com.arturskowronski.llama3babylon.hat.kernels.IKernelFactory;
import com.arturskowronski.llama3babylon.hat.kernels.PlainJavaKernelFactory;
import hat.Accelerator;
import hat.buffer.F32Array;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

@Tag("benchmark")
public class KernelMicroBenchmarkTest {

    private static final int WARMUP_ITERS = parsePositiveInt(System.getenv("BENCHMARK_WARMUP_ITERS"), 3);
    private static final int ITERS = parsePositiveInt(System.getenv("BENCHMARK_ITERS"), 20);

    @Test
    public void benchmarkMicroGEMV() {
        runAcrossModes(HybridKernelFactory.KernelType.GEMV);
    }

    @Test
    public void benchmarkMicroRMSNorm() {
        runAcrossModes(HybridKernelFactory.KernelType.RMSNORM);
    }

    @Test
    public void benchmarkMicroRoPE() {
        runAcrossModes(HybridKernelFactory.KernelType.ROPE);
    }

    @Test
    public void benchmarkMicroSiLU() {
        runAcrossModes(HybridKernelFactory.KernelType.SILU);
    }

    @Test
    public void benchmarkMicroSoftmax() {
        runAcrossModes(HybridKernelFactory.KernelType.SOFTMAX);
    }

    @Test
    public void benchmarkMicroAttention() {
        runAcrossModes(HybridKernelFactory.KernelType.ATTENTION);
    }

    private void runAcrossModes(HybridKernelFactory.KernelType kernelType) {
        List<InferenceBenchmarkSupport.BenchmarkResult> results = new ArrayList<>();

        results.add(runMode("Plain Java", kernelType, null));
        results.add(runMode("HAT Java Sequential", kernelType, BackendType.JAVA_SEQ));
        results.add(runMode("HAT Java MT", kernelType, BackendType.JAVA_MT));

        String runOpenCL = System.getenv("RUN_OPENCL_BENCHMARKS");
        if (runOpenCL != null && runOpenCL.matches("(?i)true|1|yes")) {
            results.add(runMode("HAT OpenCL", kernelType, BackendType.OPENCL));
        } else {
            results.add(InferenceBenchmarkSupport.skipped(
                    "Kernel Micro " + kernelType + " | HAT OpenCL",
                    "Set RUN_OPENCL_BENCHMARKS=true to enable"
            ));
        }

        InferenceBenchmarkSupport.recordResults(results);
    }

    private InferenceBenchmarkSupport.BenchmarkResult runMode(
            String mode,
            HybridKernelFactory.KernelType kernelType,
            BackendType backendType
    ) {
        String name = "Kernel Micro " + kernelType + " | " + mode;
        try {
            Accelerator acc;
            IKernelFactory factory;
            if (backendType == null) {
                acc = new Accelerator(MethodHandles.lookup(), BackendType.JAVA_SEQ.predicate());
                factory = new PlainJavaKernelFactory();
            } else {
                acc = new Accelerator(MethodHandles.lookup(), backendType.predicate());
                factory = new HybridKernelFactory(Set.of(kernelType));
            }

            Runnable op = createOperation(acc, factory, kernelType);

            for (int i = 0; i < WARMUP_ITERS; i++) {
                op.run();
            }

            long startNs = System.nanoTime();
            for (int i = 0; i < ITERS; i++) {
                op.run();
            }
            long endNs = System.nanoTime();

            double elapsedSec = (endNs - startNs) / 1_000_000_000.0;
            double opsPerSec = ITERS / elapsedSec;
            return new InferenceBenchmarkSupport.BenchmarkResult(name, 0.0, elapsedSec, opsPerSec, null);
        } catch (Throwable t) {
            return new InferenceBenchmarkSupport.BenchmarkResult(name, 0.0, -1.0, -1.0, t.getClass().getSimpleName());
        }
    }

    private Runnable createOperation(Accelerator acc, IKernelFactory factory, HybridKernelFactory.KernelType kernelType) {
        return switch (kernelType) {
            case GEMV -> {
                int rows = parsePositiveInt(System.getenv("BENCHMARK_GEMV_ROWS"), 512);
                int cols = parsePositiveInt(System.getenv("BENCHMARK_GEMV_COLS"), 512);

                F32Array matrix = F32Array.create(acc, rows * cols);
                F32Array vector = F32Array.create(acc, cols);
                F32Array result = F32Array.create(acc, rows);
                fillLinear(matrix, 0.001f, 1.0f);
                fillLinear(vector, 0.002f, 0.5f);

                var kernel = factory.createGEMV(acc);
                yield () -> kernel.apply(matrix, vector, result, rows, cols);
            }
            case RMSNORM -> {
                int size = parsePositiveInt(System.getenv("BENCHMARK_RMSNORM_SIZE"), 2048);

                F32Array input = F32Array.create(acc, size);
                F32Array weight = F32Array.create(acc, size);
                fillLinear(input, 0.001f, 0.25f);
                fillLinear(weight, 0.0005f, 1.0f);

                var kernel = factory.createRMSNorm(acc);
                yield () -> kernel.apply(input, weight, size);
            }
            case ROPE -> {
                int numHeads = parsePositiveInt(System.getenv("BENCHMARK_ROPE_HEADS"), 32);
                int headDim = parsePositiveInt(System.getenv("BENCHMARK_ROPE_HEAD_DIM"), 64);
                int size = numHeads * headDim;
                int pos = parsePositiveInt(System.getenv("BENCHMARK_ROPE_POS"), 128);

                F32Array vec = F32Array.create(acc, size);
                fillLinear(vec, 0.001f, 0.1f);

                var kernel = factory.createRoPE(acc);
                yield () -> kernel.apply(vec, pos, numHeads, headDim, 500000.0f);
            }
            case SILU -> {
                int size = parsePositiveInt(System.getenv("BENCHMARK_SILU_SIZE"), 8192);

                F32Array input = F32Array.create(acc, size);
                fillLinear(input, 0.001f, -2.0f);

                var kernel = factory.createSiLU(acc);
                yield () -> kernel.apply(input, size);
            }
            case SOFTMAX -> {
                int size = parsePositiveInt(System.getenv("BENCHMARK_SOFTMAX_SIZE"), 2048);

                F32Array input = F32Array.create(acc, size);
                fillLinear(input, 0.001f, -1.0f);

                var kernel = factory.createSoftmax(acc);
                yield () -> kernel.apply(input, size);
            }
            case ATTENTION -> {
                int seqLen = parsePositiveInt(System.getenv("BENCHMARK_ATTN_SEQ_LEN"), 256);
                int headDim = parsePositiveInt(System.getenv("BENCHMARK_ATTN_HEAD_DIM"), 64);

                F32Array query = F32Array.create(acc, headDim);
                F32Array keys = F32Array.create(acc, seqLen * headDim);
                F32Array scores = F32Array.create(acc, seqLen);
                F32Array values = F32Array.create(acc, seqLen * headDim);
                F32Array output = F32Array.create(acc, headDim);

                fillLinear(query, 0.001f, 0.1f);
                fillLinear(keys, 0.001f, 0.2f);
                fillLinear(values, 0.001f, 0.3f);

                var kernel = factory.createAttention(acc);
                yield () -> {
                    kernel.computeScores(query, keys, scores, seqLen, headDim);
                    kernel.computeValues(scores, values, output, seqLen, headDim);
                };
            }
        };
    }

    private static void fillLinear(F32Array array, float step, float base) {
        for (int i = 0; i < array.length(); i++) {
            array.array(i, base + i * step);
        }
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
}
