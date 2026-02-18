package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.kernels.HybridKernelFactory;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("benchmark")
public class KernelModeInferenceBenchmarkTest {

    @Test
    public void benchmarkGEMVAcrossModes() {
        benchmarkKernelAcrossModes(HybridKernelFactory.KernelType.GEMV);
    }

    @Test
    public void benchmarkRMSNormAcrossModes() {
        benchmarkKernelAcrossModes(HybridKernelFactory.KernelType.RMSNORM);
    }

    @Test
    public void benchmarkRoPEAcrossModes() {
        benchmarkKernelAcrossModes(HybridKernelFactory.KernelType.ROPE);
    }

    @Test
    public void benchmarkSiLUAcrossModes() {
        benchmarkKernelAcrossModes(HybridKernelFactory.KernelType.SILU);
    }

    @Test
    public void benchmarkSoftmaxAcrossModes() {
        benchmarkKernelAcrossModes(HybridKernelFactory.KernelType.SOFTMAX);
    }

    @Test
    public void benchmarkAttentionAcrossModes() {
        benchmarkKernelAcrossModes(HybridKernelFactory.KernelType.ATTENTION);
    }

    private void benchmarkKernelAcrossModes(HybridKernelFactory.KernelType kernel) {
        String model = System.getenv("LLAMA_FP16_PATH");
        if (model == null || model.isBlank()) {
            InferenceBenchmarkSupport.recordResult(
                    InferenceBenchmarkSupport.skipped("Kernel " + kernel.name(), "LLAMA_FP16_PATH is not set"));
            return;
        }

        var results = InferenceBenchmarkSupport.runKernelModeComparison(InferenceBenchmarkSupport.modelPathFromEnv(), kernel);
        InferenceBenchmarkSupport.recordResults(results);
    }
}
