package com.arturskowronski.llama3babylon.hat.integration.benchmark;

import com.arturskowronski.llama3babylon.hat.kernels.HybridKernelFactory;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
@Tag("benchmark")
@Tag("benchmark-kernel")
public class KernelModeInferenceBenchmarkTest {

    @Test
    @Tag("benchmark-kernel-gemv")
    public void benchmarkGEMVAcrossModes() {
        benchmarkKernelAcrossModes(HybridKernelFactory.KernelType.GEMV);
    }

    @Test
    @Tag("benchmark-kernel-rmsnorm")
    public void benchmarkRMSNormAcrossModes() {
        benchmarkKernelAcrossModes(HybridKernelFactory.KernelType.RMSNORM);
    }

    @Test
    @Tag("benchmark-kernel-rope")
    public void benchmarkRoPEAcrossModes() {
        benchmarkKernelAcrossModes(HybridKernelFactory.KernelType.ROPE);
    }

    @Test
    @Tag("benchmark-kernel-silu")
    public void benchmarkSiLUAcrossModes() {
        benchmarkKernelAcrossModes(HybridKernelFactory.KernelType.SILU);
    }

    @Test
    @Tag("benchmark-kernel-softmax")
    public void benchmarkSoftmaxAcrossModes() {
        benchmarkKernelAcrossModes(HybridKernelFactory.KernelType.SOFTMAX);
    }

    @Test
    @Tag("benchmark-kernel-attention")
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
        InferenceBenchmarkSupport.gcPause();
    }
}
