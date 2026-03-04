package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("benchmark")
@Tag("benchmark-chat")
public class HatOpenclF32InferenceBenchmarkTest {

    @Test
    public void benchmarkHatOpenclF32() {
        if (!InferenceBenchmarkSupport.isModelAvailable()) {
            InferenceBenchmarkSupport.recordResult(InferenceBenchmarkSupport.skipped("HAT OpenCL GPU (F32)", "LLAMA_FP16_PATH is not set"));
            return;
        }
        if (!InferenceBenchmarkSupport.isOpenClEnabled()) {
            InferenceBenchmarkSupport.recordResult(InferenceBenchmarkSupport.skipped("HAT OpenCL GPU (F32)", "Set RUN_OPENCL_BENCHMARKS=true to enable"));
            return;
        }
        InferenceBenchmarkSupport.recordResult(
                InferenceBenchmarkSupport.runHat(InferenceBenchmarkSupport.modelPathFromEnv(), BackendType.OPENCL, WeightStorageMode.F32, "HAT OpenCL GPU (F32)"));
    }
}
