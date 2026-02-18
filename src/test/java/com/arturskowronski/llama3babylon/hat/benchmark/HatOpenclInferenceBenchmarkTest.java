package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("benchmark")
public class HatOpenclInferenceBenchmarkTest {

    @Test
    public void benchmarkHatOpencl() {
        String model = System.getenv("LLAMA_FP16_PATH");
        if (model == null || model.isBlank()) {
            InferenceBenchmarkSupport.recordResult(
                    InferenceBenchmarkSupport.skipped("HAT OpenCL GPU", "LLAMA_FP16_PATH is not set"));
            return;
        }

        // OpenCL can be flaky/hard-failing on some hosts; require explicit opt-in.
        String run = System.getenv("RUN_OPENCL_BENCHMARKS");
        if (run == null || !run.matches("(?i)true|1|yes")) {
            InferenceBenchmarkSupport.recordResult(
                    InferenceBenchmarkSupport.skipped("HAT OpenCL GPU", "Set RUN_OPENCL_BENCHMARKS=true to enable"));
            return;
        }

        var result = InferenceBenchmarkSupport.runHat(InferenceBenchmarkSupport.modelPathFromEnv(), BackendType.OPENCL, "HAT OpenCL GPU");
        InferenceBenchmarkSupport.recordResult(result);
    }
}
