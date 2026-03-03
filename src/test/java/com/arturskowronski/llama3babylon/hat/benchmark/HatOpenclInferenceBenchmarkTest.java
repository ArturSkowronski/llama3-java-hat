package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.List;

@Tag("benchmark")
@Tag("benchmark-chat")
public class HatOpenclInferenceBenchmarkTest {

    @Test
    public void benchmarkHatOpencl() {
        String model = System.getenv("LLAMA_FP16_PATH");
        if (model == null || model.isBlank()) {
            InferenceBenchmarkSupport.recordResults(List.of(
                    InferenceBenchmarkSupport.skipped("HAT OpenCL GPU (F16)", "LLAMA_FP16_PATH is not set"),
                    InferenceBenchmarkSupport.skipped("HAT OpenCL GPU (F16_FAST)", "LLAMA_FP16_PATH is not set"),
                    InferenceBenchmarkSupport.skipped("HAT OpenCL GPU (F32)", "LLAMA_FP16_PATH is not set")));
            return;
        }

        String run = System.getenv("RUN_OPENCL_BENCHMARKS");
        if (run == null || !run.matches("(?i)true|1|yes")) {
            InferenceBenchmarkSupport.recordResults(List.of(
                    InferenceBenchmarkSupport.skipped("HAT OpenCL GPU (F16)", "Set RUN_OPENCL_BENCHMARKS=true to enable"),
                    InferenceBenchmarkSupport.skipped("HAT OpenCL GPU (F16_FAST)", "Set RUN_OPENCL_BENCHMARKS=true to enable"),
                    InferenceBenchmarkSupport.skipped("HAT OpenCL GPU (F32)", "Set RUN_OPENCL_BENCHMARKS=true to enable")));
            return;
        }

        var path = InferenceBenchmarkSupport.modelPathFromEnv();
        InferenceBenchmarkSupport.recordResults(List.of(
                InferenceBenchmarkSupport.runHat(path, BackendType.OPENCL, WeightStorageMode.F16, "HAT OpenCL GPU (F16)"),
                InferenceBenchmarkSupport.runHat(path, BackendType.OPENCL, WeightStorageMode.F16_FAST, "HAT OpenCL GPU (F16_FAST)"),
                InferenceBenchmarkSupport.runHat(path, BackendType.OPENCL, WeightStorageMode.F32, "HAT OpenCL GPU (F32)")
        ));
    }
}
