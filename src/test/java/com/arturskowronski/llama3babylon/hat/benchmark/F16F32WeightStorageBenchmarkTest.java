package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

@Tag("benchmark")
@Tag("benchmark-chat")
public class F16F32WeightStorageBenchmarkTest {

    @Test
    public void benchmarkF16vsF32WeightStorage() {
        String model = System.getenv("LLAMA_FP16_PATH");
        if (model == null || model.isBlank()) {
            InferenceBenchmarkSupport.recordResult(
                    InferenceBenchmarkSupport.skipped("F16 vs F32 Weight Storage", "LLAMA_FP16_PATH is not set"));
            return;
        }

        Path modelPath = InferenceBenchmarkSupport.modelPathFromEnv();

        List<InferenceBenchmarkSupport.BenchmarkResult> results = new ArrayList<>();

        // CPU (Java Sequential)
        results.add(InferenceBenchmarkSupport.runWeightMode(
                modelPath, WeightStorageMode.F16, BackendType.JAVA_SEQ, "CPU Java Seq: F16 (native)"));
        results.add(InferenceBenchmarkSupport.runWeightMode(
                modelPath, WeightStorageMode.F32, BackendType.JAVA_SEQ, "CPU Java Seq: F32 (eager dequant)"));

        // GPU (OpenCL) — only if opted in
        String runOpenCL = System.getenv("RUN_OPENCL_BENCHMARKS");
        if (runOpenCL != null && runOpenCL.matches("(?i)true|1|yes")) {
            results.add(InferenceBenchmarkSupport.runWeightMode(
                    modelPath, WeightStorageMode.F16, BackendType.OPENCL, "GPU OpenCL: F16 (native)"));
            results.add(InferenceBenchmarkSupport.runWeightMode(
                    modelPath, WeightStorageMode.F32, BackendType.OPENCL, "GPU OpenCL: F32 (eager dequant)"));
        } else {
            results.add(InferenceBenchmarkSupport.skipped("GPU OpenCL: F16", "Set RUN_OPENCL_BENCHMARKS=true"));
            results.add(InferenceBenchmarkSupport.skipped("GPU OpenCL: F32", "Set RUN_OPENCL_BENCHMARKS=true"));
        }

        InferenceBenchmarkSupport.recordResults(results);
    }
}
