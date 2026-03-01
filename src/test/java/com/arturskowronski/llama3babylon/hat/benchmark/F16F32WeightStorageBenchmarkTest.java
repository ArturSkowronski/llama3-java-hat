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
    public void benchmarkF16vsF32GPUOnly() {
        String model = System.getenv("LLAMA_FP16_PATH");
        if (model == null || model.isBlank()) {
            InferenceBenchmarkSupport.recordResult(
                    InferenceBenchmarkSupport.skipped("F16 vs F32 Weight Storage GPU", "LLAMA_FP16_PATH is not set"));
            return;
        }
        String runOpenCL = System.getenv("RUN_OPENCL_BENCHMARKS");
        if (runOpenCL == null || !runOpenCL.matches("(?i)true|1|yes")) {
            InferenceBenchmarkSupport.recordResult(
                    InferenceBenchmarkSupport.skipped("F16 vs F32 Weight Storage GPU", "Set RUN_OPENCL_BENCHMARKS=true"));
            return;
        }

        Path modelPath = InferenceBenchmarkSupport.modelPathFromEnv();

        List<InferenceBenchmarkSupport.BenchmarkResult> results = new ArrayList<>();
        results.add(InferenceBenchmarkSupport.runWeightModeHAT(
                modelPath, WeightStorageMode.F16, BackendType.OPENCL, "GPU HAT GEMV: F16"));
        results.add(InferenceBenchmarkSupport.runWeightModeHAT(
                modelPath, WeightStorageMode.F32, BackendType.OPENCL, "GPU HAT GEMV: F32"));

        InferenceBenchmarkSupport.recordResults(results);
    }

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

        // CPU — plain Java (no HAT dispatch)
        results.add(InferenceBenchmarkSupport.runWeightMode(
                modelPath, WeightStorageMode.F16, BackendType.JAVA_SEQ, "CPU Plain Java: F16"));
        results.add(InferenceBenchmarkSupport.runWeightMode(
                modelPath, WeightStorageMode.F32, BackendType.JAVA_SEQ, "CPU Plain Java: F32"));

        // CPU — HAT GEMV dispatch (Java Sequential backend)
        results.add(InferenceBenchmarkSupport.runWeightModeHAT(
                modelPath, WeightStorageMode.F16, BackendType.JAVA_SEQ, "CPU HAT GEMV: F16"));
        results.add(InferenceBenchmarkSupport.runWeightModeHAT(
                modelPath, WeightStorageMode.F32, BackendType.JAVA_SEQ, "CPU HAT GEMV: F32"));

        // GPU — HAT GEMV dispatch (OpenCL backend) — only if opted in
        String runOpenCL = System.getenv("RUN_OPENCL_BENCHMARKS");
        if (runOpenCL != null && runOpenCL.matches("(?i)true|1|yes")) {
            results.add(InferenceBenchmarkSupport.runWeightModeHAT(
                    modelPath, WeightStorageMode.F16, BackendType.OPENCL, "GPU HAT GEMV: F16"));
            results.add(InferenceBenchmarkSupport.runWeightModeHAT(
                    modelPath, WeightStorageMode.F32, BackendType.OPENCL, "GPU HAT GEMV: F32"));
        } else {
            results.add(InferenceBenchmarkSupport.skipped("GPU HAT GEMV: F16", "Set RUN_OPENCL_BENCHMARKS=true"));
            results.add(InferenceBenchmarkSupport.skipped("GPU HAT GEMV: F32", "Set RUN_OPENCL_BENCHMARKS=true"));
        }

        InferenceBenchmarkSupport.recordResults(results);
    }
}
