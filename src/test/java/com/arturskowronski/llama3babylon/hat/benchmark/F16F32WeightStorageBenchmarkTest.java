package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
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

        var results = List.of(
                InferenceBenchmarkSupport.runWeightMode(modelPath, WeightStorageMode.F16, "Weight Storage: F16 (native)"),
                InferenceBenchmarkSupport.runWeightMode(modelPath, WeightStorageMode.F32, "Weight Storage: F32 (eager dequant)")
        );

        InferenceBenchmarkSupport.recordResults(results);
    }
}
