package com.arturskowronski.llama3babylon.hat.integration.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
@Tag("benchmark")
@Tag("benchmark-hat-seq")
public class HatJavaSequentialInferenceBenchmarkTest {

    @Test
    public void benchmarkHatJavaSequential() {
        String model = System.getenv("LLAMA_FP16_PATH");
        if (model == null || model.isBlank()) {
            InferenceBenchmarkSupport.recordResult(
                    InferenceBenchmarkSupport.skipped("HAT Java Sequential", "LLAMA_FP16_PATH is not set"));
            return;
        }

        var result = InferenceBenchmarkSupport.runHat(InferenceBenchmarkSupport.modelPathFromEnv(), BackendType.JAVA_SEQ, "HAT Java Sequential");
        InferenceBenchmarkSupport.recordResult(result);
        InferenceBenchmarkSupport.gcPause();
    }
}
