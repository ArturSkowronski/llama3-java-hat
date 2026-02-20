package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("benchmark")
@Tag("benchmark-chat")
public class HatJavaMtInferenceBenchmarkTest {

    @Test
    public void benchmarkHatJavaMt() {
        String model = System.getenv("LLAMA_FP16_PATH");
        if (model == null || model.isBlank()) {
            InferenceBenchmarkSupport.recordResult(
                    InferenceBenchmarkSupport.skipped("HAT Java MT", "LLAMA_FP16_PATH is not set"));
            return;
        }

        var result = InferenceBenchmarkSupport.runHat(InferenceBenchmarkSupport.modelPathFromEnv(), BackendType.JAVA_MT, "HAT Java MT");
        InferenceBenchmarkSupport.recordResult(result);
    }
}
