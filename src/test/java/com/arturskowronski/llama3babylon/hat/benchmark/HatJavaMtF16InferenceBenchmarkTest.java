package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("benchmark")
@Tag("benchmark-chat")
public class HatJavaMtF16InferenceBenchmarkTest {

    @Test
    public void benchmarkHatJavaMtF16() {
        if (!InferenceBenchmarkSupport.isModelAvailable()) {
            InferenceBenchmarkSupport.recordResult(InferenceBenchmarkSupport.skipped("HAT Java MT (F16)", "LLAMA_FP16_PATH is not set"));
            return;
        }
        InferenceBenchmarkSupport.recordResult(
                InferenceBenchmarkSupport.runHat(InferenceBenchmarkSupport.modelPathFromEnv(), BackendType.JAVA_MT, WeightStorageMode.F16, "HAT Java MT (F16)"));
    }
}
