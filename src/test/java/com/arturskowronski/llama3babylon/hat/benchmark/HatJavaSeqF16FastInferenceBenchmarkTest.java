package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("benchmark")
@Tag("benchmark-chat")
public class HatJavaSeqF16FastInferenceBenchmarkTest {

    @Test
    public void benchmarkHatJavaSeqF16Fast() {
        if (!InferenceBenchmarkSupport.isModelAvailable()) {
            InferenceBenchmarkSupport.recordResult(InferenceBenchmarkSupport.skipped("HAT Java Sequential (F16_FAST)", "LLAMA_FP16_PATH is not set"));
            return;
        }
        InferenceBenchmarkSupport.recordResult(
                InferenceBenchmarkSupport.runHat(InferenceBenchmarkSupport.modelPathFromEnv(), BackendType.JAVA_SEQ, WeightStorageMode.F16_FAST, "HAT Java Sequential (F16_FAST)"));
    }
}
