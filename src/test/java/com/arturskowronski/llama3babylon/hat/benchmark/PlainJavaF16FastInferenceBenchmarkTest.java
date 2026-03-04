package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("benchmark")
@Tag("benchmark-chat")
public class PlainJavaF16FastInferenceBenchmarkTest {

    @Test
    public void benchmarkPlainJavaF16Fast() {
        if (!InferenceBenchmarkSupport.isModelAvailable()) {
            InferenceBenchmarkSupport.recordResult(InferenceBenchmarkSupport.skipped("Plain Java (F16_FAST)", "LLAMA_FP16_PATH is not set"));
            return;
        }
        InferenceBenchmarkSupport.recordResult(
                InferenceBenchmarkSupport.runWeightMode(InferenceBenchmarkSupport.modelPathFromEnv(), WeightStorageMode.F16_FAST, BackendType.JAVA_SEQ, "Plain Java (F16_FAST)"));
    }
}
