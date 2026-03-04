package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("benchmark")
@Tag("benchmark-chat")
public class PlainJavaF16InferenceBenchmarkTest {

    @Test
    public void benchmarkPlainJavaF16() {
        if (!InferenceBenchmarkSupport.isModelAvailable()) {
            InferenceBenchmarkSupport.recordResult(InferenceBenchmarkSupport.skipped("Plain Java (F16)", "LLAMA_FP16_PATH is not set"));
            return;
        }
        InferenceBenchmarkSupport.recordResult(
                InferenceBenchmarkSupport.runWeightMode(InferenceBenchmarkSupport.modelPathFromEnv(), WeightStorageMode.F16, BackendType.JAVA_SEQ, "Plain Java (F16)"));
    }
}
