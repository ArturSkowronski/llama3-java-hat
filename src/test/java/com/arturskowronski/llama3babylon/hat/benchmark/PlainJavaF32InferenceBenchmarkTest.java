package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("benchmark")
@Tag("benchmark-chat")
public class PlainJavaF32InferenceBenchmarkTest {

    @Test
    public void benchmarkPlainJavaF32() {
        if (!InferenceBenchmarkSupport.isModelAvailable()) {
            InferenceBenchmarkSupport.recordResult(InferenceBenchmarkSupport.skipped("Plain Java (F32)", "LLAMA_FP16_PATH is not set"));
            return;
        }
        InferenceBenchmarkSupport.recordResult(
                InferenceBenchmarkSupport.runWeightMode(InferenceBenchmarkSupport.modelPathFromEnv(), WeightStorageMode.F32, BackendType.JAVA_SEQ, "Plain Java (F32)"));
    }
}
