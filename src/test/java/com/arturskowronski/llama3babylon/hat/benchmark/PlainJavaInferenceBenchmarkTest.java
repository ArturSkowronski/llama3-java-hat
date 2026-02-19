package com.arturskowronski.llama3babylon.hat.benchmark;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("benchmark")
public class PlainJavaInferenceBenchmarkTest {

    @Test
    public void benchmarkPlainJava() {
        String model = System.getenv("LLAMA_FP16_PATH");
        if (model == null || model.isBlank()) {
            InferenceBenchmarkSupport.recordResult(
                    InferenceBenchmarkSupport.skipped("Plain Java", "LLAMA_FP16_PATH is not set"));
            return;
        }

        var result = InferenceBenchmarkSupport.runPlainJava(InferenceBenchmarkSupport.modelPathFromEnv());
        InferenceBenchmarkSupport.recordResult(result);
    }
}
