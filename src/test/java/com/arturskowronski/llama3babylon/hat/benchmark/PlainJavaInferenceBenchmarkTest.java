package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.List;

@Tag("benchmark")
@Tag("benchmark-chat")
public class PlainJavaInferenceBenchmarkTest {

    @Test
    public void benchmarkPlainJava() {
        String model = System.getenv("LLAMA_FP16_PATH");
        if (model == null || model.isBlank()) {
            InferenceBenchmarkSupport.recordResults(List.of(
                    InferenceBenchmarkSupport.skipped("Plain Java (F16)", "LLAMA_FP16_PATH is not set"),
                    InferenceBenchmarkSupport.skipped("Plain Java (F16_FAST)", "LLAMA_FP16_PATH is not set"),
                    InferenceBenchmarkSupport.skipped("Plain Java (F32)", "LLAMA_FP16_PATH is not set")));
            return;
        }

        var path = InferenceBenchmarkSupport.modelPathFromEnv();
        InferenceBenchmarkSupport.recordResults(List.of(
                InferenceBenchmarkSupport.runWeightMode(path, WeightStorageMode.F16, BackendType.JAVA_SEQ, "Plain Java (F16)"),
                InferenceBenchmarkSupport.runWeightMode(path, WeightStorageMode.F16_FAST, BackendType.JAVA_SEQ, "Plain Java (F16_FAST)"),
                InferenceBenchmarkSupport.runWeightMode(path, WeightStorageMode.F32, BackendType.JAVA_SEQ, "Plain Java (F32)")
        ));
    }
}
