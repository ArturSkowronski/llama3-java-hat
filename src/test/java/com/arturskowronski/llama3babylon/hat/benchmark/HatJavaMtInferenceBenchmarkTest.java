package com.arturskowronski.llama3babylon.hat.benchmark;

import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.List;

@Tag("benchmark")
@Tag("benchmark-chat")
public class HatJavaMtInferenceBenchmarkTest {

    @Test
    public void benchmarkHatJavaMt() {
        String model = System.getenv("LLAMA_FP16_PATH");
        if (model == null || model.isBlank()) {
            InferenceBenchmarkSupport.recordResults(List.of(
                    InferenceBenchmarkSupport.skipped("HAT Java MT (F16)", "LLAMA_FP16_PATH is not set"),
                    InferenceBenchmarkSupport.skipped("HAT Java MT (F16_FAST)", "LLAMA_FP16_PATH is not set"),
                    InferenceBenchmarkSupport.skipped("HAT Java MT (F32)", "LLAMA_FP16_PATH is not set")));
            return;
        }

        var path = InferenceBenchmarkSupport.modelPathFromEnv();
        InferenceBenchmarkSupport.recordResults(List.of(
                InferenceBenchmarkSupport.runHat(path, BackendType.JAVA_MT, WeightStorageMode.F16, "HAT Java MT (F16)"),
                InferenceBenchmarkSupport.runHat(path, BackendType.JAVA_MT, WeightStorageMode.F16_FAST, "HAT Java MT (F16_FAST)"),
                InferenceBenchmarkSupport.runHat(path, BackendType.JAVA_MT, WeightStorageMode.F32, "HAT Java MT (F32)")
        ));
    }
}
