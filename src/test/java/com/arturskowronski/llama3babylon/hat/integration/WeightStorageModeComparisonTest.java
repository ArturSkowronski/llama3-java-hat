package com.arturskowronski.llama3babylon.hat.integration;

import com.arturskowronski.llama3babylon.hat.LlamaInference;
import com.arturskowronski.llama3babylon.hat.LlamaModel;
import com.arturskowronski.llama3babylon.hat.kernels.PlainJavaKernelFactory;
import com.arturskowronski.llama3babylon.hat.BackendType;
import com.arturskowronski.llama3babylon.hat.WeightStorageMode;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.*;

@Tag("plain-integration")
public class WeightStorageModeComparisonTest {

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testForwardPassLogitsIdentical() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));

        LlamaInference f16Inference = new LlamaInference(
                modelPath, new PlainJavaKernelFactory(), BackendType.JAVA_SEQ, WeightStorageMode.F16);
        LlamaInference f32Inference = new LlamaInference(
                modelPath, new PlainJavaKernelFactory(), BackendType.JAVA_SEQ, WeightStorageMode.F32);

        int bosToken = 128000;
        float[] f16Logits = f16Inference.forward(bosToken, 0);
        float[] f32Logits = f32Inference.forward(bosToken, 0);

        assertEquals(LlamaModel.VOCAB_SIZE, f16Logits.length);
        assertEquals(LlamaModel.VOCAB_SIZE, f32Logits.length);

        // Logits should be bit-identical since both paths ultimately
        // dequantize the same F16 values before computation
        assertArrayEquals(f32Logits, f16Logits,
                "F16 and F32 weight storage modes should produce identical logits");
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testChatOutputIdentical() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));

        LlamaInference f16Inference = new LlamaInference(
                modelPath, new PlainJavaKernelFactory(), BackendType.JAVA_SEQ, WeightStorageMode.F16);
        LlamaInference f32Inference = new LlamaInference(
                modelPath, new PlainJavaKernelFactory(), BackendType.JAVA_SEQ, WeightStorageMode.F32);

        String systemPrompt = "You are a helpful assistant.";
        String userPrompt = "Tell a joke about programming";
        int maxTokens = 32;

        String f16Response = f16Inference.chat(systemPrompt, userPrompt, maxTokens);
        String f32Response = f32Inference.chat(systemPrompt, userPrompt, maxTokens);

        assertFalse(f16Response.isEmpty(), "F16 response should not be empty");
        assertFalse(f32Response.isEmpty(), "F32 response should not be empty");
        assertEquals(f32Response, f16Response,
                "F16 and F32 weight storage modes should produce identical chat output");
    }
}
