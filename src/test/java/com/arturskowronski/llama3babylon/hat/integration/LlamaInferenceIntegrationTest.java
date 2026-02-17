package com.arturskowronski.llama3babylon.hat.integration;

import com.arturskowronski.llama3babylon.hat.LlamaInference;
import com.arturskowronski.llama3babylon.hat.LlamaModel;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.*;

@Tag("integration")
@Tag("plain-java")
public class LlamaInferenceIntegrationTest {

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testFullForwardPass() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));
        assertTrue(modelPath.toFile().exists(), "Model file not found: " + modelPath);

        LlamaInference inference = new LlamaInference(modelPath);

        // BOS token for Llama 3
        int bosToken = 128000;
        float[] logits = inference.forward(bosToken, 0);

        // Logits should have VOCAB_SIZE elements
        assertEquals(LlamaModel.VOCAB_SIZE, logits.length);

        // All logits should be finite (no NaN or Inf)
        for (int i = 0; i < logits.length; i++) {
            assertFalse(Float.isNaN(logits[i]), "Logit is NaN at index " + i);
            assertFalse(Float.isInfinite(logits[i]), "Logit is Inf at index " + i);
        }

        // argmax should return a valid token ID
        int predicted = LlamaInference.argmax(logits);
        assertTrue(predicted >= 0 && predicted < LlamaModel.VOCAB_SIZE,
                "Predicted token ID out of range: " + predicted);
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testMultiTokenForwardPass() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));
        LlamaInference inference = new LlamaInference(modelPath);

        // Process a small sequence: BOS + a few tokens
        int[] tokens = {128000, 1, 2, 3};
        float[] logits = null;
        for (int i = 0; i < tokens.length; i++) {
            logits = inference.forward(tokens[i], i);
        }

        assertNotNull(logits);
        assertEquals(LlamaModel.VOCAB_SIZE, logits.length);

        for (int i = 0; i < logits.length; i++) {
            assertFalse(Float.isNaN(logits[i]), "Logit is NaN at index " + i);
            assertFalse(Float.isInfinite(logits[i]), "Logit is Inf at index " + i);
        }
    }
}
