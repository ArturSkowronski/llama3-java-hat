package com.arturskowronski.llama3babylon.hat.integration.chat;

import com.arturskowronski.llama3babylon.hat.LlamaInference;
import com.arturskowronski.llama3babylon.hat.utils.ResponseAssertions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * End-to-end chat integration test using the real Llama 3.2 1B Instruct FP16 model.
 * Runs with ALL kernels in plain-Java mode (no HAT @Reflect dispatch).
 */
@Tag("plain-integration")
public class ChatPlainIntegrationTest {

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testTellAJokeAboutProgramming() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));
        LlamaInference inference = new LlamaInference(modelPath);

        int maxTokens = System.getenv("CI") != null ? 32 : 128;
        String response = inference.chat(
                "You are a helpful assistant.",
                "Tell a joke about programming",
                maxTokens
        );

        System.out.println("=== Model Response ===");
        System.out.println(response);
        System.out.println("======================");

        ResponseAssertions.assertValidResponse(response);
    }
}
