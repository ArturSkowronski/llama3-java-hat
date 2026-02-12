package com.arturskowronski.llama3babylon.hat.integration;

import com.arturskowronski.llama3babylon.hat.LlamaInference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.*;

@Tag("integration")
public class ChatIntegrationTest {

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testTellAJokeAboutProgramming() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));
        LlamaInference inference = new LlamaInference(modelPath);

        String response = inference.chat(
                "You are a helpful assistant.",
                "Tell a joke about programming",
                64
        );

        System.out.println("=== Model Response ===");
        System.out.println(response);
        System.out.println("======================");

        assertNotNull(response);
        assertFalse(response.strip().isEmpty(), "Response should contain non-whitespace text");
    }
}
