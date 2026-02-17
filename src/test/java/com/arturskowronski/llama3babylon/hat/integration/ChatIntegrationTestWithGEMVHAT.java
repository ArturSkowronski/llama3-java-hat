package com.arturskowronski.llama3babylon.hat.integration;

import com.arturskowronski.llama3babylon.hat.LlamaInference;
import com.arturskowronski.llama3babylon.hat.kernels.HybridKernelFactory;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Set;

/**
 * Integration test for GEMV kernel using HAT @Reflect dispatch in real 16-layer inference.
 * Enables ONLY GEMV for HAT dispatch. Most intensive kernel: ~113 GEMV operations per token.
 */
@Tag("integration")
@Tag("hat-sequential")
public class ChatIntegrationTestWithGEMVHAT {

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testChatWithGEMVHAT() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));

        HybridKernelFactory factory = new HybridKernelFactory(
            Set.of(HybridKernelFactory.KernelType.GEMV)
        );

        LlamaInference inference = new LlamaInference(modelPath, factory);

        int maxTokens = System.getenv("CI") != null ? 32 : 128;
        String response = inference.chat(
                "You are a helpful assistant.",
                "Tell a joke about programming",
                maxTokens
        );

        System.out.println("=== Model Response (GEMV HAT) ===");
        System.out.println(response);
        System.out.println("==================================");

        ResponseAssertions.assertValidResponse(response);
    }
}
