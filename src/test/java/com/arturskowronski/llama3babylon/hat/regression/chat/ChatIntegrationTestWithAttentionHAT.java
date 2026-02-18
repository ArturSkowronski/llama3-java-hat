package com.arturskowronski.llama3babylon.hat.regression.chat;

import com.arturskowronski.llama3babylon.hat.LlamaInference;
import com.arturskowronski.llama3babylon.hat.utils.ResponseAssertions;
import com.arturskowronski.llama3babylon.hat.kernels.HybridKernelFactory;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Set;

/**
 * Integration test for Attention kernel using HAT @Reflect dispatch in real 16-layer inference.
 * Enables ONLY Attention for HAT dispatch. Two-step: compute scores (Q*K^T) and compute values (Scores*V).
 */
@Tag("integration")
@Tag("regression")
@Tag("hat-sequential")
public class ChatIntegrationTestWithAttentionHAT {

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testChatWithAttentionHAT() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));

        HybridKernelFactory factory = new HybridKernelFactory(
            Set.of(HybridKernelFactory.KernelType.ATTENTION)
        );

        LlamaInference inference = new LlamaInference(modelPath, factory);

        int maxTokens = System.getenv("CI") != null ? 32 : 128;
        String response = inference.chat(
                "You are a helpful assistant.",
                "Tell a joke about programming",
                maxTokens
        );

        System.out.println("=== Model Response (Attention HAT) ===");
        System.out.println(response);
        System.out.println("======================================");

        ResponseAssertions.assertValidResponse(response);
    }
}
