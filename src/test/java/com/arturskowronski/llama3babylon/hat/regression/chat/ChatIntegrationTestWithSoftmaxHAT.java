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
 * Integration test for Softmax kernel using HAT @Reflect dispatch in real 16-layer inference.
 * Enables ONLY Softmax for HAT dispatch. Only the normalize step uses HAT (reductions stay in Java).
 */
@Tag("regression")
public class ChatIntegrationTestWithSoftmaxHAT {

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testChatWithSoftmaxHAT() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));

        HybridKernelFactory factory = new HybridKernelFactory(
            Set.of(HybridKernelFactory.KernelType.SOFTMAX)
        );

        LlamaInference inference = new LlamaInference(modelPath, factory);

        int maxTokens = System.getenv("CI") != null ? 32 : 128;
        String response = inference.chat(
                "You are a helpful assistant.",
                "Tell a joke about programming",
                maxTokens
        );

        System.out.println("=== Model Response (Softmax HAT) ===");
        System.out.println(response);
        System.out.println("====================================");

        ResponseAssertions.assertValidResponse(response);
    }
}
