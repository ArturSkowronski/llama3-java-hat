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
 * Integration test with ALL 6 kernels using HAT @Reflect dispatch simultaneously.
 * Validates that all HAT kernels work together in real 16-layer inference.
 * Total HAT dispatches per token: ~250, ~8,000 for 32-token inference.
 */
@Tag("integration")
@Tag("hat-sequential")
public class ChatIntegrationTestWithAllHAT {

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testChatWithAllHATKernels() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));

        HybridKernelFactory factory = new HybridKernelFactory(
            Set.of(
                HybridKernelFactory.KernelType.GEMV,
                HybridKernelFactory.KernelType.RMSNORM,
                HybridKernelFactory.KernelType.ROPE,
                HybridKernelFactory.KernelType.SILU,
                HybridKernelFactory.KernelType.SOFTMAX,
                HybridKernelFactory.KernelType.ATTENTION
            )
        );

        LlamaInference inference = new LlamaInference(modelPath, factory);

        int maxTokens = System.getenv("CI") != null ? 32 : 128;
        String response = inference.chat(
                "You are a helpful assistant.",
                "Tell a joke about programming",
                maxTokens
        );

        System.out.println("=== Model Response (ALL HAT KERNELS) ===");
        System.out.println(response);
        System.out.println("=========================================");

        ResponseAssertions.assertValidResponse(response);
    }
}
