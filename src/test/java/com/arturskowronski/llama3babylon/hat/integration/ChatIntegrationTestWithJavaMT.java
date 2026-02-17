package com.arturskowronski.llama3babylon.hat.integration;

import com.arturskowronski.llama3babylon.hat.BackendType;
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
 * Integration test running all 6 HAT kernels on the Java Multi-Threaded backend.
 * Validates that backend switching works correctly before attempting GPU (OpenCL).
 */
@Tag("integration")
@Tag("hat-gpu")
public class ChatIntegrationTestWithJavaMT {

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testChatWithAllHATKernelsOnJavaMT() throws IOException {
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

        LlamaInference inference = new LlamaInference(modelPath, factory, BackendType.JAVA_MT);

        int maxTokens = System.getenv("CI") != null ? 32 : 128;
        String response = inference.chat(
                "You are a helpful assistant.",
                "Tell a joke about programming",
                maxTokens
        );

        System.out.println("=== Model Response (ALL HAT KERNELS + JAVA MT BACKEND) ===");
        System.out.println(response);
        System.out.println("===========================================================");

        ResponseAssertions.assertValidResponse(response);
    }
}
