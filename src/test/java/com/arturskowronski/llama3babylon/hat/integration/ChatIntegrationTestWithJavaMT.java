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
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration test running all 6 HAT kernels on the Java Multi-Threaded backend.
 *
 * Validates that backend switching works correctly before attempting GPU (OpenCL).
 * The Java MT backend is a pure Java parallel backend — no native libraries needed.
 */
@Tag("integration")
public class ChatIntegrationTestWithJavaMT {

    private static final Pattern REPEATED_CHAR = Pattern.compile("(.)\\1{9,}");
    private static final Pattern HIGH_NON_ASCII_RATIO = Pattern.compile("[^\\x20-\\x7E\\n\\r\\t]");
    private static final double MAX_NON_ASCII_RATIO = 0.3;
    private static final long MIN_ALPHA_COUNT = 3;
    private static final double MIN_UNIQUE_CHAR_RATIO = 0.05;
    private static final int TRUNCATE_LENGTH = 200;

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

        assertNotNull(response, "Response must not be null");
        assertFalse(response.strip().isEmpty(), "Response must not be empty");
        assertNotGibberish(response);
    }

    private static void assertNotGibberish(String response) {
        String trimmed = response.strip();

        assertFalse(REPEATED_CHAR.matcher(trimmed).find(),
                "Response looks like gibberish (repeated character run): " + truncate(trimmed));

        long nonAscii = HIGH_NON_ASCII_RATIO.matcher(trimmed).results().count();
        double nonAsciiRatio = (double) nonAscii / trimmed.length();
        assertTrue(nonAsciiRatio < MAX_NON_ASCII_RATIO,
                String.format("Response has %.0f%% non-printable-ASCII characters: %s",
                        nonAsciiRatio * 100, truncate(trimmed)));

        long alphaCount = trimmed.chars().filter(Character::isLetter).count();
        assertTrue(alphaCount >= MIN_ALPHA_COUNT,
                "Response contains almost no alphabetic characters: " + truncate(trimmed));

        long uniqueChars = trimmed.chars().distinct().count();
        double uniqueRatio = (double) uniqueChars / trimmed.length();
        assertTrue(uniqueRatio > MIN_UNIQUE_CHAR_RATIO,
                String.format("Response has very low character diversity (%.1f%% unique): %s",
                        uniqueRatio * 100, truncate(trimmed)));
    }

    private static String truncate(String s) {
        return s.length() <= TRUNCATE_LENGTH ? s : s.substring(0, TRUNCATE_LENGTH) + "...";
    }
}
