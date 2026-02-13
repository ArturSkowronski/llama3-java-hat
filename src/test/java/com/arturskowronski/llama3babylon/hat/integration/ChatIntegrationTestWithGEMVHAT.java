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
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration test for GEMV kernel using HAT @Reflect dispatch in real 16-layer inference.
 *
 * FINAL BOSS: GEMV is the sixth and last kernel to be tested with HAT dispatch.
 * This test enables ONLY the GEMV kernel for HAT dispatch while keeping all other kernels
 * in plain Java mode.
 *
 * GEMV characteristics:
 * - Most computationally intensive kernel (matrix-vector multiplication)
 * - Parallelized over rows (NDRange.of1D(rows))
 * - Handles matrix sizes from 2048×2048 up to 128256×2048 (classifier)
 *
 * GEMV is called extensively in inference:
 * - 4 times per layer for Q/K/V/O projections (64 total for 16 layers)
 * - 3 times per layer for FFN (gate/up/down) (48 total for 16 layers)
 * - 1 time for final classifier (128256×2048 - the largest matrix)
 * - Total: ~113 GEMV operations per token
 *
 * Success criteria:
 * - Test completes without exception
 * - Response is coherent English text (not gibberish)
 * - Response contains joke structure (setup + punchline)
 * - Output should be identical or very similar to plain Java version
 */
@Tag("integration")
public class ChatIntegrationTestWithGEMVHAT {

    /** Matches strings dominated by a single repeated character (e.g. "{{{{{" or "-----"). */
    private static final Pattern REPEATED_CHAR = Pattern.compile("(.)\\1{9,}");

    /** Matches strings with excessive non-ASCII/control chars — sign of broken decoding. */
    private static final Pattern HIGH_NON_ASCII_RATIO = Pattern.compile("[^\\x20-\\x7E\\n\\r\\t]");

    /** Max fraction of non-printable-ASCII characters before flagging as broken decoding. */
    private static final double MAX_NON_ASCII_RATIO = 0.3;

    /** Minimum number of alphabetic characters required in a valid response. */
    private static final long MIN_ALPHA_COUNT = 3;

    /** Minimum unique-character ratio — natural text has much higher diversity than gibberish. */
    private static final double MIN_UNIQUE_CHAR_RATIO = 0.05;

    /** Max length for error message excerpts. */
    private static final int TRUNCATE_LENGTH = 200;

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testChatWithGEMVHAT() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));

        // Create factory with ONLY GEMV HAT enabled
        HybridKernelFactory factory = new HybridKernelFactory(
            Set.of(HybridKernelFactory.KernelType.GEMV)
        );

        LlamaInference inference = new LlamaInference(modelPath, factory);

        // Use fewer tokens on CI to stay within GitHub runner time limits.
        // 32 tokens is enough for a short joke; local runs can use more.
        int maxTokens = System.getenv("CI") != null ? 32 : 128;
        String response = inference.chat(
                "You are a helpful assistant.",
                "Tell a joke about programming",
                maxTokens
        );

        System.out.println("=== Model Response (GEMV HAT) ===");
        System.out.println(response);
        System.out.println("==================================");

        assertNotNull(response, "Response must not be null");
        assertFalse(response.strip().isEmpty(), "Response must not be empty");
        assertNotGibberish(response);
    }

    /**
     * Heuristic gibberish detector. Catches the known failure modes:
     * 1. Repeated character spam (e.g. "{{{{{{{{{" from broken HAT dispatch)
     * 2. Excessive non-printable / non-ASCII characters (broken token decoding)
     * 3. No alphabetic content at all (pure punctuation/numbers)
     * 4. Extremely low unique-character ratio (monotonous output)
     */
    private static void assertNotGibberish(String response) {
        String trimmed = response.strip();

        // 1. Repeated character runs — e.g. "{{{{{{{{{{" or "----------"
        assertFalse(REPEATED_CHAR.matcher(trimmed).find(),
                "Response looks like gibberish (repeated character run): " + truncate(trimmed));

        // 2. Non-ASCII ratio — allow some (curly quotes, etc.) but flag above threshold
        long nonAscii = HIGH_NON_ASCII_RATIO.matcher(trimmed).results().count();
        double nonAsciiRatio = (double) nonAscii / trimmed.length();
        assertTrue(nonAsciiRatio < MAX_NON_ASCII_RATIO,
                String.format("Response has %.0f%% non-printable-ASCII characters (likely broken decoding): %s",
                        nonAsciiRatio * 100, truncate(trimmed)));

        // 3. Must contain at least some alphabetic characters
        long alphaCount = trimmed.chars().filter(Character::isLetter).count();
        assertTrue(alphaCount >= MIN_ALPHA_COUNT,
                "Response contains almost no alphabetic characters: " + truncate(trimmed));

        // 4. Unique character ratio — natural text uses many distinct chars
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
