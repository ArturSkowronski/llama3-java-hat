package com.arturskowronski.llama3babylon.hat.integration;

import com.arturskowronski.llama3babylon.hat.LlamaInference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end chat integration test using the real Llama 3.2 1B Instruct FP16 model.
 *
 * NOTE: This test runs with ALL kernels in plain-Java mode (no HAT @Reflect dispatch).
 * HAT dispatch passes unit tests but produces gibberish in real 16-layer inference
 * due to a data visibility bug in the Java Sequential Backend.
 * See HAT_BUG_REPORT.md and SiLUHatVsJavaTest for details.
 */
@Tag("integration")
public class ChatIntegrationTest {

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
    public void testTellAJokeAboutProgramming() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));
        LlamaInference inference = new LlamaInference(modelPath);

        // Use fewer tokens on CI to stay within GitHub runner time limits.
        // 32 tokens is enough for a short joke; local runs can use more.
        int maxTokens = System.getenv("CI") != null ? 32 : 128;
        String response = inference.chat(
                "You are a helpful assistant.",
                "Tell a joke about programming",
                maxTokens
        );

        System.out.println("=== Model Response ===");
        System.out.println(response);
        System.out.println("======================");

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
