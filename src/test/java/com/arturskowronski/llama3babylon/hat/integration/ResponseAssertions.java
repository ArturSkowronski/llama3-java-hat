package com.arturskowronski.llama3babylon.hat.integration;

import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Shared assertions for integration tests that validate model output quality.
 */
public final class ResponseAssertions {

    private static final Pattern REPEATED_CHAR = Pattern.compile("(.)\\1{9,}");
    private static final Pattern HIGH_NON_ASCII_RATIO = Pattern.compile("[^\\x20-\\x7E\\n\\r\\t]");
    private static final double MAX_NON_ASCII_RATIO = 0.3;
    private static final long MIN_ALPHA_COUNT = 3;
    private static final double MIN_UNIQUE_CHAR_RATIO = 0.05;
    private static final int TRUNCATE_LENGTH = 200;

    private ResponseAssertions() {}

    /**
     * Asserts that a model response is non-null, non-empty, and not gibberish.
     */
    public static void assertValidResponse(String response) {
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
    public static void assertNotGibberish(String response) {
        String trimmed = response.strip();

        assertFalse(REPEATED_CHAR.matcher(trimmed).find(),
                "Response looks like gibberish (repeated character run): " + truncate(trimmed));

        long nonAscii = HIGH_NON_ASCII_RATIO.matcher(trimmed).results().count();
        double nonAsciiRatio = (double) nonAscii / trimmed.length();
        assertTrue(nonAsciiRatio < MAX_NON_ASCII_RATIO,
                String.format("Response has %.0f%% non-printable-ASCII characters (likely broken decoding): %s",
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
