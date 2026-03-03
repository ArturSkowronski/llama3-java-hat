package com.arturskowronski.llama3babylon.hat;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.*;

public class WeightStorageModeTest {

    @ParameterizedTest
    @CsvSource({
            "F16,       F16",
            "f16,       F16",
            "F32,       F32",
            "f32,       F32",
            "F16_FAST,  F16_FAST",
            "f16_fast,  F16_FAST",
            "F16FAST,   F16_FAST",
            "FAST,      F16_FAST",
            "fast,      F16_FAST",
            " F16 ,     F16"
    })
    public void testFromString(String input, WeightStorageMode expected) {
        assertEquals(expected, WeightStorageMode.fromString(input));
    }

    @Test
    public void testFromStringRejectsUnknown() {
        assertThrows(IllegalArgumentException.class, () -> WeightStorageMode.fromString("Q4_K"));
    }

    @Test
    public void testFromStringRejectsEmpty() {
        assertThrows(IllegalArgumentException.class, () -> WeightStorageMode.fromString(""));
    }

    @Test
    public void testEnumDescriptors() {
        // Verify all modes have non-empty metadata
        for (WeightStorageMode mode : WeightStorageMode.values()) {
            assertNotNull(mode.storage());
            assertNotNull(mode.summary());
            assertFalse(mode.storage().isBlank(), mode + " storage is blank");
            assertFalse(mode.summary().isBlank(), mode + " summary is blank");
        }
    }

    @Test
    public void testFromEnvDefaultsToF16() {
        // When no env/sysprop is set, should default to F16
        // (This test assumes WEIGHT_STORAGE_MODE is not set in the test env)
        String envVal = System.getenv("WEIGHT_STORAGE_MODE");
        if (envVal == null || envVal.isBlank()) {
            assertEquals(WeightStorageMode.F16, WeightStorageMode.fromEnv());
        }
    }
}
