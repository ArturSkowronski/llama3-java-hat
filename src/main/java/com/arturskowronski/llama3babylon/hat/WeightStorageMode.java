package com.arturskowronski.llama3babylon.hat;

/**
 * Controls how F16 weight tensors are stored in memory during inference.
 *
 * <p>Pick a mode based on your target backend:
 * <pre>
 *   Mode      | Storage    | CPU speed | GPU dispatch     | Memory
 *   ----------|------------|-----------|------------------|-------
 *   F16       | F16Array   | slow      | native (direct)  | 1x     (default)
 *   F16_FAST  | short[]    | fast      | lazy materialize | 1x
 *   F32       | F32Array   | fast      | fast             | 2x
 * </pre>
 *
 * <p>Switchable at runtime via env var or system property:
 * <pre>
 *   WEIGHT_STORAGE_MODE=F16_FAST ./gradlew benchmarkF16F32
 *   -Dweight.storage.mode=F16_FAST
 * </pre>
 */
public enum WeightStorageMode {

    /** Eagerly dequantize F16 tensors to F32Array (2x memory, fast CPU & GPU). */
    F32("F32Array", "2x mem, fast CPU & GPU"),

    /** Native F16Array — HAT can transfer directly to GPU. Default. */
    F16("F16Array", "1x mem, native GPU, slow CPU"),

    /** CPU-optimized plain short[] — fast CPU, lazy F16Array materialization for GPU. */
    F16_FAST("short[]", "1x mem, fast CPU, lazy GPU");

    private static final String ENV_KEY = "WEIGHT_STORAGE_MODE";

    private final String storage;
    private final String summary;

    WeightStorageMode(String storage, String summary) {
        this.storage = storage;
        this.summary = summary;
    }

    /** Backing storage type (for display/logging). */
    public String storage() { return storage; }

    /** One-line human summary (for display/logging). */
    public String summary() { return summary; }

    /**
     * Resolve from env var {@code WEIGHT_STORAGE_MODE} or system property
     * {@code weight.storage.mode}. Falls back to {@link #F16} if neither is set.
     *
     * <p>Accepts: {@code F16}, {@code F16_FAST} (also {@code F16FAST}, {@code FAST}),
     * {@code F32}. Case-insensitive.
     */
    public static WeightStorageMode fromEnv() {
        String val = System.getenv(ENV_KEY);
        if (val == null || val.isBlank()) {
            val = System.getProperty("weight.storage.mode");
        }
        return (val != null && !val.isBlank()) ? fromString(val) : F16;
    }

    /**
     * Parse a mode name (case-insensitive).
     * @throws IllegalArgumentException on unknown value
     */
    public static WeightStorageMode fromString(String s) {
        return switch (s.strip().toUpperCase()) {
            case "F32" -> F32;
            case "F16" -> F16;
            case "F16_FAST", "F16FAST", "FAST" -> F16_FAST;
            default -> throw new IllegalArgumentException(
                    "Unknown weight storage mode: '" + s + "'. Valid: F16, F16_FAST, F32");
        };
    }
}
