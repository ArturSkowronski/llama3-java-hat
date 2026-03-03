package com.arturskowronski.llama3babylon.hat;

/**
 * Controls how F16 weight tensors are stored in memory during inference.
 */
public enum WeightStorageMode {
    /** Eagerly dequantize F16 tensors to F32Array (original behavior, 2x memory). */
    F32,
    /** Keep F16 tensors as native F16Array (half memory, dequant at compute time). */
    F16
}
