package com.arturskowronski.llama3babylon.hat;

import hat.Accelerator;
import hat.buffer.F16Array;

/**
 * CPU-optimized F16 weight storage backed by a plain {@code short[]} array.
 *
 * <p>Unlike {@link F16Array}, which goes through HAT's iface-mapper proxy (~40 cycles/element),
 * {@code F16Weights} stores raw F16 bit patterns in a plain Java array that the JIT can
 * vectorize freely. This gives F32-class CPU performance at F16 memory cost.
 *
 * <p>For GPU dispatch, call {@link #toF16Array(Accelerator)} to lazily materialize
 * an {@code F16Array} (one-time cost per tensor, amortized over inference).
 */
public class F16Weights {

    private final short[] data;
    private final int rows;
    private final int cols;
    private volatile F16Array cachedF16Array;

    public F16Weights(short[] data, int rows, int cols) {
        if (data.length != rows * cols) {
            throw new IllegalArgumentException(
                    "data.length (" + data.length + ") != rows*cols (" + rows + "*" + cols + "=" + (rows * cols) + ")");
        }
        this.data = data;
        this.rows = rows;
        this.cols = cols;
    }

    /** Raw F16 bit patterns — direct array access for GEMV hot path. */
    public short[] data() {
        return data;
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public int length() {
        return data.length;
    }

    /** Dequantize a single element (for embedding lookup). */
    public float getFloat(int index) {
        return Float.float16ToFloat(data[index]);
    }

    /**
     * Bulk-dequantize one row into a pre-allocated float[] buffer.
     * JIT can vectorize this loop since both arrays are plain Java arrays.
     */
    public void dequantRow(int row, float[] dest) {
        int offset = row * cols;
        for (int c = 0; c < cols; c++) {
            dest[c] = Float.float16ToFloat(data[offset + c]);
        }
    }

    /**
     * Lazily materializes an {@link F16Array} for HAT/GPU dispatch.
     * Cached after first call — one-time cost per tensor.
     */
    public F16Array toF16Array(Accelerator accelerator) {
        F16Array result = cachedF16Array;
        if (result == null) {
            synchronized (this) {
                result = cachedF16Array;
                if (result == null) {
                    result = F16Array.create(accelerator, data.length);
                    for (int i = 0; i < data.length; i++) {
                        result.array(i).value(data[i]);
                    }
                    cachedF16Array = result;
                }
            }
        }
        return result;
    }
}
