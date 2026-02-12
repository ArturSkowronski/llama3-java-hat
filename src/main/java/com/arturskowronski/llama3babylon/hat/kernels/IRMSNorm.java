package com.arturskowronski.llama3babylon.hat.kernels;

import hat.buffer.F32Array;

/**
 * Interface for RMSNorm (Root Mean Square Layer Normalization) kernel implementations.
 */
public interface IRMSNorm {

    /**
     * Applies RMSNorm to the input array in-place.
     *
     * @param input the input array (modified in-place)
     * @param weight the weight tensor
     * @param size the size of the hidden dimension
     */
    void apply(F32Array input, F32Array weight, int size);
}
