package com.arturskowronski.llama3babylon.hat.kernels;

import hat.buffer.F32Array;

/**
 * Interface for SiLU (Sigmoid Linear Unit) kernel implementations.
 */
public interface ISiLU {

    /**
     * Applies SiLU to the input array in-place.
     *
     * @param input the input array (modified in-place)
     * @param size the number of elements to process
     */
    void apply(F32Array input, int size);
}
