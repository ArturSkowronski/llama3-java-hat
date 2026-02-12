package com.arturskowronski.llama3babylon.hat.kernels;

import hat.buffer.F32Array;

/**
 * Interface for Softmax kernel implementations.
 */
public interface ISoftmax {

    /**
     * Applies softmax in-place to the input array.
     *
     * @param input the input array (modified in-place)
     * @param size the number of elements to process
     */
    void apply(F32Array input, int size);

    /**
     * Applies softmax to a specific row of a 2D array (stored as 1D).
     * Useful for attention scores where each row is softmaxed independently.
     *
     * @param input the input array
     * @param rowOffset starting index of the row
     * @param rowSize number of elements in the row
     */
    void applyRow(F32Array input, int rowOffset, int rowSize);
}
