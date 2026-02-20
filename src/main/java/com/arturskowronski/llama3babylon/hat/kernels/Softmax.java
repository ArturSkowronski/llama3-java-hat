package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.buffer.F32Array;

/**
 * Softmax kernel for Llama 3.2 1B Instruct (FP16).
 *
 * Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * This is a numerically stable implementation that:
 * 1. Finds the maximum value in the input
 * 2. Subtracts max before exponentiating (prevents overflow)
 * 3. Normalizes by the sum of exponentials
 */
public class Softmax implements ISoftmax {

    public Softmax(Accelerator accelerator) {
        // Kept for factory symmetry with HAT implementation.
    }

    /**
     * Applies softmax in-place to the input array.
     * 
     * @param input the input array (modified in-place)
     * @param size the number of elements to process
     */
    public void apply(F32Array input, int size) {
        // Step 1: Find max value (CPU-side for simplicity, can be GPU-optimized later)
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            float val = input.array(i);
            if (val > maxVal) {
                maxVal = val;
            }
        }

        // Step 2: Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            float expVal = (float) Math.exp(input.array(i) - maxVal);
            input.array(i, expVal);
            sum += expVal;
        }

        // Step 3: Normalize by sum
        float invSum = 1.0f / sum;
        for (int i = 0; i < size; i++) {
            input.array(i, input.array(i) * invSum);
        }
    }

    /**
     * Applies softmax to a specific row of a 2D array (stored as 1D).
     * Useful for attention scores where each row is softmaxed independently.
     * 
     * @param input the input array
     * @param rowOffset starting index of the row
     * @param rowSize number of elements in the row
     */
    public void applyRow(F32Array input, int rowOffset, int rowSize) {
        // Step 1: Find max value in row
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < rowSize; i++) {
            float val = input.array(rowOffset + i);
            if (val > maxVal) {
                maxVal = val;
            }
        }

        // Step 2: Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < rowSize; i++) {
            float expVal = (float) Math.exp(input.array(rowOffset + i) - maxVal);
            input.array(rowOffset + i, expVal);
            sum += expVal;
        }

        // Step 3: Normalize by sum
        float invSum = 1.0f / sum;
        for (int i = 0; i < rowSize; i++) {
            input.array(rowOffset + i, input.array(rowOffset + i) * invSum);
        }
    }
}
