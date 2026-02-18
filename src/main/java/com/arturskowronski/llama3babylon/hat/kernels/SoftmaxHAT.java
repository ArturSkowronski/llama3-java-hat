package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;
import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.RW;

/**
 * Softmax kernel using HAT @Reflect dispatch.
 *
 * Third kernel to be tested with HAT dispatch in real 16-layer inference.
 * Softmax is already partially HAT-enabled in the base implementation (normalize step),
 * so this version makes the HAT usage explicit and consistent.
 *
 * Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * Implementation strategy:
 * 1. Find max value (plain Java - reduction not efficiently parallelizable)
 * 2. Compute exp(x - max) and sum (plain Java - reduction required)
 * 3. Normalize by sum (HAT dispatch - perfectly parallelizable)
 *
 * Note: Steps 1-2 use plain Java because they require reduction operations (max, sum)
 * which don't parallelize well on sequential backend. The normalize step (element-wise
 * multiplication) is ideal for HAT dispatch.
 */
public class SoftmaxHAT implements ISoftmax {

    private final Accelerator accelerator;

    public SoftmaxHAT(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    /**
     * Applies softmax in-place to the input array using HAT dispatch.
     *
     * @param input the input array (modified in-place)
     * @param size the number of elements to process
     */
    @Override
    public void apply(F32Array input, int size) {
        // Step 1: Find max value (CPU-side - reduction operation)
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            float val = input.array(i);
            if (val > maxVal) {
                maxVal = val;
            }
        }

        // Step 2: Compute exp(x - max) and sum (CPU-side - reduction operation)
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            float expVal = (float) Math.exp(input.array(i) - maxVal);
            input.array(i, expVal);
            sum += expVal;
        }

        // Step 3: Normalize by sum (HAT dispatch - element-wise parallelizable)
        final float invSum = 1.0f / sum;
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
            computeNormalize(cc, input, invSum, size)
        );
    }

    @Reflect
    public static void computeNormalize(@RO ComputeContext cc, @RW F32Array input, @RO float invSum, @RO int size) {
        cc.dispatchKernel(NDRange.of1D(size), kc -> normalizeKernel(kc, input, invSum));
    }

    @Reflect
    public static void normalizeKernel(@RO KernelContext kc, @RW F32Array input, @RO float invSum) {
        int i = kc.gix;
        input.array(i, input.array(i) * invSum);
    }

    /**
     * Applies softmax to a specific row of a 2D array (stored as 1D).
     * Used for attention scores where each query processes independently.
     *
     * @param input the input array
     * @param rowOffset starting index of the row
     * @param rowSize number of elements in the row
     */
    @Override
    public void applyRow(F32Array input, int rowOffset, int rowSize) {
        // Step 1: Find max value in row (CPU-side - reduction)
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < rowSize; i++) {
            float val = input.array(rowOffset + i);
            if (val > maxVal) {
                maxVal = val;
            }
        }

        // Step 2: Compute exp(x - max) and sum (CPU-side - reduction)
        float sum = 0.0f;
        for (int i = 0; i < rowSize; i++) {
            float expVal = (float) Math.exp(input.array(rowOffset + i) - maxVal);
            input.array(rowOffset + i, expVal);
            sum += expVal;
        }

        // Step 3: Normalize by sum (plain Java for row-level - small size)
        // Note: Using plain Java here because row sizes are small (seqLen),
        // and HAT dispatch overhead would exceed benefit
        float invSum = 1.0f / sum;
        for (int i = 0; i < rowSize; i++) {
            input.array(rowOffset + i, input.array(rowOffset + i) * invSum);
        }
    }
}
