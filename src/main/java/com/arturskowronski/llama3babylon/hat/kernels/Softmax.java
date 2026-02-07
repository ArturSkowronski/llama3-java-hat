package com.arturskowronski.llama3babylon.hat.kernels;

import jdk.incubator.code.Reflect;
import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;

import java.lang.invoke.MethodHandles;

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
public class Softmax {

    private final Accelerator accelerator;

    public Softmax(Accelerator accelerator) {
        this.accelerator = accelerator;
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
        final float invSum = 1.0f / sum;
        accelerator.compute((Accelerator.@Reflect Compute) cc -> 
            computeNormalize(cc, input, invSum, size)
        );
    }

    @Reflect
    public static void normalizeKernel(KernelContext kc, F32Array input, float invSum) {
        int i = kc.gix;
        input.array(i, input.array(i) * invSum);
    }

    @Reflect
    public static void computeNormalize(ComputeContext cc, F32Array input, float invSum, int size) {
        cc.dispatchKernel(NDRange.of1D(size), kc -> normalizeKernel(kc, input, invSum));
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

    // ========== Test/Demo ==========

    public static void main(String[] args) {
        System.out.println("=== Softmax Kernel Test ===");

        Accelerator accelerator = new Accelerator(MethodHandles.lookup());
        Softmax softmax = new Softmax(accelerator);

        // Test case: simple array
        int size = 5;
        F32Array input = F32Array.create(accelerator, size);
        
        // Initialize with test values: [1.0, 2.0, 3.0, 4.0, 5.0]
        for (int i = 0; i < size; i++) {
            input.array(i, (float) (i + 1));
        }

        System.out.println("Input: ");
        for (int i = 0; i < size; i++) {
            System.out.printf("  [%d] = %.4f%n", i, input.array(i));
        }

        softmax.apply(input, size);

        System.out.println("Output (softmax): ");
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            float val = input.array(i);
            System.out.printf("  [%d] = %.6f%n", i, val);
            sum += val;
        }

        System.out.printf("Sum of outputs: %.6f (should be ~1.0)%n", sum);

        // Verify: softmax outputs should sum to 1.0
        if (Math.abs(sum - 1.0f) < 0.0001f) {
            System.out.println("Softmax test PASSED ✅");
        } else {
            System.out.println("Softmax test FAILED ❌");
        }

        // Test expected values for [1,2,3,4,5]:
        // exp([1,2,3,4,5]) = [2.718, 7.389, 20.086, 54.598, 148.413]
        // sum = 233.204
        // softmax = [0.0117, 0.0317, 0.0861, 0.2341, 0.6364]
        float[] expected = {0.01165f, 0.03168f, 0.08612f, 0.23412f, 0.63640f};
        boolean valuesCorrect = true;
        for (int i = 0; i < size; i++) {
            if (Math.abs(input.array(i) - expected[i]) > 0.001f) {
                System.out.printf("Value mismatch at [%d]: expected %.5f, got %.5f%n", 
                    i, expected[i], input.array(i));
                valuesCorrect = false;
            }
        }
        if (valuesCorrect) {
            System.out.println("Softmax values verified ✅");
        }
    }
}
