package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;

/**
 * RMSNorm (Root Mean Square Layer Normalization) kernel using HAT @Reflect dispatch.
 *
 * Fourth kernel to be tested with HAT dispatch in real 16-layer inference.
 * RMSNorm follows the same hybrid pattern as Softmax: reduction in plain Java,
 * element-wise normalization with HAT dispatch.
 *
 * Formula: y = (x / RMS(x)) * weight
 * where RMS(x) = sqrt(1/n * sum(x_i^2) + epsilon)
 *
 * Implementation strategy:
 * 1. Compute sum of squares (plain Java - reduction operation)
 * 2. Compute invRms scalar (plain Java - simple math)
 * 3. Normalize and scale (HAT dispatch - element-wise parallelizable)
 *
 * Note: Step 1 uses plain Java because reduction (sum) doesn't parallelize well
 * on sequential backend. Step 3 (element-wise multiply) is ideal for HAT dispatch.
 */
public class RMSNormHAT implements IRMSNorm {

    private final Accelerator accelerator;
    private static final float EPSILON = 1e-5f;

    public RMSNormHAT(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    /**
     * Applies RMSNorm to the input array in-place using HAT dispatch.
     *
     * @param input the input array (modified in-place)
     * @param weight the weight tensor
     * @param size the size of the hidden dimension
     */
    @Override
    public void apply(F32Array input, F32Array weight, int size) {
        // Step 1: Compute sum of squares (CPU-side - reduction operation)
        float ss = 0.0f;
        for (int i = 0; i < size; i++) {
            float val = input.array(i);
            ss += val * val;
        }

        // Step 2: Compute inverse RMS scalar (CPU-side - simple calculation)
        float invRms = 1.0f / (float) Math.sqrt(ss / size + EPSILON);

        // Step 3: Normalize and scale (HAT dispatch - element-wise parallelizable)
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
            dispatchNormalize(cc, input, weight, invRms, size)
        );
    }

    @Reflect
    public static void dispatchNormalize(ComputeContext cc, F32Array input, F32Array weight, float invRms, int size) {
        cc.dispatchKernel(NDRange.of1D(size), kc -> normalizeKernel(kc, input, weight, invRms));
    }

    @Reflect
    public static void normalizeKernel(KernelContext kc, F32Array input, F32Array weight, float invRms) {
        int i = kc.gix;
        input.array(i, input.array(i) * invRms * weight.array(i));
    }
}
