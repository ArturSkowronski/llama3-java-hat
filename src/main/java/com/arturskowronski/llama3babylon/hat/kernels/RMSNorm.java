package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;

/**
 * RMSNorm (Root Mean Square Layer Normalization) kernel for Llama 3.2 1B.
 * 
 * Formula: y = (x / RMS(x)) * weight
 * where RMS(x) = sqrt(1/n * sum(x_i^2) + epsilon)
 */
public class RMSNorm {

    private final Accelerator accelerator;
    private static final float EPSILON = 1e-5f;

    public RMSNorm(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    /**
     * Applies RMSNorm to the input array in-place.
     * 
     * @param input the input array (modified in-place)
     * @param weight the weight tensor
     * @param size the size of the hidden dimension
     */
    public void apply(F32Array input, F32Array weight, int size) {
        // Step 1: Compute sum of squares on CPU (for now, consistent with Softmax approach)
        float ss = 0.0f;
        for (int i = 0; i < size; i++) {
            float val = input.array(i);
            ss += val * val;
        }

        float invRms = 1.0f / (float) Math.sqrt(ss / size + EPSILON);

        // Step 2: Normalize and scale on GPU
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
                dispatchNormalize(cc, input, weight, invRms, size)
        );
    }

    @Reflect
    public static void normalizeKernel(KernelContext kc, F32Array input, F32Array weight, float invRms) {
        int i = kc.gix;
        input.array(i, input.array(i) * invRms * weight.array(i));
    }

    @Reflect
    public static void dispatchNormalize(ComputeContext cc, F32Array input, F32Array weight, float invRms, int size) {
        cc.dispatchKernel(NDRange.of1D(size), kc -> normalizeKernel(kc, input, weight, invRms));
    }
}
