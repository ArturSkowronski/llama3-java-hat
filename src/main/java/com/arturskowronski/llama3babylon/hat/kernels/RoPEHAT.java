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
 * RoPE (Rotary Positional Embedding) kernel using HAT @Reflect dispatch.
 *
 * Second kernel to be tested with HAT dispatch in real 16-layer inference.
 * RoPE applies rotations to query and key vectors, with each head processed independently.
 * This makes it naturally parallelizable and ideal for HAT acceleration.
 *
 * Formula: Applies 2D rotations to pairs of elements based on token position.
 * For each head, for each pair (i, i+1):
 *   freq = 1 / (theta ^ (i / head_dim))
 *   angle = pos * freq
 *   out[i]   = v[i] * cos(angle) - v[i+1] * sin(angle)
 *   out[i+1] = v[i] * sin(angle) + v[i+1] * cos(angle)
 */
public class RoPEHAT implements IRoPE {

    private final Accelerator accelerator;

    public RoPEHAT(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    /**
     * Applies RoPE to a vector (Q or K) using HAT dispatch.
     *
     * @param vec input/output vector [num_heads, head_dim] (modified in-place)
     * @param pos current token position in sequence
     * @param numHeads number of heads
     * @param headDim dimension of each head
     * @param theta base for frequency calculation (typically 10000.0 or 500000.0 for Llama 3)
     */
    @Override
    public void apply(F32Array vec, int pos, int numHeads, int headDim, float theta) {
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
            dispatchRoPE(cc, vec, pos, numHeads, headDim, theta)
        );
    }

    @Reflect
    public static void dispatchRoPE(@RO ComputeContext cc, @RW F32Array vec, @RO int pos, @RO int numHeads, @RO int headDim, @RO float theta) {
        cc.dispatchKernel(NDRange.of1D(numHeads), kc -> ropeKernel(kc, vec, pos, headDim, theta));
    }

    @Reflect
    public static void ropeKernel(@RO KernelContext kc, @RW F32Array vec, @RO int pos, @RO int headDim, @RO float theta) {
        int h = kc.gix; // head index
        int headOffset = h * headDim;

        for (int i = 0; i < headDim; i += 2) {
            float freq = (float) (1.0 / Math.pow(theta, (double) i / headDim));
            float val = pos * freq;
            float cos = (float) Math.cos(val);
            float sin = (float) Math.sin(val);

            float v0 = vec.array(headOffset + i);
            float v1 = vec.array(headOffset + i + 1);

            vec.array(headOffset + i, v0 * cos - v1 * sin);
            vec.array(headOffset + i + 1, v0 * sin + v1 * cos);
        }
    }
}
