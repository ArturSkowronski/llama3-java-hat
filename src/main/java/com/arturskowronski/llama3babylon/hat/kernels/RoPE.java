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
 * RoPE (Rotary Positional Embedding) kernel for Llama 3.2 1B Instruct.
 *
 * Specifically designed for HIDDEN_SIZE=2048, NUM_HEADS=32, HEAD_DIM=64.
 * RoPE is applied to Query and Key vectors.
 */
public class RoPE implements IRoPE {

    private final Accelerator accelerator;

    public RoPE(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    /**
     * Applies RoPE to a vector (Q or K).
     * 
     * @param vec input/output vector [num_heads, head_dim] (modified in-place)
     * @param pos current token position in sequence
     * @param numHeads number of heads
     * @param headDim dimension of each head
     * @param theta base for frequency calculation (typically 10000.0 or 500000.0 for Llama 3)
     */
    public void apply(F32Array vec, int pos, int numHeads, int headDim, float theta) {
        // Plain Java — HAT dispatch has buffer sync issues with subsequent
        // plain Java reads on the same buffer (e.g., Attention reading q/k)
        for (int h = 0; h < numHeads; h++) {
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

    @Reflect
    public static void dispatchRoPE(@RO ComputeContext cc, @RW F32Array vec, @RO int pos, @RO int numHeads, @RO int headDim, @RO float theta) {
        cc.dispatchKernel(NDRange.of1D(numHeads), kc -> ropeKernel(kc, vec, pos, headDim, theta));
    }
}
