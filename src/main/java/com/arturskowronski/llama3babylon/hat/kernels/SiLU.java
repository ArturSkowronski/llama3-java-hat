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
 * SiLU (Sigmoid Linear Unit) kernel for Llama 3.2 1B.
 *
 * Formula: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
public class SiLU implements ISiLU {

    private final Accelerator accelerator;

    public SiLU(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    /**
     * Applies SiLU to the input array in-place.
     * 
     * @param input the input array (modified in-place)
     * @param size the number of elements to process
     */
    public void apply(F32Array input, int size) {
        // Plain Java — HAT @Reflect dispatch passes unit tests (same-buffer reuse)
        // but produces garbage in real 16-layer inference ({{{... instead of coherent text).
        // The data visibility bug is more subtle than buffer reuse alone.
        for (int i = 0; i < size; i++) {
            float x = input.array(i);
            input.array(i, x / (1.0f + (float) Math.exp(-x)));
        }
    }

    @Reflect
    public static void siluKernel(@RO KernelContext kc, @RW F32Array input) {
        int i = kc.gix;
        float x = input.array(i);
        input.array(i, x / (1.0f + (float) Math.exp(-x)));
    }

    @Reflect
    public static void dispatchSiLU(@RO ComputeContext cc, @RW F32Array input, @RO int size) {
        cc.dispatchKernel(NDRange.of1D(size), kc -> siluKernel(kc, input));
    }
}
