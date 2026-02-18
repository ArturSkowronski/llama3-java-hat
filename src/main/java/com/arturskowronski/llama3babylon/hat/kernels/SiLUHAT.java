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
 * SiLU (Sigmoid Linear Unit) kernel using HAT @Reflect dispatch.
 *
 * This is the first kernel to be tested with HAT dispatch in real 16-layer inference.
 * SiLU is element-wise with no inter-element dependencies, making it ideal for testing
 * HAT buffer reuse patterns in the full inference pipeline.
 *
 * Formula: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
public class SiLUHAT implements ISiLU {

    private final Accelerator accelerator;

    public SiLUHAT(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    /**
     * Applies SiLU to the input array in-place using HAT dispatch.
     *
     * @param input the input array (modified in-place)
     * @param size the number of elements to process
     */
    @Override
    public void apply(F32Array input, int size) {
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
            dispatchSiLU(cc, input, size)
        );
    }

    @Reflect
    public static void dispatchSiLU(@RO ComputeContext cc, @RW F32Array input, @RO int size) {
        cc.dispatchKernel(NDRange.of1D(size), kc -> siluKernel(kc, input));
    }

    @Reflect
    public static void siluKernel(@RO KernelContext kc, @RW F32Array input) {
        int i = kc.gix;
        float x = input.array(i);
        input.array(i, x / (1.0f + (float) Math.exp(-x)));
    }
}
