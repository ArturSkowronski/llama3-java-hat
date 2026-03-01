package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.types.F16;
import jdk.incubator.code.Reflect;
import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.RW;
import static optkl.ifacemapper.MappableIface.WO;

/**
 * GEMV (Matrix-Vector Multiplication) kernel using HAT @Reflect dispatch.
 * Supports both F32 and F16 weight matrices.
 * <p>
 * Computes: y = Ax
 * where A is a matrix [rows, cols] and x is a vector [cols].
 * Parallelization: Each row computed independently (NDRange.of1D(rows))
 * <p>
 * Usage: ~113 GEMV operations per token:
 * - 4 per layer for Q/K/V/O projections (64 total for 16 layers)
 * - 3 per layer for FFN gate/up/down (48 total for 16 layers)
 * - 1 final classifier projection (128256x2048 - largest matrix)
 */
public class GEMVHAT implements IGEMV {

    private final Accelerator accelerator;

    public GEMVHAT(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    @Override
    public void apply(F32Array matrix, F32Array vector, F32Array result, int rows, int cols) {
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
            dispatchGEMV(cc, matrix, vector, result, rows, cols)
        );
    }

    @Override
    public void apply(F16Array matrix, F32Array vector, F32Array result, int rows, int cols) {
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
            dispatchGEMVF16(cc, matrix, vector, result, rows, cols)
        );
    }

    @Reflect
    public static void dispatchGEMV(@RO ComputeContext cc, @RO F32Array matrix, @RO F32Array vector, @WO F32Array result, @RO int rows, @RO int cols) {
        cc.dispatchKernel(NDRange.of1D(rows), kc -> gemvKernel(kc, matrix, vector, result, cols));
    }

    @Reflect
    public static void dispatchGEMVF16(@RO ComputeContext cc, @RO F16Array matrix, @RO F32Array vector, @WO F32Array result, @RO int rows, @RO int cols) {
        cc.dispatchKernel(NDRange.of1D(rows), kc -> gemvKernelF16(kc, matrix, vector, result, cols));
    }

    @Reflect
    public static void gemvKernel(@RO KernelContext kc, @RO F32Array matrix, @RO F32Array vector, @WO F32Array result, @RO int cols) {
        GEMV.gemvKernel(kc, matrix, vector, result, cols);
    }

    @Reflect
    public static void gemvKernelF16(@RO KernelContext kc, @RO F16Array matrix, @RO F32Array vector, @WO F32Array result, @RO int cols) {
        GEMV.gemvKernelF16(kc, matrix, vector, result, cols);
    }
}
