package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;
import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.RW;
import static optkl.ifacemapper.MappableIface.WO;

/**
 * GEMV (Matrix-Vector Multiplication) kernel using HAT @Reflect dispatch.
 * <p>
 * FINAL BOSS: Sixth and last kernel to be tested with HAT dispatch.
 * GEMV is the most computationally intensive kernel in the Llama inference pipeline.
 * <p>
 * Computes: y = Ax
 * where A is a matrix [rows, cols] and x is a vector [cols].
 * Parallelization: Each row computed independently (NDRange.of1D(rows))
 * <p>
 * Usage: ~113 GEMV operations per token:
 * - 4 per layer for Q/K/V/O projections (64 total for 16 layers)
 * - 3 per layer for FFN gate/up/down (48 total for 16 layers)
 * - 1 final classifier projection (128256×2048 - largest matrix)
 */
public class GEMVHAT implements IGEMV {

    private final Accelerator accelerator;

    public GEMVHAT(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    /**
     * Computes Matrix-Vector multiplication y = Ax using HAT dispatch.
     *
     * @param matrix input matrix A [rows, cols]
     * @param vector input vector x [cols]
     * @param result output vector y [rows]
     * @param rows number of rows in matrix
     * @param cols number of columns in matrix
     */
    @Override
    public void apply(F32Array matrix, F32Array vector, F32Array result, int rows, int cols) {
        accelerator.compute(cc ->
            dispatchGEMV(cc, matrix, vector, result, rows, cols)
        );
    }

    @Reflect
    public static void dispatchGEMV(@RO ComputeContext cc, @RO F32Array matrix, @RO F32Array vector, @WO F32Array result, @RO int rows, @RO int cols) {
        cc.dispatchKernel(NDRange.of1D(rows), kc -> gemvKernel(kc, matrix, vector, result, cols));
    }

    @Reflect
    public static void gemvKernel(@RO KernelContext kc, @RO F32Array matrix, @RO F32Array vector, @WO F32Array result, @RO int cols) {
        GEMV.gemvKernel(kc, matrix, vector, result, cols);
    }
}
