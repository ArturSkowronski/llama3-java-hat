package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;

/**
 * GEMV (Matrix-Vector Multiplication) kernel using HAT @Reflect dispatch.
 *
 * FINAL BOSS: Sixth and last kernel to be tested with HAT dispatch.
 * GEMV is the most computationally intensive kernel in the Llama inference pipeline.
 *
 * Computes: y = Ax
 * where A is a matrix [rows, cols] and x is a vector [cols].
 *
 * Parallelization: Each row computed independently (NDRange.of1D(rows))
 *
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
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
            dispatchGEMV(cc, matrix, vector, result, rows, cols)
        );
    }

    @Reflect
    public static void dispatchGEMV(ComputeContext cc, F32Array matrix, F32Array vector, F32Array result, int rows, int cols) {
        cc.dispatchKernel(NDRange.of1D(rows), kc -> gemvKernel(kc, matrix, vector, result, cols));
    }

    @Reflect
    public static void gemvKernel(KernelContext kc, F32Array matrix, F32Array vector, F32Array result, int cols) {
        int row = kc.gix;
        float sum = 0.0f;
        int rowOffset = row * cols;
        for (int c = 0; c < cols; c++) {
            sum += matrix.array(rowOffset + c) * vector.array(c);
        }
        result.array(row, sum);
    }
}
