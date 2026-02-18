package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;

import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.WO;

/**
 * GEMV (Matrix-Vector Multiplication) kernel for Llama 3.2 1B Instruct.
 *
 * Computes: y = Ax
 * where A is a matrix [rows, cols] and x is a vector [cols].
 */
public class GEMV implements IGEMV {

    private final Accelerator accelerator;

    public GEMV(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    /**
     * Computes Matrix-Vector multiplication y = Ax.
     * 
     * @param matrix input matrix A [rows, cols]
     * @param vector input vector x [cols]
     * @param result output vector y [rows]
     * @param rows number of rows in matrix
     * @param cols number of columns in matrix
     */
    public void apply(F32Array matrix, F32Array vector, F32Array result, int rows, int cols) {
        // Plain Java — HAT sequential backend dispatch produces incorrect results
        // when reading from weight matrices mapped from GGUF
        for (int row = 0; row < rows; row++) {
            float sum = 0.0f;
            int rowOffset = row * cols;
            for (int c = 0; c < cols; c++) {
                sum += matrix.array(rowOffset + c) * vector.array(c);
            }
            result.array(row, sum);
        }
    }

    @Reflect
    public static void gemvKernel(@RO KernelContext kc, @RO F32Array matrix, @RO F32Array vector, @WO F32Array result, @RO int cols) {
        int row = kc.gix;
        float sum = 0.0f;
        int rowOffset = row * cols;
        for (int c = 0; c < cols; c++) {
            sum += matrix.array(rowOffset + c) * vector.array(c);
        }
        result.array(row, sum);
    }

    @Reflect
    public static void dispatchGEMV(@RO ComputeContext cc, @RO F32Array matrix, @RO F32Array vector, @WO F32Array result, @RO int rows, @RO int cols) {
        cc.dispatchKernel(NDRange.of1D(rows), kc -> gemvKernel(kc, matrix, vector, result, cols));
    }
}
