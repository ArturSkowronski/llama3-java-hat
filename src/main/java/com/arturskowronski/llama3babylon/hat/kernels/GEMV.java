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
import static optkl.ifacemapper.MappableIface.WO;

/**
 * GEMV (Matrix-Vector Multiplication) kernel for Llama 3.2 1B Instruct.
 *
 * Computes: y = Ax
 * where A is a matrix [rows, cols] and x is a vector [cols].
 * Supports both F32 and F16 weight matrices.
 */
public class GEMV implements IGEMV {

    private final Accelerator accelerator;

    public GEMV(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    @Override
    public void apply(F32Array matrix, F32Array vector, F32Array result, int rows, int cols) {
        for (int row = 0; row < rows; row++) {
            float sum = 0.0f;
            int rowOffset = row * cols;
            for (int c = 0; c < cols; c++) {
                sum += matrix.array(rowOffset + c) * vector.array(c);
            }
            result.array(row, sum);
        }
    }

    @Override
    public void apply(F16Array matrix, F32Array vector, F32Array result, int rows, int cols) {
        for (int row = 0; row < rows; row++) {
            float sum = 0.0f;
            int rowOffset = row * cols;
            for (int c = 0; c < cols; c++) {
                sum += F16.f16ToFloat(matrix.array(rowOffset + c)) * vector.array(c);
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
    public static void gemvKernelF16(@RO KernelContext kc, @RO F16Array matrix, @RO F32Array vector, @WO F32Array result, @RO int cols) {
        int row = kc.gix;
        float sum = 0.0f;
        int rowOffset = row * cols;
        for (int c = 0; c < cols; c++) {
            sum += F16.f16ToFloat(matrix.array(rowOffset + c)) * vector.array(c);
        }
        result.array(row, sum);
    }

    @Reflect
    public static void dispatchGEMV(@RO ComputeContext cc, @RO F32Array matrix, @RO F32Array vector, @WO F32Array result, @RO int rows, @RO int cols) {
        cc.dispatchKernel(NDRange.of1D(rows), kc -> gemvKernel(kc, matrix, vector, result, cols));
    }

    @Reflect
    public static void dispatchGEMVF16(@RO ComputeContext cc, @RO F16Array matrix, @RO F32Array vector, @WO F32Array result, @RO int rows, @RO int cols) {
        cc.dispatchKernel(NDRange.of1D(rows), kc -> gemvKernelF16(kc, matrix, vector, result, cols));
    }
}
