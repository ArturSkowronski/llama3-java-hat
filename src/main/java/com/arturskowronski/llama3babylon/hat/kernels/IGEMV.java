package com.arturskowronski.llama3babylon.hat.kernels;

import hat.buffer.F32Array;

/**
 * Interface for GEMV (Matrix-Vector Multiplication) kernel implementations.
 */
public interface IGEMV {

    /**
     * Computes Matrix-Vector multiplication y = Ax.
     *
     * @param matrix input matrix A [rows, cols]
     * @param vector input vector x [cols]
     * @param result output vector y [rows]
     * @param rows number of rows in matrix
     * @param cols number of columns in matrix
     */
    void apply(F32Array matrix, F32Array vector, F32Array result, int rows, int cols);
}
