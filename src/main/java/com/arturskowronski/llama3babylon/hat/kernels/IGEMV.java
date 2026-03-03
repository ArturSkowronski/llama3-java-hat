package com.arturskowronski.llama3babylon.hat.kernels;

import com.arturskowronski.llama3babylon.hat.F16Weights;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.types.F16;

/**
 * Interface for GEMV (Matrix-Vector Multiplication) kernel implementations.
 */
public interface IGEMV {

    /**
     * Computes Matrix-Vector multiplication y = Ax with F32 matrix.
     *
     * @param matrix input matrix A [rows, cols]
     * @param vector input vector x [cols]
     * @param result output vector y [rows]
     * @param rows number of rows in matrix
     * @param cols number of columns in matrix
     */
    void apply(F32Array matrix, F32Array vector, F32Array result, int rows, int cols);

    /**
     * Computes Matrix-Vector multiplication y = Ax with F16 matrix.
     * Each F16 element is converted to float during the dot product.
     *
     * @param matrix input matrix A [rows, cols] in half-precision
     * @param vector input vector x [cols] in single-precision
     * @param result output vector y [rows] in single-precision
     * @param rows number of rows in matrix
     * @param cols number of columns in matrix
     */
    default void apply(F16Array matrix, F32Array vector, F32Array result, int rows, int cols) {
        // Default fallback: dequantize row-by-row (suboptimal but correct)
        for (int row = 0; row < rows; row++) {
            float sum = 0.0f;
            int rowOffset = row * cols;
            for (int c = 0; c < cols; c++) {
                sum += F16.f16ToFloat(matrix.array(rowOffset + c)) * vector.array(c);
            }
            result.array(row, sum);
        }
    }

    /**
     * Computes Matrix-Vector multiplication y = Ax with F16Weights (CPU-optimized short[]).
     * Default fallback: row-by-row dequant + dot product on plain arrays.
     */
    default void apply(F16Weights matrix, F32Array vector, F32Array result, int rows, int cols) {
        short[] data = matrix.data();
        float[] rowBuf = new float[cols];
        float[] vecBuf = new float[cols];
        for (int c = 0; c < cols; c++) {
            vecBuf[c] = vector.array(c);
        }
        for (int row = 0; row < rows; row++) {
            int rowOffset = row * cols;
            for (int c = 0; c < cols; c++) {
                rowBuf[c] = Float.float16ToFloat(data[rowOffset + c]);
            }
            float sum = 0.0f;
            for (int c = 0; c < cols; c++) {
                sum += rowBuf[c] * vecBuf[c];
            }
            result.array(row, sum);
        }
    }
}
