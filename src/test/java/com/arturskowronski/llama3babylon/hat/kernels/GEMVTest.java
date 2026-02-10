package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.buffer.F32Array;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class GEMVTest {

    @Test
    public void testGEMV() {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup());
        GEMV gemv = new GEMV(accelerator);

        int rows = 4;
        int cols = 3;

        F32Array matrix = F32Array.create(accelerator, rows * cols);
        F32Array vector = F32Array.create(accelerator, cols);
        F32Array result = F32Array.create(accelerator, rows);

        // Matrix:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        // 10 11 12
        for (int i = 0; i < rows * cols; i++) {
            matrix.array(i, (float) (i + 1));
        }

        // Vector:
        // 1
        // 1
        // 1
        for (int i = 0; i < cols; i++) {
            vector.array(i, 1.0f);
        }

        gemv.apply(matrix, vector, result, rows, cols);

        // Expected result:
        // 1*1 + 2*1 + 3*1 = 6
        // 4*1 + 5*1 + 6*1 = 15
        // 7*1 + 8*1 + 9*1 = 24
        // 10*1 + 11*1 + 12*1 = 33
        assertEquals(6.0f, result.array(0), 1e-5f);
        assertEquals(15.0f, result.array(1), 1e-5f);
        assertEquals(24.0f, result.array(2), 1e-5f);
        assertEquals(33.0f, result.array(3), 1e-5f);
    }
}
