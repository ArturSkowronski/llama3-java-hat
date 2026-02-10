package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.buffer.F32Array;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class RMSNormTest {

    private Accelerator accelerator;
    private RMSNorm rmsNorm;

    @BeforeEach
    void setUp() {
        accelerator = new Accelerator(MethodHandles.lookup());
        rmsNorm = new RMSNorm(accelerator);
    }

    @Test
    void testRMSNormSimple() {
        int size = 4;
        F32Array input = F32Array.create(accelerator, size);
        F32Array weight = F32Array.create(accelerator, size);

        // Input: [1.0, 2.0, 3.0, 4.0]
        for (int i = 0; i < size; i++) {
            input.array(i, i + 1.0f);
            weight.array(i, 1.0f); // Identity weight
        }

        // Mean of squares: (1+4+9+16)/4 = 30/4 = 7.5
        // RMS = sqrt(7.5 + 1e-5) ≈ 2.7386
        // Expected: input / RMS
        
        rmsNorm.apply(input, weight, size);

        float sumOfSquares = 0.0f;
        for (int i = 0; i < size; i++) {
            sumOfSquares += input.array(i) * input.array(i);
        }
        
        // After RMSNorm, the mean of squares should be approximately 1.0
        float meanOfSquares = sumOfSquares / size;
        assertEquals(1.0f, meanOfSquares, 1e-5f);
    }

    @Test
    void testRMSNormWithWeights() {
        int size = 2;
        F32Array input = F32Array.create(accelerator, size);
        F32Array weight = F32Array.create(accelerator, size);

        input.array(0, 3.0f);
        input.array(1, 4.0f);
        
        weight.array(0, 0.5f);
        weight.array(1, 2.0f);

        // SS = 3^2 + 4^2 = 9 + 16 = 25
        // Mean SS = 25 / 2 = 12.5
        // RMS = sqrt(12.5) ≈ 3.5355
        // Normalized: [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
        // Scaled: [0.8485 * 0.5, 1.1314 * 2.0] ≈ [0.4243, 2.2627]

        rmsNorm.apply(input, weight, size);

        assertEquals(3.0f / (float)Math.sqrt(12.5) * 0.5f, input.array(0), 1e-5f);
        assertEquals(4.0f / (float)Math.sqrt(12.5) * 2.0f, input.array(1), 1e-5f);
    }
}
