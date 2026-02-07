package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.buffer.F32Array;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;

import static org.junit.jupiter.api.Assertions.*;

public class SoftmaxTest {

    private Accelerator accelerator;
    private Softmax softmax;

    @BeforeEach
    void setUp() {
        accelerator = new Accelerator(MethodHandles.lookup());
        softmax = new Softmax(accelerator);
    }

    @Test
    void testSoftmaxSumsToOne() {
        int size = 5;
        F32Array input = F32Array.create(accelerator, size);
        
        // Initialize with [1.0, 2.0, 3.0, 4.0, 5.0]
        for (int i = 0; i < size; i++) {
            input.array(i, (float) (i + 1));
        }

        softmax.apply(input, size);

        // Softmax outputs should sum to 1.0
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += input.array(i);
        }
        assertEquals(1.0f, sum, 0.0001f, "Softmax outputs should sum to 1.0");
    }

    @Test
    void testSoftmaxValues() {
        int size = 5;
        F32Array input = F32Array.create(accelerator, size);
        
        // Initialize with [1.0, 2.0, 3.0, 4.0, 5.0]
        for (int i = 0; i < size; i++) {
            input.array(i, (float) (i + 1));
        }

        softmax.apply(input, size);

        // Expected values for softmax([1,2,3,4,5])
        float[] expected = {0.01165f, 0.03168f, 0.08612f, 0.23412f, 0.63640f};
        for (int i = 0; i < size; i++) {
            assertEquals(expected[i], input.array(i), 0.001f, 
                "Softmax value at index " + i + " should match expected");
        }
    }

    @Test
    void testSoftmaxMonotonicity() {
        // Larger input values should produce larger softmax outputs
        int size = 4;
        F32Array input = F32Array.create(accelerator, size);
        
        input.array(0, 1.0f);
        input.array(1, 2.0f);
        input.array(2, 3.0f);
        input.array(3, 4.0f);

        softmax.apply(input, size);

        // Verify monotonicity: output[i] < output[i+1]
        for (int i = 0; i < size - 1; i++) {
            assertTrue(input.array(i) < input.array(i + 1),
                "Softmax should preserve ordering: output[" + i + "] < output[" + (i+1) + "]");
        }
    }

    @Test
    void testSoftmaxRowApply() {
        // Test applying softmax to a specific row in a larger array
        int totalSize = 10;
        int rowOffset = 3;
        int rowSize = 4;
        
        F32Array input = F32Array.create(accelerator, totalSize);
        
        // Fill entire array with zeros
        for (int i = 0; i < totalSize; i++) {
            input.array(i, 0.0f);
        }
        
        // Set row values: [1.0, 2.0, 3.0, 4.0] at offset 3
        for (int i = 0; i < rowSize; i++) {
            input.array(rowOffset + i, (float) (i + 1));
        }

        softmax.applyRow(input, rowOffset, rowSize);

        // Verify row sums to 1.0
        float rowSum = 0.0f;
        for (int i = 0; i < rowSize; i++) {
            rowSum += input.array(rowOffset + i);
        }
        assertEquals(1.0f, rowSum, 0.0001f, "Softmax row should sum to 1.0");

        // Verify elements outside the row are unchanged
        for (int i = 0; i < rowOffset; i++) {
            assertEquals(0.0f, input.array(i), 0.0001f, 
                "Elements before row should be unchanged");
        }
        for (int i = rowOffset + rowSize; i < totalSize; i++) {
            assertEquals(0.0f, input.array(i), 0.0001f, 
                "Elements after row should be unchanged");
        }
    }

    @Test
    void testSoftmaxNumericalStability() {
        // Test with large values that could cause overflow without max subtraction
        int size = 3;
        F32Array input = F32Array.create(accelerator, size);
        
        input.array(0, 1000.0f);
        input.array(1, 1001.0f);
        input.array(2, 1002.0f);

        softmax.apply(input, size);

        // Should still sum to 1.0 and not produce NaN/Inf
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            float val = input.array(i);
            assertFalse(Float.isNaN(val), "Softmax should not produce NaN");
            assertFalse(Float.isInfinite(val), "Softmax should not produce Inf");
            sum += val;
        }
        assertEquals(1.0f, sum, 0.0001f, "Softmax outputs should sum to 1.0 even with large inputs");
    }
}
