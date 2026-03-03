package com.arturskowronski.llama3babylon.hat;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class F16WeightsTest {

    @Test
    public void testConstructorAndAccessors() {
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        short[] data = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            data[i] = Float.floatToFloat16(values[i]);
        }

        F16Weights weights = new F16Weights(data, 2, 3);
        assertEquals(2, weights.rows());
        assertEquals(3, weights.cols());
        assertEquals(6, weights.length());
        assertSame(data, weights.data());
    }

    @Test
    public void testConstructorRejectsShapeMismatch() {
        short[] data = new short[6];
        assertThrows(IllegalArgumentException.class, () -> new F16Weights(data, 2, 4));
    }

    @Test
    public void testGetFloat() {
        float[] values = {1.5f, 2.5f, 3.5f, 4.5f};
        short[] data = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            data[i] = Float.floatToFloat16(values[i]);
        }

        F16Weights weights = new F16Weights(data, 2, 2);
        for (int i = 0; i < values.length; i++) {
            assertEquals(values[i], weights.getFloat(i), 0.01f, "Mismatch at index " + i);
        }
    }

    @Test
    public void testDequantRow() {
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        short[] data = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            data[i] = Float.floatToFloat16(values[i]);
        }

        F16Weights weights = new F16Weights(data, 2, 3);

        float[] row0 = new float[3];
        weights.dequantRow(0, row0);
        assertEquals(1.0f, row0[0], 0.01f);
        assertEquals(2.0f, row0[1], 0.01f);
        assertEquals(3.0f, row0[2], 0.01f);

        float[] row1 = new float[3];
        weights.dequantRow(1, row1);
        assertEquals(4.0f, row1[0], 0.01f);
        assertEquals(5.0f, row1[1], 0.01f);
        assertEquals(6.0f, row1[2], 0.01f);
    }

    @Test
    public void testDequantRowMatchesGetFloat() {
        // Verify dequantRow produces same results as element-by-element getFloat
        short[] data = new short[12];
        for (int i = 0; i < 12; i++) {
            data[i] = Float.floatToFloat16((float) (i * 0.1));
        }

        F16Weights weights = new F16Weights(data, 3, 4);
        float[] rowBuf = new float[4];

        for (int row = 0; row < 3; row++) {
            weights.dequantRow(row, rowBuf);
            for (int col = 0; col < 4; col++) {
                assertEquals(weights.getFloat(row * 4 + col), rowBuf[col], 0.0f,
                        "Mismatch at row=" + row + " col=" + col);
            }
        }
    }

    @Test
    public void testF16PrecisionRoundTrip() {
        // Values that are exactly representable in F16
        float[] exact = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 65504.0f};
        short[] data = new short[exact.length];
        for (int i = 0; i < exact.length; i++) {
            data[i] = Float.floatToFloat16(exact[i]);
        }

        F16Weights weights = new F16Weights(data, 1, exact.length);
        for (int i = 0; i < exact.length; i++) {
            assertEquals(exact[i], weights.getFloat(i), 0.0f, "Exact F16 value at " + i);
        }
    }
}
