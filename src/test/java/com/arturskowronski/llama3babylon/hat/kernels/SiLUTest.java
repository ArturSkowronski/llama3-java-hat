package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.buffer.F32Array;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SiLUTest {

    private Accelerator accelerator;
    private SiLU silu;

    @BeforeEach
    void setUp() {
        accelerator = new Accelerator(MethodHandles.lookup());
        silu = new SiLU(accelerator);
    }

    @Test
    void testSiLU() {
        int size = 3;
        F32Array input = F32Array.create(accelerator, size);

        input.array(0, -1.0f);
        input.array(1, 0.0f);
        input.array(2, 1.0f);

        silu.apply(input, size);

        // SiLU(x) = x / (1 + exp(-x))
        // SiLU(-1) = -1 / (1 + exp(1)) ≈ -1 / 3.718 ≈ -0.2689
        // SiLU(0) = 0 / (1 + 1) = 0
        // SiLU(1) = 1 / (1 + exp(-1)) ≈ 1 / 1.3678 ≈ 0.7310

        assertEquals(-1.0f / (1.0f + (float) Math.exp(1.0f)), input.array(0), 1e-6f);
        assertEquals(0.0f, input.array(1), 1e-6f);
        assertEquals(1.0f / (1.0f + (float) Math.exp(-1.0f)), input.array(2), 1e-6f);
    }
}
