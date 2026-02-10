package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.buffer.F32Array;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RoPETest {

    @Test
    public void testRoPE() {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup());
        RoPE rope = new RoPE(accelerator);

        int numHeads = 2;
        int headDim = 4;
        int pos = 1;
        float theta = 10000.0f;

        F32Array vec = F32Array.create(accelerator, numHeads * headDim);

        // Initialize vector: [1, 0, 1, 0, 1, 0, 1, 0]
        for (int i = 0; i < numHeads * headDim; i += 2) {
            vec.array(i, 1.0f);
            vec.array(i + 1, 0.0f);
        }

        rope.apply(vec, pos, numHeads, headDim, theta);

        // Expected for pos=1:
        // freq_i = 1.0 / (theta^(i/headDim))
        // i=0: freq_0 = 1.0 / 10000^0 = 1.0
        //      val = 1 * 1.0 = 1.0
        //      cos(1.0) = 0.5403023
        //      sin(1.0) = 0.84147098
        //      v0_new = 1*cos - 0*sin = cos(1.0)
        //      v1_new = 1*sin + 0*cos = sin(1.0)
        // i=2: freq_2 = 1.0 / 10000^(2/4) = 1.0 / 100 = 0.01
        //      val = 1 * 0.01 = 0.01
        //      cos(0.01) = 0.99995
        //      sin(0.01) = 0.0099998
        //      v2_new = 1*cos - 0*sin = cos(0.01)
        //      v3_new = 1*sin + 0*cos = sin(0.01)

        float cos1 = (float) Math.cos(1.0);
        float sin1 = (float) Math.sin(1.0);
        float cos001 = (float) Math.cos(0.01);
        float sin001 = (float) Math.sin(0.01);

        for (int h = 0; h < numHeads; h++) {
            int offset = h * headDim;
            assertEquals(cos1, vec.array(offset + 0), 1e-5f);
            assertEquals(sin1, vec.array(offset + 1), 1e-5f);
            assertEquals(cos001, vec.array(offset + 2), 1e-5f);
            assertEquals(sin001, vec.array(offset + 3), 1e-5f);
        }
    }
}
