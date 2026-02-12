package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Compare HAT @Reflect SiLU dispatch vs plain Java at various scales.
 * Goal: find if HAT dispatch works for a single in-place element-wise kernel.
 */
public class SiLUHatVsJavaTest {

    private Accelerator accelerator;

    @BeforeEach
    void setUp() {
        accelerator = new Accelerator(MethodHandles.lookup());
    }

    // ──── HAT dispatch (calls SiLU's @Reflect kernel directly) ────

    private void hatSiLU(F32Array input, int size) {
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
                SiLU.dispatchSiLU(cc, input, size));
    }

    // ──── Plain Java reference ────

    private static void javaSiLU(F32Array input, int size) {
        for (int i = 0; i < size; i++) {
            float x = input.array(i);
            input.array(i, x / (1.0f + (float) Math.exp(-x)));
        }
    }

    // ──── Helpers ────

    private F32Array createAndFill(int size, float startVal) {
        F32Array buf = F32Array.create(accelerator, size);
        for (int i = 0; i < size; i++) {
            buf.array(i, startVal + i * 0.01f);
        }
        return buf;
    }

    private void compareResults(String label, F32Array hatBuf, F32Array javaBuf, int size) {
        float maxErr = 0;
        int maxErrIdx = -1;
        for (int i = 0; i < size; i++) {
            float err = Math.abs(hatBuf.array(i) - javaBuf.array(i));
            if (err > maxErr) {
                maxErr = err;
                maxErrIdx = i;
            }
        }
        System.out.printf("[%s] size=%d  maxErr=%.8f at idx=%d  hat[0]=%.6f java[0]=%.6f%n",
                label, size, maxErr, maxErrIdx, hatBuf.array(0), javaBuf.array(0));

        // Hard assert: results must match within float precision
        for (int i = 0; i < size; i++) {
            assertEquals(javaBuf.array(i), hatBuf.array(i), 1e-5f,
                    String.format("%s: mismatch at index %d (hat=%.6f java=%.6f)",
                            label, i, hatBuf.array(i), javaBuf.array(i)));
        }
    }

    // ──── Tests ────

    @Test
    void testTiny_3elements() {
        int size = 3;
        F32Array hatBuf = createAndFill(size, -1.0f);
        F32Array javaBuf = createAndFill(size, -1.0f);

        hatSiLU(hatBuf, size);
        javaSiLU(javaBuf, size);

        compareResults("tiny", hatBuf, javaBuf, size);
    }

    @Test
    void testSmall_64elements() {
        int size = 64;
        F32Array hatBuf = createAndFill(size, -2.0f);
        F32Array javaBuf = createAndFill(size, -2.0f);

        hatSiLU(hatBuf, size);
        javaSiLU(javaBuf, size);

        compareResults("small_64", hatBuf, javaBuf, size);
    }

    @Test
    void testMedium_2048elements() {
        int size = 2048;
        F32Array hatBuf = createAndFill(size, -5.0f);
        F32Array javaBuf = createAndFill(size, -5.0f);

        hatSiLU(hatBuf, size);
        javaSiLU(javaBuf, size);

        compareResults("medium_2048", hatBuf, javaBuf, size);
    }

    @Test
    void testLarge_8192elements() {
        int size = 8192;
        F32Array hatBuf = createAndFill(size, -10.0f);
        F32Array javaBuf = createAndFill(size, -10.0f);

        hatSiLU(hatBuf, size);
        javaSiLU(javaBuf, size);

        compareResults("large_8192", hatBuf, javaBuf, size);
    }

    /**
     * KNOWN BUG: HAT dispatch with a NEW F32Array after a previous dispatch
     * does not execute the kernel — returns unmodified input data.
     *
     * This documents the HAT Sequential Backend data visibility bug.
     * Workaround: reuse the same F32Array buffer (see testRefillAndRedispatch).
     */
    @Test
    void testTwoConsecutiveDispatches_knownBug() {
        int size = 2048;

        // First dispatch: buffer with values starting at 1.0
        F32Array hatBuf1 = createAndFill(size, 1.0f);
        F32Array javaBuf1 = createAndFill(size, 1.0f);
        hatSiLU(hatBuf1, size);
        javaSiLU(javaBuf1, size);
        compareResults("dispatch_1", hatBuf1, javaBuf1, size);

        // Second dispatch: NEW buffer with values starting at -3.0
        F32Array hatBuf2 = createAndFill(size, -3.0f);
        F32Array javaBuf2 = createAndFill(size, -3.0f);
        hatSiLU(hatBuf2, size);
        javaSiLU(javaBuf2, size);

        // BUG: HAT returns unmodified input (-3.0) instead of SiLU(-3.0) = -0.142
        float hatVal = hatBuf2.array(0);
        float javaVal = javaBuf2.array(0);
        boolean bugPresent = Math.abs(hatVal - javaVal) > 0.1f;
        System.out.printf("[known_bug] dispatch_2: hat=%.6f java=%.6f bugPresent=%s%n",
                hatVal, javaVal, bugPresent);
        if (bugPresent) {
            System.out.println("CONFIRMED: HAT dispatch with new F32Array ignores kernel (returns raw input)");
        }
    }

    /**
     * Refill same buffer and re-dispatch.
     * Does HAT see the new values?
     */
    @Test
    void testRefillAndRedispatch() {
        int size = 2048;

        F32Array hatBuf = F32Array.create(accelerator, size);
        F32Array javaBuf = F32Array.create(accelerator, size);

        // Fill with value A and dispatch
        for (int i = 0; i < size; i++) {
            hatBuf.array(i, 1.0f + i * 0.001f);
            javaBuf.array(i, 1.0f + i * 0.001f);
        }
        hatSiLU(hatBuf, size);
        javaSiLU(javaBuf, size);
        float hatResultA = hatBuf.array(0);
        float javaResultA = javaBuf.array(0);
        System.out.printf("After dispatch A: hat=%.6f java=%.6f%n", hatResultA, javaResultA);

        // Refill SAME buffers with different values and re-dispatch
        for (int i = 0; i < size; i++) {
            hatBuf.array(i, -2.0f + i * 0.001f);
            javaBuf.array(i, -2.0f + i * 0.001f);
        }
        hatSiLU(hatBuf, size);
        javaSiLU(javaBuf, size);
        float hatResultB = hatBuf.array(0);
        float javaResultB = javaBuf.array(0);
        System.out.printf("After dispatch B: hat=%.6f java=%.6f%n", hatResultB, javaResultB);

        // Results should be different (different input values)
        System.out.printf("Results differ? hat: %.6f vs %.6f, java: %.6f vs %.6f%n",
                hatResultA, hatResultB, javaResultA, javaResultB);

        compareResults("refill_B", hatBuf, javaBuf, size);
    }

    /**
     * Key insight from testRefillAndRedispatch: reusing the SAME F32Array works.
     * Key insight from testTwoConsecutiveDispatches: NEW F32Array doesn't.
     *
     * In TransformerBlock, SiLU always operates on ffn1Out — the SAME buffer.
     * So SiLU should work with HAT dispatch if we reuse the buffer!
     */
    @Test
    void testReuseSameBufferAcrossMultipleDispatches() {
        int size = 8192; // INTERMEDIATE_SIZE in Llama 3.2 1B

        F32Array hatBuf = F32Array.create(accelerator, size);
        F32Array javaBuf = F32Array.create(accelerator, size);

        // Dispatch 1: fill with positive values
        for (int i = 0; i < size; i++) {
            hatBuf.array(i, 1.0f + i * 0.001f);
            javaBuf.array(i, 1.0f + i * 0.001f);
        }
        hatSiLU(hatBuf, size);
        javaSiLU(javaBuf, size);
        compareResults("reuse_dispatch_1", hatBuf, javaBuf, size);

        // Dispatch 2: refill SAME buffer with negative values
        for (int i = 0; i < size; i++) {
            hatBuf.array(i, -3.0f + i * 0.001f);
            javaBuf.array(i, -3.0f + i * 0.001f);
        }
        hatSiLU(hatBuf, size);
        javaSiLU(javaBuf, size);
        compareResults("reuse_dispatch_2", hatBuf, javaBuf, size);

        // Dispatch 3: refill again with different values
        for (int i = 0; i < size; i++) {
            hatBuf.array(i, 0.5f + i * 0.0001f);
            javaBuf.array(i, 0.5f + i * 0.0001f);
        }
        hatSiLU(hatBuf, size);
        javaSiLU(javaBuf, size);
        compareResults("reuse_dispatch_3", hatBuf, javaBuf, size);
    }

    /**
     * Simulate real TransformerBlock usage: HAT SiLU then plain Java reads.
     * This checks if HAT writes are visible to subsequent plain Java reads.
     */
    @Test
    void testHatWriteVisibleToPlainJavaRead() {
        int size = 8192; // INTERMEDIATE_SIZE in Llama 3.2 1B

        F32Array buf = createAndFill(size, 0.5f);

        // Compute expected values
        float[] expected = new float[size];
        for (int i = 0; i < size; i++) {
            float x = 0.5f + i * 0.01f;
            expected[i] = x / (1.0f + (float) Math.exp(-x));
        }

        // Dispatch via HAT
        hatSiLU(buf, size);

        // Read back via plain Java — are HAT writes visible?
        float maxErr = 0;
        for (int i = 0; i < size; i++) {
            float err = Math.abs(buf.array(i) - expected[i]);
            if (err > maxErr) maxErr = err;
        }
        System.out.printf("[hat_write_visibility] size=%d maxErr=%.8f buf[0]=%.6f expected[0]=%.6f%n",
                size, maxErr, buf.array(0), expected[0]);

        for (int i = 0; i < size; i++) {
            assertEquals(expected[i], buf.array(i), 1e-5f,
                    String.format("HAT write not visible at index %d (hat=%.6f expected=%.6f)",
                            i, buf.array(i), expected[i]));
        }
    }

    /**
     * Simulate exact TransformerBlock SiLU usage:
     * 1. GEMV fills ffn1Out (plain Java write)
     * 2. HAT SiLU modifies ffn1Out in-place
     * 3. elementWiseMul reads ffn1Out (plain Java read)
     *
     * Repeat this pattern multiple times (simulating multiple layers).
     */
    @Test
    void testSimulateTransformerBlockSiLUUsage() {
        int size = 8192; // INTERMEDIATE_SIZE

        F32Array ffn1Out = F32Array.create(accelerator, size);
        F32Array ffn3Out = F32Array.create(accelerator, size);

        for (int layer = 0; layer < 3; layer++) {
            // Step 1: Simulate GEMV filling ffn1Out (plain Java write)
            for (int i = 0; i < size; i++) {
                ffn1Out.array(i, (layer + 1) * 0.5f + i * 0.001f);
                ffn3Out.array(i, 0.1f + i * 0.0001f);
            }

            // Compute expected SiLU result
            float[] expectedSilu = new float[size];
            for (int i = 0; i < size; i++) {
                float x = ffn1Out.array(i);
                expectedSilu[i] = x / (1.0f + (float) Math.exp(-x));
            }

            // Step 2: HAT SiLU dispatch
            hatSiLU(ffn1Out, size);

            // Step 3: Simulate elementWiseMul (plain Java read of SiLU output)
            float maxErr = 0;
            for (int i = 0; i < size; i++) {
                float siluResult = ffn1Out.array(i);
                float err = Math.abs(siluResult - expectedSilu[i]);
                if (err > maxErr) maxErr = err;
                // Do the multiply (like TransformerBlock does)
                ffn1Out.array(i, siluResult * ffn3Out.array(i));
            }
            System.out.printf("[layer_%d] maxErr=%.8f silu[0]=%.6f expected=%.6f%n",
                    layer, maxErr, ffn1Out.array(0), expectedSilu[0]);

            assertEquals(0.0f, maxErr, 1e-5f,
                    String.format("Layer %d: HAT SiLU output not visible to plain Java (maxErr=%.8f)", layer, maxErr));
        }
    }
}
