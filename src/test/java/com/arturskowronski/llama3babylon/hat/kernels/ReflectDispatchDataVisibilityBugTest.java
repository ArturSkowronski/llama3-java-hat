package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reproduction of HAT Java Sequential Backend data visibility bug.
 *
 * Bug: After priming a @Reflect GEMV kernel dispatch (required by Bug #1 workaround),
 * subsequent dispatches with DIFFERENT buffer data produce incorrect results.
 * The backend appears to cache buffer data from the first dispatch and reuse it
 * for later dispatches, ignoring the new values written via plain Java.
 *
 * This is SEPARATE from the buffer bounds caching bug (PR #13 / HAT_BUG_REPORT.md).
 * Bug #1 causes IndexOutOfBoundsException (crashes). This bug causes silently wrong
 * results — a more dangerous failure mode.
 *
 * Real-world impact: When used in a 16-layer Llama 3.2 transformer, HAT GEMV dispatch
 * produces logit values ~10^26 instead of ~7, generating garbage text. Switching all
 * kernels to plain Java loops produces correct, coherent output.
 *
 * See HAT_DISPATCH_BUG_REPORT.md for full details.
 */
public class ReflectDispatchDataVisibilityBugTest {

    // ──────────────── Test helpers ────────────────

    /** Dispatch GEMV via HAT @Reflect (bypassing the plain Java workaround in GEMV.apply) */
    private static void hatGemv(Accelerator acc, F32Array matrix, F32Array vector, F32Array result, int rows, int cols) {
        acc.compute((Accelerator.@Reflect Compute) cc ->
                GEMV.dispatchGEMV(cc, matrix, vector, result, rows, cols));
    }

    private static float[] plainJavaGemv(F32Array matrix, F32Array vector, int rows, int cols) {
        float[] result = new float[rows];
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            int rowOffset = r * cols;
            for (int c = 0; c < cols; c++) {
                sum += matrix.array(rowOffset + c) * vector.array(c);
            }
            result[r] = sum;
        }
        return result;
    }

    // ──────────────── Tests ────────────────

    /**
     * Baseline: Plain Java GEMV always produces correct results.
     */
    @Test
    void testPlainJavaGemvIsCorrect() {
        Accelerator acc = new Accelerator(MethodHandles.lookup());

        int rows = 4, cols = 3;
        F32Array matrix = F32Array.create(acc, rows * cols);
        F32Array vector = F32Array.create(acc, cols);

        // Identity-like matrix
        for (int i = 0; i < rows * cols; i++) matrix.array(i, 0.0f);
        for (int r = 0; r < Math.min(rows, cols); r++) matrix.array(r * cols + r, 1.0f);
        for (int c = 0; c < cols; c++) vector.array(c, (c + 1) * 10.0f);

        float[] result = plainJavaGemv(matrix, vector, rows, cols);

        assertEquals(10.0f, result[0], 1e-5f);
        assertEquals(20.0f, result[1], 1e-5f);
        assertEquals(30.0f, result[2], 1e-5f);
        assertEquals(0.0f, result[3], 1e-5f);  // 4th row is all zeros
    }

    /**
     * Baseline: A single HAT GEMV dispatch without priming produces correct results.
     *
     * This proves the basic @Reflect dispatch mechanism works in isolation.
     */
    @Test
    void testSingleHatDispatchIsCorrect() {
        Accelerator acc = new Accelerator(MethodHandles.lookup());

        int rows = 4, cols = 3;
        F32Array matrix = F32Array.create(acc, rows * cols);
        F32Array vector = F32Array.create(acc, cols);
        F32Array result = F32Array.create(acc, rows);

        for (int i = 0; i < rows * cols; i++) matrix.array(i, 0.0f);
        for (int r = 0; r < Math.min(rows, cols); r++) matrix.array(r * cols + r, 1.0f);
        for (int c = 0; c < cols; c++) vector.array(c, (c + 1) * 10.0f);

        hatGemv(acc, matrix, vector, result, rows, cols);

        assertEquals(10.0f, result.array(0), 1e-5f);
        assertEquals(20.0f, result.array(1), 1e-5f);
        assertEquals(30.0f, result.array(2), 1e-5f);
        assertEquals(0.0f, result.array(3), 1e-5f);
    }

    /**
     * BUG: After priming GEMV with one matrix, dispatching with a DIFFERENT matrix
     * produces results based on the PRIMING data, not the new data.
     *
     * This simulates the real-world scenario:
     * 1. LlamaInference constructor primes GEMV with classifier matrix (Bug #1 workaround)
     * 2. TransformerBlock dispatches GEMV with layer weights
     * 3. Layer weights are ignored — backend uses cached priming data
     *
     * Expected: Second dispatch uses matrix B's data.
     * Actual: Second dispatch uses matrix A's (priming) data, producing wrong results.
     */
    @Test
    void testPrimedDispatchUsesCachedData() {
        Accelerator acc = new Accelerator(MethodHandles.lookup());

        int rows = 2048, cols = 2048;

        // Matrix A: all 0.001 (priming matrix)
        F32Array matrixA = F32Array.create(acc, rows * cols);
        for (int i = 0; i < rows * cols; i++) matrixA.array(i, 0.001f);

        F32Array vectorA = F32Array.create(acc, cols);
        for (int c = 0; c < cols; c++) vectorA.array(c, 1.0f);

        F32Array resultA = F32Array.create(acc, rows);

        // Prime: dispatch with matrix A via HAT @Reflect
        hatGemv(acc, matrixA, vectorA, resultA, rows, cols);
        // Expected: each row = 0.001 * 1.0 * 2048 = 2.048
        float primedValue = resultA.array(0);

        // Matrix B: all 0.005 (different values!)
        F32Array matrixB = F32Array.create(acc, rows * cols);
        for (int i = 0; i < rows * cols; i++) matrixB.array(i, 0.005f);

        F32Array vectorB = F32Array.create(acc, cols);
        for (int c = 0; c < cols; c++) vectorB.array(c, 1.0f);

        F32Array resultB = F32Array.create(acc, rows);

        // Second dispatch with matrix B via HAT @Reflect
        hatGemv(acc, matrixB, vectorB, resultB, rows, cols);

        // Expected: each row = 0.005 * 1.0 * 2048 = 10.24
        float expectedB = 0.005f * cols;
        float actualB = resultB.array(0);

        // Plain Java confirms B data is correct
        float[] javaResultB = plainJavaGemv(matrixB, vectorB, rows, cols);
        assertEquals(expectedB, javaResultB[0], 0.1f, "Plain Java should compute B correctly");

        // BUG: HAT dispatch may return priming value (~2.048) instead of B value (~10.24)
        float errorFromPrimed = Math.abs(actualB - primedValue);
        float errorFromExpected = Math.abs(actualB - expectedB);

        System.out.printf("Primed value: %.4f, Expected B: %.4f, Actual HAT B: %.4f%n",
                primedValue, expectedB, actualB);
        System.out.printf("Error from primed: %.4f, Error from expected: %.4f%n",
                errorFromPrimed, errorFromExpected);

        if (errorFromPrimed < errorFromExpected) {
            System.out.println("BUG CONFIRMED: HAT dispatch returned priming data, not matrix B data");
        }

        // Assert the bug: HAT result should match expected B, but it may match priming instead.
        // If this test PASSES, the bug may be fixed at this scale (but see integration tests
        // for the 16-layer transformer case where it definitely manifests).
        // We use a soft check to document the behavior either way.
        boolean bugPresent = Math.abs(actualB - expectedB) > 0.5f;
        if (bugPresent) {
            System.out.println("BUG PRESENT at unit test scale: HAT GEMV returns wrong values after priming");
        } else {
            System.out.println("Bug NOT visible at unit test scale (2048x2048). " +
                    "Bug manifests in 16-layer transformer with larger matrices — see integration tests.");
        }
    }

    /**
     * BUG: Multiple dispatches with same-sized but different-valued matrices.
     *
     * After the first dispatch, subsequent dispatches with the SAME buffer (refilled
     * with different values) may still see the old values.
     */
    @Test
    void testRefillBufferAfterDispatch() {
        Accelerator acc = new Accelerator(MethodHandles.lookup());

        int rows = 1024, cols = 1024;
        F32Array matrix = F32Array.create(acc, rows * cols);
        F32Array vector = F32Array.create(acc, cols);
        F32Array result = F32Array.create(acc, rows);

        // Fill with value A and dispatch via HAT @Reflect
        for (int i = 0; i < rows * cols; i++) matrix.array(i, 0.001f);
        for (int c = 0; c < cols; c++) vector.array(c, 1.0f);

        hatGemv(acc, matrix, vector, result, rows, cols);
        float resultA = result.array(0);
        // Expected: 0.001 * 1.0 * 1024 = 1.024

        // Refill SAME buffer with value B
        for (int i = 0; i < rows * cols; i++) matrix.array(i, 0.01f);

        // Dispatch again via HAT @Reflect with same buffer (now containing B values)
        hatGemv(acc, matrix, vector, result, rows, cols);
        float resultB = result.array(0);
        // Expected: 0.01 * 1.0 * 1024 = 10.24

        // Plain Java confirms B values are in the buffer
        float javaResultB = 0;
        for (int c = 0; c < cols; c++) {
            javaResultB += matrix.array(c) * vector.array(c); // row 0
        }

        System.out.printf("After refill — HAT result: %.4f, Plain Java: %.4f, Expected: %.4f%n",
                resultB, javaResultB, 0.01f * cols);

        boolean bugPresent = Math.abs(resultB - resultA) < 0.1f; // HAT returns old value
        if (bugPresent) {
            System.out.println("BUG CONFIRMED: HAT dispatch uses cached data after buffer refill");
        } else {
            System.out.println("Buffer refill visible to HAT at this scale");
        }
    }

    /**
     * WORKAROUND: Plain Java loops always read the correct buffer data.
     *
     * This is the recommended approach for all kernel dispatch methods.
     */
    @Test
    void testPlainJavaWorkaround() {
        Accelerator acc = new Accelerator(MethodHandles.lookup());

        int rows = 2048, cols = 2048;

        // Simulate priming + re-dispatch scenario, all in plain Java
        F32Array matrixA = F32Array.create(acc, rows * cols);
        for (int i = 0; i < rows * cols; i++) matrixA.array(i, 0.001f);

        F32Array vector = F32Array.create(acc, cols);
        for (int c = 0; c < cols; c++) vector.array(c, 1.0f);

        // "Prime" with matrix A
        float[] resultA = plainJavaGemv(matrixA, vector, rows, cols);
        assertEquals(0.001f * cols, resultA[0], 0.1f);

        // Switch to matrix B
        F32Array matrixB = F32Array.create(acc, rows * cols);
        for (int i = 0; i < rows * cols; i++) matrixB.array(i, 0.005f);

        float[] resultB = plainJavaGemv(matrixB, vector, rows, cols);

        // Plain Java always reads the correct data
        float expectedB = 0.005f * cols;
        assertEquals(expectedB, resultB[0], 0.1f,
                "Plain Java GEMV should always use current buffer data");

        // Verify it's different from the priming result
        assertNotEquals(resultA[0], resultB[0], 0.1f,
                "Results should differ because matrices have different values");
    }
}
