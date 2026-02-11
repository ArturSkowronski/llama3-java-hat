package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.buffer.F32Array;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Minimal reproduction of HAT Java Sequential Backend buffer caching bug.
 *
 * Bug: When a @Reflect-annotated kernel is dispatched multiple times through
 * the same Accelerator with F32Array buffers of different sizes, the backend
 * caches the buffer bounds from the FIRST dispatch. Subsequent dispatches
 * with LARGER buffers fail with IndexOutOfBoundsException.
 *
 * Each test uses a fresh Accelerator to avoid cross-test contamination,
 * since the caching is at the Accelerator level.
 *
 * See HAT_BUG_REPORT.md for full details.
 */
public class GEMVBufferCachingBugTest {

    /**
     * Baseline: A single GEMV call with a small matrix works fine.
     */
    @Test
    void testSmallMatrixAlone() {
        Accelerator acc = new Accelerator(MethodHandles.lookup());
        GEMV gemv = new GEMV(acc);

        int rows = 4, cols = 3;
        F32Array matrix = F32Array.create(acc, rows * cols);
        F32Array vector = F32Array.create(acc, cols);
        F32Array result = F32Array.create(acc, rows);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                matrix.array(r * cols + c, (r == c) ? 1.0f : 0.0f);
            }
        }
        for (int c = 0; c < cols; c++) {
            vector.array(c, (c + 1) * 10.0f); // [10, 20, 30]
        }

        gemv.apply(matrix, vector, result, rows, cols);

        assertEquals(10.0f, result.array(0), 1e-5f);
        assertEquals(20.0f, result.array(1), 1e-5f);
        assertEquals(30.0f, result.array(2), 1e-5f);
    }

    /**
     * Baseline: A single GEMV call with a large matrix works fine.
     */
    @Test
    void testLargeMatrixAlone() {
        Accelerator acc = new Accelerator(MethodHandles.lookup());
        GEMV gemv = new GEMV(acc);

        int rows = 8192, cols = 2048;
        F32Array matrix = F32Array.create(acc, rows * cols);
        F32Array vector = F32Array.create(acc, cols);
        F32Array result = F32Array.create(acc, rows);

        for (int i = 0; i < rows * cols; i++) {
            matrix.array(i, 0.001f);
        }
        for (int c = 0; c < cols; c++) {
            vector.array(c, 1.0f);
        }

        gemv.apply(matrix, vector, result, rows, cols);

        // Each row dot product: 0.001 * 1.0 * 2048 = 2.048
        assertEquals(2.048f, result.array(0), 0.01f);
    }

    /**
     * BUG: Call small matrix first, then large matrix → fails.
     *
     * The backend caches buffer bounds from the first dispatch (small matrix = 4M elements).
     * The second dispatch uses a larger matrix (16M elements) but the backend still
     * enforces the old 4M bound, causing the access at index 4194304 to fail.
     *
     * Expected: Both calls succeed (buffers are correctly sized).
     * Actual: Second call throws RuntimeException wrapping IndexOutOfBoundsException.
     */
    @Test
    void testSmallThenLarge_FAILS_due_to_buffer_caching() {
        Accelerator acc = new Accelerator(MethodHandles.lookup());
        GEMV gemv = new GEMV(acc);

        // Step 1: Small matrix (2048 x 2048 = 4,194,304 elements)
        int smallRows = 2048, smallCols = 2048;
        F32Array smallMatrix = F32Array.create(acc, smallRows * smallCols);
        F32Array smallVector = F32Array.create(acc, smallCols);
        F32Array smallResult = F32Array.create(acc, smallRows);

        for (int i = 0; i < smallRows * smallCols; i++) {
            smallMatrix.array(i, 0.001f);
        }
        for (int c = 0; c < smallCols; c++) {
            smallVector.array(c, 1.0f);
        }

        // First dispatch — succeeds, caches buffer bounds (4M)
        gemv.apply(smallMatrix, smallVector, smallResult, smallRows, smallCols);

        // Step 2: Large matrix (8192 x 2048 = 16,777,216 elements)
        int largeRows = 8192, largeCols = 2048;
        F32Array largeMatrix = F32Array.create(acc, largeRows * largeCols);
        F32Array largeVector = F32Array.create(acc, largeCols);
        F32Array largeResult = F32Array.create(acc, largeRows);

        for (int i = 0; i < largeRows * largeCols; i++) {
            largeMatrix.array(i, 0.001f);
        }
        for (int c = 0; c < largeCols; c++) {
            largeVector.array(c, 1.0f);
        }

        // Second dispatch — FAILS: HAT wraps IndexOutOfBoundsException in RuntimeException
        RuntimeException ex = assertThrows(
                RuntimeException.class,
                () -> gemv.apply(largeMatrix, largeVector, largeResult, largeRows, largeCols),
                "BUG: Backend caches F32Array bounds from first dispatch. " +
                "Second call with larger buffers should succeed but fails."
        );

        // Verify the root cause is IndexOutOfBoundsException with the cached bound
        Throwable root = getRootCause(ex);
        assertInstanceOf(IndexOutOfBoundsException.class, root,
                "Root cause should be IndexOutOfBoundsException, got: " + root.getClass().getName());
        assertTrue(root.getMessage().contains("4194304"),
                "Error should reference the cached buffer size (4,194,304), got: " + root.getMessage());
    }

    /**
     * BUG: Separate GEMV instances sharing the same Accelerator still fail.
     *
     * This proves the caching happens at the Accelerator/backend level,
     * not at the kernel class level.
     */
    @Test
    void testSeparateInstances_STILL_FAILS() {
        Accelerator acc = new Accelerator(MethodHandles.lookup());
        GEMV gemv1 = new GEMV(acc);
        GEMV gemv2 = new GEMV(acc);

        // Use gemv1 with small matrix
        int smallRows = 2048, smallCols = 2048;
        F32Array smallMatrix = F32Array.create(acc, smallRows * smallCols);
        F32Array smallVector = F32Array.create(acc, smallCols);
        F32Array smallResult = F32Array.create(acc, smallRows);
        for (int i = 0; i < smallRows * smallCols; i++) smallMatrix.array(i, 0.001f);
        for (int c = 0; c < smallCols; c++) smallVector.array(c, 1.0f);

        gemv1.apply(smallMatrix, smallVector, smallResult, smallRows, smallCols);

        // Use gemv2 with large matrix — still fails
        int largeRows = 8192, largeCols = 2048;
        F32Array largeMatrix = F32Array.create(acc, largeRows * largeCols);
        F32Array largeVector = F32Array.create(acc, largeCols);
        F32Array largeResult = F32Array.create(acc, largeRows);
        for (int i = 0; i < largeRows * largeCols; i++) largeMatrix.array(i, 0.001f);
        for (int c = 0; c < largeCols; c++) largeVector.array(c, 1.0f);

        assertThrows(
                RuntimeException.class,
                () -> gemv2.apply(largeMatrix, largeVector, largeResult, largeRows, largeCols),
                "BUG: Separate GEMV instances sharing the same Accelerator still fail. " +
                "Caching occurs at the Accelerator/backend level."
        );
    }

    /**
     * WORKAROUND: Call the largest matrix FIRST to prime the cache.
     *
     * If the first dispatch uses the largest buffers, all subsequent calls
     * with smaller buffers succeed because the cached bounds are large enough.
     */
    @Test
    void testWorkaround_largestFirst() {
        Accelerator acc = new Accelerator(MethodHandles.lookup());
        GEMV gemv = new GEMV(acc);

        // Step 1: Large matrix FIRST (8192 x 2048 = 16,777,216 elements)
        int largeRows = 8192, largeCols = 2048;
        F32Array largeMatrix = F32Array.create(acc, largeRows * largeCols);
        F32Array largeVector = F32Array.create(acc, largeCols);
        F32Array largeResult = F32Array.create(acc, largeRows);

        for (int i = 0; i < largeRows * largeCols; i++) {
            largeMatrix.array(i, 0.001f);
        }
        for (int c = 0; c < largeCols; c++) {
            largeVector.array(c, 1.0f);
        }

        // Prime the cache with the largest buffer
        gemv.apply(largeMatrix, largeVector, largeResult, largeRows, largeCols);
        assertEquals(2.048f, largeResult.array(0), 0.01f);

        // Step 2: Small matrix SECOND (2048 x 2048) — succeeds
        int smallRows = 2048, smallCols = 2048;
        F32Array smallMatrix = F32Array.create(acc, smallRows * smallCols);
        F32Array smallVector = F32Array.create(acc, smallCols);
        F32Array smallResult = F32Array.create(acc, smallRows);

        for (int i = 0; i < smallRows * smallCols; i++) {
            smallMatrix.array(i, 0.001f);
        }
        for (int c = 0; c < smallCols; c++) {
            smallVector.array(c, 1.0f);
        }

        // This works because cached bounds (16M) are >= required bounds (4M)
        // Note: the kernel executes without crashing, but results may be written
        // to the cached (larger) result buffer rather than the new smaller one.
        // In practice, the workaround works when using the SAME buffers across calls
        // (as in TransformerBlock), not when creating fresh smaller buffers.
        assertDoesNotThrow(
                () -> gemv.apply(smallMatrix, smallVector, smallResult, smallRows, smallCols),
                "Workaround: priming with the largest matrix first should prevent crashes"
        );
    }

    private static Throwable getRootCause(Throwable t) {
        Throwable cause = t;
        while (cause.getCause() != null && cause.getCause() != cause) {
            cause = cause.getCause();
        }
        return cause;
    }
}
