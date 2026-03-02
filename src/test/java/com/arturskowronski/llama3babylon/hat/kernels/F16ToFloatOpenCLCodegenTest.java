package com.arturskowronski.llama3babylon.hat.kernels;

import com.arturskowronski.llama3babylon.hat.BackendType;
import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.types.F16;
import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;

import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.WO;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Reproducer for Babylon HAT bug: F16.f16ToFloat() generates incorrect OpenCL C
 * when called directly on an F16Array element (not a local variable).
 *
 * Bug report: docs/babylon-bug-report-f16ToFloat.md
 * File for upstream: babylon-dev@openjdk.org
 * Root cause: OpenCLHATKernelBuilder.hatF16ToFloatConvOp() line 266 —
 *   isLocal() returns false for array element access, causing codegen to emit
 *   ->value (pointer dereference) instead of .value (struct member access).
 *
 * Contains two tests with identical arithmetic, same OpenCL backend:
 *   testF16ToFloatDirectOnArrayElement  — FAILS (buggy codegen path)
 *   testF16ToFloatViaLocalVariable      — PASSES (workaround: extract to local first)
 */
@Tag("babylon-bug")
public class F16ToFloatOpenCLCodegenTest {

    // ─── Kernel A: BUGGY pattern ─────────────────────────────────────────────
    // F16.f16ToFloat() called directly on F16Array element.
    // Generated OpenCL: (float)&matrix->array[i]->value  ← BROKEN
    //   error 1: & takes address of rvalue (illegal)
    //   error 2: -> dereferences a pointer, but matrix->array[i] is a struct value

    @Reflect
    public static void dispatchF16DirectCall(
            @RO ComputeContext cc,
            @RO F16Array matrix, @RO F32Array vector, @WO F32Array result,
            @RO int rows, @RO int cols) {
        cc.dispatchKernel(NDRange.of1D(rows),
                kc -> gemvKernelF16Direct(kc, matrix, vector, result, cols));
    }

    @Reflect
    public static void gemvKernelF16Direct(
            @RO KernelContext kc,
            @RO F16Array matrix, @RO F32Array vector, @WO F32Array result,
            @RO int cols) {
        int row = kc.gix;
        float sum = 0.0f;
        int rowOffset = row * cols;
        for (int c = 0; c < cols; c++) {
            // BUG: f16ToFloat() on array element → (float)&matrix->array[i]->value
            sum += F16.f16ToFloat(matrix.array(rowOffset + c)) * vector.array(c);
        }
        result.array(row, sum);
    }

    // ─── Kernel B: WORKING workaround ────────────────────────────────────────
    // Extract element to local variable first, then call f16ToFloat().
    // isLocal() returns true for a local variable → codegen emits .value (correct).
    // Generated OpenCL: (float)matrix->array[i].value  ← CORRECT

    @Reflect
    public static void dispatchF16ViaLocal(
            @RO ComputeContext cc,
            @RO F16Array matrix, @RO F32Array vector, @WO F32Array result,
            @RO int rows, @RO int cols) {
        cc.dispatchKernel(NDRange.of1D(rows),
                kc -> gemvKernelF16ViaLocal(kc, matrix, vector, result, cols));
    }

    @Reflect
    public static void gemvKernelF16ViaLocal(
            @RO KernelContext kc,
            @RO F16Array matrix, @RO F32Array vector, @WO F32Array result,
            @RO int cols) {
        int row = kc.gix;
        float sum = 0.0f;
        int rowOffset = row * cols;
        for (int c = 0; c < cols; c++) {
            // WORKAROUND: extract to local variable first.
            // isLocal() == true → codegen emits .value → valid OpenCL C.
            F16 weight = matrix.array(rowOffset + c);
            sum += F16.f16ToFloat(weight) * vector.array(c);
        }
        result.array(row, sum);
    }

    // ─── Shared setup ────────────────────────────────────────────────────────

    private static void fillBuffers(F16Array matrix, F32Array vector, int rows, int cols) {
        // Matrix row 0: [1, 2, 3, 4], row 1: [5, 6, 7, 8]
        float[] matrixVals = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        for (int i = 0; i < rows * cols; i++) {
            matrix.array(i).value(Float.floatToFloat16(matrixVals[i]));
        }
        // Vector: [1, 1, 1, 1]
        for (int c = 0; c < cols; c++) {
            vector.array(c, 1.0f);
        }
    }

    // ─── Tests ───────────────────────────────────────────────────────────────

    /**
     * FAILS on OpenCL backend: codegen emits (float)&matrix->array[i]->value
     * which is invalid C99 — two errors (spurious &, wrong -> instead of .).
     * Will pass once OpenCLHATKernelBuilder.hatF16ToFloatConvOp() is fixed.
     */
    @Test
    void testF16ToFloatDirectOnArrayElement() {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), BackendType.OPENCL.predicate());

        int rows = 2, cols = 4;
        F16Array matrix = F16Array.create(accelerator, rows * cols);
        F32Array vector = F32Array.create(accelerator, cols);
        F32Array result = F32Array.create(accelerator, rows);
        fillBuffers(matrix, vector, rows, cols);

        // Bug present  → OpenCL compile error → exception → test FAILS
        // Bug fixed    → kernel compiles and produces correct output
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
                dispatchF16DirectCall(cc, matrix, vector, result, rows, cols));

        assertEquals(10.0f, result.array(0), 0.01f, "Row 0 dot product");  // 1+2+3+4
        assertEquals(26.0f, result.array(1), 0.01f, "Row 1 dot product");  // 5+6+7+8
    }

    /**
     * PASSES on OpenCL backend: extracting to a local variable makes isLocal() return true,
     * so codegen correctly emits (float)matrix->array[i].value.
     * Same arithmetic as the failing test — only difference is the local variable.
     *
     * Skipped when cl_khr_fp16 is unavailable (e.g. PoCL on CI) because even correct
     * F16Array codegen requires the FP16 extension at kernel execution time.
     * Set OPENCL_HAS_F16=true to run (or use a GPU with native FP16 support).
     */
    @Test
    void testF16ToFloatViaLocalVariable() {
        assumeTrue("true".equals(System.getenv("OPENCL_HAS_F16")),
                "Skipping: cl_khr_fp16 not available on this OpenCL platform (set OPENCL_HAS_F16=true to run)");

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), BackendType.OPENCL.predicate());

        int rows = 2, cols = 4;
        F16Array matrix = F16Array.create(accelerator, rows * cols);
        F32Array vector = F32Array.create(accelerator, cols);
        F32Array result = F32Array.create(accelerator, rows);
        fillBuffers(matrix, vector, rows, cols);

        accelerator.compute((Accelerator.@Reflect Compute) cc ->
                dispatchF16ViaLocal(cc, matrix, vector, result, rows, cols));

        assertEquals(10.0f, result.array(0), 0.01f, "Row 0 dot product");  // 1+2+3+4
        assertEquals(26.0f, result.array(1), 0.01f, "Row 1 dot product");  // 5+6+7+8
    }
}
