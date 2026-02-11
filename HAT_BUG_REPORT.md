## Bug: Java Sequential Backend caches F32Array bounds from first kernel dispatch

### Environment

- **Java**: Project Babylon (babylon-jdk branch)
- **HAT**: Java Sequential Backend (`hat.backend.java.JavaSequentialBackend`)
- **Module**: `jdk.incubator.code`

### Summary

When a `@Reflect`-annotated GEMV kernel is dispatched multiple times through the same `Accelerator` with `F32Array` buffers of different sizes, the Java Sequential Backend appears to cache the buffer bounds from the **first** dispatch. Subsequent dispatches that use **larger** `F32Array` buffers fail with `IndexOutOfBoundsException`, even though the new buffers are correctly sized.

### Reproduction

The following GEMV kernel is dispatched through an `Accelerator`:

```java
public class GEMV {
    private final Accelerator accelerator;

    public GEMV(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    public void apply(F32Array matrix, F32Array vector, F32Array result, int rows, int cols) {
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
                dispatchGEMV(cc, matrix, vector, result, rows, cols)
        );
    }

    @Reflect
    public static void gemvKernel(KernelContext kc, F32Array matrix, F32Array vector, F32Array result, int cols) {
        int row = kc.gix;
        float sum = 0.0f;
        int rowOffset = row * cols;
        for (int c = 0; c < cols; c++) {
            sum += matrix.array(rowOffset + c) * vector.array(c);  // <-- fails here
        }
        result.array(row, sum);
    }

    @Reflect
    public static void dispatchGEMV(ComputeContext cc, F32Array matrix, F32Array vector, F32Array result, int rows, int cols) {
        cc.dispatchKernel(NDRange.of1D(rows), kc -> gemvKernel(kc, matrix, vector, result, cols));
    }
}
```

**Steps to reproduce:**

```java
Accelerator acc = new Accelerator(MethodHandles.lookup());
GEMV gemv = new GEMV(acc);

// Step 1: First dispatch with a 2048x2048 matrix (4,194,304 elements) — SUCCEEDS
F32Array smallMatrix = F32Array.create(acc, 2048 * 2048);
F32Array vec = F32Array.create(acc, 2048);
F32Array result1 = F32Array.create(acc, 2048);
gemv.apply(smallMatrix, vec, result1, 2048, 2048);  // OK

// Step 2: Second dispatch with an 8192x2048 matrix (16,777,216 elements) — FAILS
F32Array largeMatrix = F32Array.create(acc, 8192 * 2048);
F32Array result2 = F32Array.create(acc, 8192);
gemv.apply(largeMatrix, vec, result2, 8192, 2048);  // IndexOutOfBoundsException!
```

**Expected behavior:** Both calls succeed, as both matrices are correctly sized for their respective row/col parameters.

**Actual behavior:** The second call throws:

```
java.lang.IndexOutOfBoundsException: Index 4194304 out of bounds for length 4194304
    at com.example.GEMV.gemvKernel(GEMV.java:XX)
```

The error message says the array has length `4,194,304` (= 2048 x 2048), which is the size of the **first** matrix, not the second. The kernel appears to be operating on the cached buffer from the first dispatch rather than the new `largeMatrix`.

### Key Observations

1. **Calling the larger matrix FIRST works**: If the 8192x2048 dispatch happens before the 2048x2048 dispatch, both calls succeed. This is consistent with the backend caching the first buffer's bounds and rejecting larger buffers in later calls.

2. **Fresh `F32Array` buffers do not help**: Creating entirely new buffers for the second call does not avoid the issue — the cached bounds persist.

3. **Separate GEMV instances do not help**: Creating a second `GEMV` object with the same `Accelerator` still fails, suggesting the caching occurs at the `Accelerator` or backend level, not at the kernel class level.

4. **The issue is specific to the Java Sequential Backend**: Individual kernel tests pass when only one matrix size is used.

### Context

This was discovered while implementing a Llama 3.2 transformer block using HAT. The forward pass calls GEMV for:
- QKV projections: `[2048, 2048]` and `[512, 2048]` matrices
- Output projection: `[2048, 2048]` matrix
- Feed-forward network: `[8192, 2048]` and `[2048, 8192]` matrices

The QKV/output projections dispatch first and succeed. The FFN projections dispatch later with larger matrices and fail.

### Workaround

Prime the GEMV kernel with the **largest** matrix before any other dispatch:

```java
// In constructor, after allocating buffers:
gemv.apply(w1, xb, hb, INTERMEDIATE_SIZE, HIDDEN_SIZE);  // largest matrix first
```

This ensures the backend caches the largest buffer bounds, and all subsequent calls with smaller matrices succeed.

### Suspected Root Cause

The `JavaSequentialBackend.dispatchKernel()` (or `JavaBackend.dispatchCompute()`) likely caches the `@Reflect`-analyzed method parameters — including `F32Array` buffer references or their backing array sizes — from the first invocation. Subsequent invocations reuse the cached analysis rather than re-analyzing with the new parameters, causing bounds checks to use stale array lengths.
