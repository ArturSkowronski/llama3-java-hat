## Bug: `@Reflect` kernel dispatch produces incorrect results when reading F32Array data written by plain Java

### Environment

- **Java**: Project Babylon (babylon-jdk branch)
- **HAT**: Java Sequential Backend (`hat.backend.java.JavaSequentialBackend`)
- **Module**: `jdk.incubator.code`

### Summary

When an `F32Array` buffer is populated via plain Java (e.g., `buffer.array(i, value)`) and then read inside a `@Reflect`-dispatched kernel, the kernel reads **stale or incorrect values**. The same computation implemented in plain Java (iterating the buffer directly) produces correct results. This is a data visibility issue between the Java heap and the HAT dispatch pathway.

### Impact

This bug makes `@Reflect` kernel dispatch unusable for any computation that reads from externally-populated `F32Array` buffers (e.g., model weights loaded from GGUF files). All kernels must fall back to plain Java loops.

### Minimal Reproduction

See `ReflectDispatchDataVisibilityBugTest.java` for a self-contained test suite.

The test dispatches the existing `GEMV.dispatchGEMV` kernel via `@Reflect`:

```java
// Dispatch GEMV via HAT @Reflect
private static void hatGemv(Accelerator acc, F32Array matrix, F32Array vector,
                            F32Array result, int rows, int cols) {
    acc.compute((Accelerator.@Reflect Compute) cc ->
            GEMV.dispatchGEMV(cc, matrix, vector, result, rows, cols));
}
```

**Steps to reproduce:**

```java
Accelerator acc = new Accelerator(MethodHandles.lookup());
int rows = 2048, cols = 2048;

// Matrix A: all 0.001 — dispatch via HAT @Reflect (works correctly)
F32Array matrixA = F32Array.create(acc, rows * cols);
for (int i = 0; i < rows * cols; i++) matrixA.array(i, 0.001f);
F32Array vectorA = F32Array.create(acc, cols);
for (int c = 0; c < cols; c++) vectorA.array(c, 1.0f);
F32Array resultA = F32Array.create(acc, rows);
hatGemv(acc, matrixA, vectorA, resultA, rows, cols);
// resultA[0] = 2.048 ✓ (correct)

// Matrix B: all 0.005 — dispatch via HAT @Reflect (reads wrong values)
F32Array matrixB = F32Array.create(acc, rows * cols);
for (int i = 0; i < rows * cols; i++) matrixB.array(i, 0.005f);
F32Array vectorB = F32Array.create(acc, cols);
for (int c = 0; c < cols; c++) vectorB.array(c, 1.0f);
F32Array resultB = F32Array.create(acc, rows);
hatGemv(acc, matrixB, vectorB, resultB, rows, cols);
// resultB[0] = 0.0 ✗ (expected 10.24)
```

**Expected behavior:** `resultB[0]` = 10.24 (0.005 × 1.0 × 2048).

**Actual behavior:** `resultB[0]` = 0.0. The second `@Reflect` dispatch ignores the new buffer data entirely. The first dispatch returns correct results; all subsequent dispatches with different buffers return wrong values.

### Key Observations

1. **Plain Java reads are always correct.** Iterating `matrix.array(i)` in a regular Java loop always returns the value written by `matrix.array(i, value)`. The data is physically present in the buffer.

2. **`@Reflect` dispatch reads wrong values.** The same `matrix.array(i)` call inside a `@Reflect` kernel returns different values, suggesting the dispatch pathway uses a different view or cached copy of the buffer data.

3. **The bug reproduces at unit test scale.** With 2048×2048 matrices, the first `@Reflect` dispatch returns correct results (2.048), but the second dispatch with different `F32Array` buffers returns 0.0 instead of the expected 10.24. Small buffers (~12 elements) work correctly in isolation.

4. **Multiple dispatches compound the error.** When multiple `@Reflect` kernels are chained (e.g., RMSNorm → GEMV → RoPE → GEMV), each kernel's output is wrong, and the errors compound. In a 16-layer Llama transformer, this produces logit values ~10²⁶ instead of ~7.

5. **All in-place `@Reflect` kernels are affected.** RMSNorm, RoPE, and SiLU kernels that modify `F32Array` in-place via dispatch also exhibit the issue — their writes are not visible to subsequent plain Java reads or to subsequent `@Reflect` dispatches.

### Real-World Impact

Discovered while implementing Llama 3.2 1B inference with HAT. The forward pass dispatches ~12 GEMV calls per layer across 16 layers, reading from weight matrices loaded from a GGUF model file.

| Implementation | BOS → Top logit | Top token | Entropy | Coherent output? |
|---------------|----------------|-----------|---------|-----------------|
| HAT `@Reflect` GEMV | ~10²⁶ | garbage | 0.000 | No — repeats same token |
| Plain Java GEMV | ~7 | reasonable | 0.774 | Yes — generates jokes |

After switching ALL kernels (GEMV, RMSNorm, RoPE, SiLU) from `@Reflect` dispatch to plain Java loops, the model produces correct, coherent text output.

### Test Suite

`ReflectDispatchDataVisibilityBugTest.java` contains 5 tests:

| Test | Purpose | Result |
|------|---------|--------|
| `testPlainJavaGemvIsCorrect` | Baseline — plain Java GEMV produces correct results | PASS |
| `testSingleHatDispatchIsCorrect` | Baseline — single `@Reflect` dispatch works in isolation | PASS |
| `testPrimedDispatchUsesCachedData` | **BUG** — second dispatch with different buffers returns 0.0 instead of 10.24 | BUG CONFIRMED |
| `testRefillBufferAfterDispatch` | Refilling same buffer and re-dispatching | Diagnostic |
| `testPlainJavaWorkaround` | **WORKAROUND** — identical logic in plain Java loops works correctly | PASS |

### Workaround

Replace all `@Reflect` kernel dispatch calls with equivalent plain Java loops:

```java
// Before (broken):
public void apply(F32Array matrix, F32Array vector, F32Array result, int rows, int cols) {
    accelerator.compute((Accelerator.@Reflect Compute) cc ->
            dispatchGEMV(cc, matrix, vector, result, rows, cols));
}

// After (working):
public void apply(F32Array matrix, F32Array vector, F32Array result, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        int rowOffset = row * cols;
        for (int c = 0; c < cols; c++) {
            sum += matrix.array(rowOffset + c) * vector.array(c);
        }
        result.array(row, sum);
    }
}
```

This preserves the `@Reflect` kernel code for future use but bypasses the dispatch pathway.

### Suspected Root Cause

The Java Sequential Backend likely maintains an internal copy or view of `F32Array` buffer data that is not synchronized with the backing `MemorySegment`. When `buffer.array(i, value)` writes to the segment via plain Java, the backend's cached copy is not updated. When the `@Reflect` kernel executes, it reads from the stale cached copy rather than the live segment data.

### Relationship to Bug #1 (Buffer Bounds Caching)

This is a **separate bug** from the buffer bounds caching issue reported in `HAT_BUG_REPORT.md` / PR #13. That bug causes `IndexOutOfBoundsException` when buffer sizes change between dispatches. This bug causes **silently wrong results** even when buffer sizes are consistent — a more dangerous failure mode because it produces no errors or exceptions.
