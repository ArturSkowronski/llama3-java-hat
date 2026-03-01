# F16 GEMV Performance: Row-Buffer Optimization

## Problem

Benchmarks showed F16 native weight storage is **1.67x slower** than F32 eager dequantization on CPU (371s vs 222s for 8-token inference). The bottleneck is `GEMV.apply(F16Array, ...)`, which executes ~128 times per token (8 GEMV calls × 16 layers).

## Root Cause

The original F16 GEMV inner loop interleaved three JVM-hostile operations per element:

```java
sum += F16.f16ToFloat(matrix.array(rowOffset + c)) * vector.array(c);
```

1. **`matrix.array(long)`** — returns `F16Impl`, an optkl iface-mapper struct proxy over a `MemorySegment`. Each call goes through `BoundSchema` dispatch and bounds checking.
2. **`F16.f16ToFloat(F16)`** — virtual dispatch to `value.value()` to extract the `short`, then `Float.float16ToFloat()`. The JDK intrinsic itself is fast, but buried under interface dispatch.
3. **No auto-vectorization** — the JIT cannot vectorize the dot product because element access goes through interface calls, not contiguous memory reads.

The F32 path has the same interface overhead for element access but avoids the f16→f32 conversion entirely, making it ~1.67x faster.

## Fix

Split the inner loop into two passes — conversion and dot product:

```java
// Pass 1: dequantize row into contiguous float[]
for (int c = 0; c < cols; c++)
    rowBuf[c] = F16.f16ToFloat(matrix.array(rowOffset + c));

// Pass 2: dot product on plain floats (JIT-vectorizable)
for (int c = 0; c < cols; c++)
    sum += rowBuf[c] * vector.array(c);
```

- **Pass 1** (conversion): sequential read through F16Array — still interface-heavy but no longer interleaved with FMA
- **Pass 2** (dot product): plain `float[]` multiply-accumulate — JIT can auto-vectorize this loop

The scratch buffer is `float[8192]` (max cols = INTERMEDIATE_SIZE), 32KB, fits in L1 cache, allocated once per GEMV instance and reused across rows and calls.

## Expected Impact

- The dot product (Pass 2) becomes a tight loop over `float[]` — eligible for SIMD vectorization by C2
- The conversion (Pass 1) remains interface-bound, but decoupling it from the FMA lets C2 optimize each loop independently
- Memory savings from F16 storage (~1.8 GB) are preserved
