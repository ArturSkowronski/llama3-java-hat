# F16 OpenCL Codegen Workaround: Native Half-Precision Weights on GPU

**Date**: 2026-03-01
**Branch**: `feature/f16array-native-weights`
**Status**: WORKING - F16 weights running natively on GPU OpenCL

## Summary

Achieved native F16 (half-precision) weight storage running on GPU via OpenCL through HAT,
working around a Babylon codegen bug in `F16.f16ToFloat()`. The fix is a single line change
that extracts the F16 array element to a local variable before conversion.

## Results

| Backend             | Model Load | Inference | Tokens/sec | vs CPU F16 |
|---------------------|-----------|-----------|------------|------------|
| CPU Plain Java: F16 | 22.37s    | 435.90s   | 0.02 tok/s | 1x         |
| CPU Plain Java: F32 | 6.60s     | 223.62s   | 0.04 tok/s | 2x         |
| CPU HAT GEMV: F16   | 22.11s    | 453.43s   | 0.02 tok/s | 1x         |
| CPU HAT GEMV: F32   | 33.09s    | 283.64s   | 0.03 tok/s | 1.5x       |
| **GPU HAT: F16**    | **24.49s**| **28.68s**| **0.28 tok/s** | **14x** |
| GPU HAT: F32*       | 8.78s     | 19.75s    | 0.41 tok/s | 20x        |

*F32 numbers from `feature/weight-storage-mode-benchmark` branch (different run, different weight loading)

GPU F16 is ~7x faster than CPU F32, ~14x faster than CPU F16. The F16-to-F32 gap on GPU
(0.28 vs 0.41 tok/s) is the half-precision arithmetic overhead on Metal/OpenCL, expected.

## Bug: Babylon OpenCL Codegen for F16.f16ToFloat()

### The Problem

When `F16.f16ToFloat()` is called directly on an F16Array element in a `@Reflect` kernel:

```java
sum += F16.f16ToFloat(matrix.array(rowOffset + c)) * vector.array(c);
```

Babylon generates broken OpenCL C:
```c
sum=sum+(float)&matrix->array[(long)(rowOffset+c)]->value*vector->array[(long)c];
```

Two errors:
1. `->value` should be `.value` (struct member access, not pointer dereference)
2. `&` takes address of rvalue (nonsensical)

### Root Cause

In `OpenCLHATKernelBuilder.hatF16ToFloatConvOp()`:
```java
if (!hatF16ToFloatConvOp.isLocal()) {
    rarrow().identifier("value");   // ->value (for non-local / pointer)
} else if (!hatF16ToFloatConvOp.wasFloat()) {
    dot().identifier("value");      // .value  (for local variable)
}
```

When `f16ToFloat` receives `matrix.array(i)` (an array element access), the codegen
classifies it as non-local and uses `->value`. But `matrix->array[i]` already dereferences
the array, giving a struct value, not a pointer. So `.value` is correct.

### The Workaround

Extract the F16 array element to a local variable before calling `f16ToFloat`:

```java
// BEFORE (broken on OpenCL):
sum += F16.f16ToFloat(matrix.array(rowOffset + c)) * vector.array(c);

// AFTER (works on OpenCL):
F16 weight = matrix.array(rowOffset + c);
sum += F16.f16ToFloat(weight) * vector.array(c);
```

This makes `isLocal()` return true for the `weight` variable, causing codegen to emit
the correct `.value` member access:
```c
// Generated OpenCL (correct):
F16Impl_t weight = matrix->array[(long)(rowOffset+c)];
sum = sum + (float)weight.value * vector->array[(long)c];
```

### Verification

- Official Babylon `TestF16Type`: 17/17 tests PASS on OpenCL
- Official tests use `F16.add/mul/sub` pattern (codegen handles these as special built-ins)
- `F16.f16ToFloat()` IS recognized by codegen (creates `HATF16ToFloatConvOp`) but the
  local/non-local classification is wrong for array element access
- Our workaround works because local variable access IS correctly classified

## Bug 2: Cross-Class Kernel Delegation Causes OpenCL "Redefinition" Error

### The Problem

When a HAT kernel method delegates to another class:
```java
// In GEMVHAT.java
@Reflect
public static void gemvKernel(...) {
    GEMV.gemvKernel(kc, matrix, vector, result, cols);  // cross-class delegation
}
```

Babylon codegen emits `gemvKernel` twice in the OpenCL program:
- Once as `HAT_FUNC void gemvKernel(...)` (from GEMV.java, the callee)
- Once as `HAT_KERNEL void gemvKernel(...)` (from GEMVHAT.java, the entry point)

This causes: `error: redefinition of 'gemvKernel'`

### The Fix

Inline kernel bodies directly in the HAT class (matching the official `nbody` example pattern):

```java
@Reflect
public static void gemvKernel(@RO KernelContext kc, @RO F32Array matrix,
        @RO F32Array vector, @WO F32Array result, @RO int cols) {
    int row = kc.gix;
    float sum = 0.0f;
    int rowOffset = row * cols;
    for (int c = 0; c < cols; c++) {
        sum += matrix.array(rowOffset + c) * vector.array(c);
    }
    result.array(row, sum);
}
```

### Affected Kernels

All 6 HAT kernels were fixed by inlining:
- **GEMVHAT**: `gemvKernel`, `gemvKernelF16` (was delegating to `GEMV.*`)
- **RMSNormHAT**: `normalizeKernel` (was delegating to `RMSNorm.normalizeKernel`)
- **RoPEHAT**: `ropeKernel` (was delegating to `RoPE.ropeKernel`)
- **SiLUHAT**: `siluKernel` (was delegating to `SiLU.siluKernel`)
- **SoftmaxHAT**: already inlined (no fix needed)
- **AttentionHAT**: already inlined (no fix needed)

## Codegen Pipeline Reference

```
F16 method call in @Reflect kernel
    |
    v
HATFP16Phase.dialectifyF16Ops / dialectifyF16ToFloat
    (pattern matches method name: "add", "mul", "f16ToFloat", etc.)
    |
    v
Creates specialized HATF16Op:
    - HATF16BinaryOp (add/mul/sub/div)
    - HATF16ToFloatConvOp (f16ToFloat)
    - HATF16ConvOp (floatToF16)
    |
    v
HATOpDispatcher routes to backend handler
    |
    v
OpenCLHATKernelBuilder.hatF16BinaryOp / hatF16ToFloatConvOp
    |
    v
Emits OpenCL C code
```

Key files in Babylon:
- `hat/phases/HATFP16Phase.java` - recognition & dialect transformation
- `hat/codebuilders/C99HATKernelBuilder.java` - base C99 codegen for F16 ops
- `hat/backends/ffi/opencl/.../OpenCLHATKernelBuilder.java` - OpenCL-specific lowering
- `hat/dialect/HATF16Op.java` - HAT dialect ops for F16

## Key Takeaways

1. **F16 on GPU works** with proper workaround - no need to convert to F32
2. **Local variable extraction** bypasses the codegen bug (isLocal classification)
3. **Kernel body inlining** is required for OpenCL (no cross-class delegation)
4. **F16.add/mul/sub work differently** from f16ToFloat - they're lowered as complete
   binary ops, not as value extraction + cast
5. **14x speedup** (CPU F16 → GPU F16) with native half-precision weights
