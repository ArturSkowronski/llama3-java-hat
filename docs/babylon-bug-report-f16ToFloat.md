# Bug Report: F16.f16ToFloat() generates incorrect OpenCL C when called on F16Array element

## Summary

`F16.f16ToFloat(array.array(i))` generates broken OpenCL C code when the argument is an F16Array element access (not a local variable). The generated code has two syntax errors that prevent compilation.

## Reproducer

Kernel code:
```java
@Reflect
public static void kernel(@RO KernelContext kc, @RO F16Array matrix, @RO F32Array vector, @WO F32Array result, @RO int cols) {
    int row = kc.gix;
    float sum = 0.0f;
    int rowOffset = row * cols;
    for (int c = 0; c < cols; c++) {
        sum += F16.f16ToFloat(matrix.array(rowOffset + c)) * vector.array(c);
    }
    result.array(row, sum);
}
```

## Generated OpenCL (broken)

```c
sum=sum+(float)&matrix->array[(long)(rowOffset+c)]->value*vector->array[(long)c];
```

## Expected OpenCL

```c
sum=sum+(float)matrix->array[(long)(rowOffset+c)].value*vector->array[(long)c];
```

## Errors

1. `->value` should be `.value` — `matrix->array[i]` dereferences to a struct value, not a pointer
2. `&` before the expression — takes address of an rvalue (`half`), which is illegal

## Root Cause

`OpenCLHATKernelBuilder.hatF16ToFloatConvOp()` (line 266):

```java
if (!hatF16ToFloatConvOp.isLocal()) {
    rarrow().identifier("value");      // ->value (pointer path)
} else if (!hatF16ToFloatConvOp.wasFloat()) {
    dot().identifier("value");         // .value  (local path)
}
```

When `f16ToFloat` receives `matrix.array(i)` (array element access), `isLocal()` returns `false`, so the codegen uses `->value`. But `matrix->array[i]` already dereferences the array — the result is a struct, not a pointer. It should use `.value`.

## Workaround

Extract array element to a local variable before calling `f16ToFloat`:

```java
// Works:
F16 weight = matrix.array(rowOffset + c);
sum += F16.f16ToFloat(weight) * vector.array(c);
```

With a local variable, `isLocal()` returns `true` → codegen emits `.value` → correct OpenCL.

## Notes

- `F16.add/mul/sub` are NOT affected — they go through `hatF16BinaryOp` which handles struct access correctly
- All 17 official `TestF16Type` tests pass because they use the `F16.add/mul/sub` pattern
- Tested on Babylon `code-reflection` branch (built 2026-03-01), macOS aarch64, Metal/OpenCL
- File: `hat/backends/ffi/opencl/src/main/java/hat/backend/ffi/OpenCLHATKernelBuilder.java:266`
- Same pattern exists in CUDA backend: `CudaHATKernelBuilder.java:295`
