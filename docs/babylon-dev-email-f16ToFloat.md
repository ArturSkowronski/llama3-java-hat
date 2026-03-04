**To:** babylon-dev@openjdk.org
**Subject:** [HAT] Bug: F16.f16ToFloat() generates broken OpenCL C when argument is an F16Array element

---

Hi Babylon team,

I've been experimenting with running Llama 3.2 1B inference entirely through HAT kernels — using `@Reflect`-annotated GEMV, RMSNorm, RoPE, SiLU, Softmax and Attention kernels dispatched to OpenCL on a GPU. As part of that work I migrated weight tensors from eager F32 dequantization to native `F16Array` buffers, which is where I ran into this bug.

I'd like to report a codegen bug in the OpenCL HAT backend affecting `F16.f16ToFloat()`.

**Summary**

`F16.f16ToFloat(array.array(i))` generates invalid OpenCL C when the argument is an F16Array element access. The generated code has two syntax errors that prevent compilation.

**Reproducer**

```java
@Reflect
public static void kernel(
        @RO KernelContext kc,
        @RO F16Array matrix, @RO F32Array vector, @WO F32Array result,
        @RO int cols) {
    int row = kc.gix;
    float sum = 0.0f;
    int rowOffset = row * cols;
    for (int c = 0; c < cols; c++) {
        sum += F16.f16ToFloat(matrix.array(rowOffset + c)) * vector.array(c);
    }
    result.array(row, sum);
}
```

**Generated OpenCL (broken)**

```c
sum=sum+(float)&matrix->array[(long)(rowOffset+c)]->value*vector->array[(long)c];
```

Two errors:

1. `->value` should be `.value` — `matrix->array[i]` already dereferences the array to a struct value, not a pointer
2. `&` before the expression — takes the address of an rvalue (`half`), which is illegal in C99

**Expected OpenCL**

```c
sum=sum+(float)matrix->array[(long)(rowOffset+c)].value*vector->array[(long)c];
```

**Root Cause**

`OpenCLHATKernelBuilder.hatF16ToFloatConvOp()` (line ~266):

```java
if (!hatF16ToFloatConvOp.isLocal()) {
    rarrow().identifier("value");   // ->value  ← used for array element access
} else if (!hatF16ToFloatConvOp.wasFloat()) {
    dot().identifier("value");      // .value   ← used for local variable
}
```

When `f16ToFloat` receives `matrix.array(i)` (array element access), `isLocal()` returns `false`, so codegen chooses the `->value` path. But `matrix->array[i]` already dereferences the array — the result is a struct (`F16Impl_t`), not a pointer. It should use `.value`.

The same pattern exists in `CudaHATKernelBuilder.java` (line ~295).

**Workaround**

Extracting to a local variable first makes `isLocal()` return `true`:

```java
// Works:
F16 weight = matrix.array(rowOffset + c);
sum += F16.f16ToFloat(weight) * vector.array(c);
```

**Additional notes**

- `F16.add/mul/sub` are **not affected** — they go through `hatF16BinaryOp` which handles struct access correctly
- All 17 official `TestF16Type` tests pass because they use `F16.add/mul/sub`, not `f16ToFloat` directly
- Tested on `code-reflection` branch built 2026-03-01, macOS aarch64 (Metal/OpenCL) and confirmed via CI with PoCL on Linux x86_64

**CI reproducer**

A minimal test is available in a public repository with a GitHub Actions workflow that reproduces the failure automatically:

https://github.com/ArturSkowronski/llama3-java-hat/actions/runs/22570303149

- `testF16ToFloatDirectOnArrayElement` → **FAILS** with the two codegen errors above
- `testF16ToFloatViaLocalVariable` → **SKIPPED** on PoCL (no `cl_khr_fp16`), **PASSES** on GPU with FP16 support

Relevant files:
- Test: https://github.com/ArturSkowronski/llama3-java-hat/blob/main/src/test/java/com/arturskowronski/llama3babylon/hat/kernels/F16ToFloatOpenCLCodegenTest.java
- Workflow: https://github.com/ArturSkowronski/llama3-java-hat/blob/main/.github/workflows/babylon-bug-reproducer.yml

**Running locally**

```bash
git clone https://github.com/ArturSkowronski/llama3-java-hat
cd llama3-java-hat
JAVA_BABYLON_ROOT=<path-to-babylon-repo> ./gradlew babylonBugReproducer --info
```

Where `<path-to-babylon-repo>` is a local clone of `openjdk/babylon` (`code-reflection` branch) with HAT already built (`java -cp hat/job.jar --enable-preview --source 26 hat.java bld` from `babylon/hat/`).

Happy to provide more context or test a patch. Thanks for the great work on Babylon!

Best,
Artur Skowronski
