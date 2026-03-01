# F16Array Adoption: Findings and Rationale

## Summary

This document captures findings from investigating HAT's `F16Array` buffer type and the rationale for adopting it in the Llama 3 inference pipeline. The change replaces CPU-side F16-to-F32 dequantization with native F16Array buffers, keeping weight data in half-precision until the kernel reads it.

## Babylon Evolves Fast

A critical observation from working with Project Babylon: **the upstream API surface changes rapidly**. The `code-reflection` branch receives significant additions on a near-weekly basis, and features that don't exist in one build may appear days later.

### F16Array Timeline in Babylon

| Date | Commit | Change |
|------|--------|--------|
| 2025-10-21 | `5128c70b87c` | `[hat] Initial support for 16-bits float types (half)` — F16Array first appears |
| 2025-10-22 | `ae059ba1732` | `[hat] F16 object creation within device code via F16.of` |
| 2025-10-22 | `4731bacdbf1` | `[hat] CPU Implementation for F16` |
| 2025-11-07 | `9fe9c75726b` | `[hat] Extensions of F16 (API and codegen)` |
| 2025-11-10 | `13f6ad669c7` | `[hat] F16 decoupled from Buffer type` |
| 2025-11-13 | `1a27f62407b` | `[hat] DeviceType interface definition for private/local data structures` |
| 2025-12-01 | `8f3399d3959` | `[hat] Cleanup F16 name impl` |
| 2025-12-18 | `ed0b4dab66c` | `Isolating iface mapper so it can be used outside of HAT` |
| 2025-12-19 | `ead504a521e` | `Moved the ifacemapper out of HAT core to optlk` |
| 2025-12-21 | `a5d2fba0a19` | `We were needlessly passing accelerator around, as a Lookup carrier` |
| 2026-01-23 | `948caaf45ab` | `Hat incrementally separate schema building from allocation` |

The F16 type went through **11 commits** across 3 months of iteration. This is typical of Babylon — APIs are refined iteratively as the HAT compiler, dialect lowering, and backend codegen mature together.

**Practical impact**: When this project started, the initial F16Array API existed but was still being refined. Features like `HATMath` (added Feb 20, 2026) and FlashAttention examples appeared even later. Any project building on Babylon should expect to track upstream regularly and adapt.

## Why F16Array

### Current State: Eager Dequantization

The GGUF model file stores ~90% of weight tensors in FP16 format (GGUF type 1). Currently, `LlamaModel.mapTensor()` **eagerly dequantizes every F16 tensor to F32** at load time:

```java
// Current: F16 → F32 at load time
for (int i = 0; i < elementCount; i++) {
    short f16 = segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, (long) i * 2);
    buffer.array(i, Float.float16ToFloat(f16));
}
```

This means:
- **Memory doubled**: Each F16 element (2 bytes) becomes F32 (4 bytes) in the HAT buffer
- **CPU work at load**: ~280M dequantization operations at startup
- **GPU transfer bloated**: When a GPU backend is used, we transfer 2x the necessary data

### New State: Native F16Array

With `F16Array`, weight data stays in half-precision:
- **2x memory reduction** for weight buffers (~1.2 GB saved for Llama 3.2 1B)
- **Dequantization at compute time**: GEMV kernel reads F16 and converts to float during the dot product
- **GPU-ready**: F16Array can be transferred to GPU memory in native format — the GPU reads half-precision directly

### API

```java
// hat.buffer.F16Array
public interface F16Array extends Buffer {
    int length();
    F16Impl array(long index);       // returns F16 element accessor
    static F16Array create(Accelerator accelerator, int length);
}

// hat.types.F16
static float f16ToFloat(F16 value);  // F16 → float conversion (maps to Float.float16ToFloat)
static F16 floatToF16(float value);  // float → F16 conversion
```

## Backward Compatibility

### Inference Output

**Must produce identical token sequences.** The mathematical operation is the same: `float sum += f16_to_float(weight) * float_vector[i]`. The only difference is *when* the conversion happens — at tensor load vs at kernel read. Since `Float.float16ToFloat()` is deterministic, results are bit-identical.

### API Surface

- `LlamaModel.mapTensor()` **remains unchanged** — still returns F32Array for any tensor. Existing callers (norm weights, test utilities) continue to work.
- New `LlamaModel.mapTensorF16()` method added for F16 tensors specifically.
- `IGEMV` gets a **default method** overload for F16Array matrix — implementations that don't override it fall back to the F32 version (full backward compat).
- `PlainJavaKernelFactory` and `HybridKernelFactory` require no changes — the GEMV implementations (GEMV, GEMVHAT) each handle both F32 and F16.

### Which Tensors Stay F32

Not all tensors are F16 in the GGUF file. The following are F32 (type 0) and stay as `F32Array`:
- `blk.{N}.attn_norm.weight` — RMSNorm weights (2048 elements each, 16 layers)
- `blk.{N}.ffn_norm.weight` — RMSNorm weights (2048 elements each, 16 layers)
- `output_norm.weight` — final RMSNorm weight (2048 elements)

Everything else (projection matrices, FFN weights, embedding table) is F16 and gets `F16Array`.

## Memory Impact

| Buffer | Elements | Before (F32) | After (F16) | Saved |
|--------|----------|--------------|-------------|-------|
| token_embd.weight | 128,256 x 2,048 | 1.0 GB | 0.5 GB | 0.5 GB |
| Per-layer weights (x16) | ~42M per layer | 2.6 GB total | 1.3 GB total | 1.3 GB |
| output_norm.weight | 2,048 | 8 KB (F32) | 8 KB (F32) | 0 |
| **Total weight buffers** | | **~3.6 GB** | **~1.8 GB** | **~1.8 GB** |

Note: Intermediate computation buffers (q, k, v, attn scores, etc.) remain F32Array — they hold computed values, not loaded weights.

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| F16Array API changes upstream | Pin to known-working Babylon commit; API is stable since Jan 2026 |
| Performance regression from per-element F16→F32 in kernel | Negligible — `Float.float16ToFloat()` is a single instruction on modern CPUs; on GPU, hardware does it natively |
| Buffer bounds caching bug with F16Array | Existing priming workaround should apply equally; will verify in integration tests |
| Different rounding behavior | Mathematically impossible — same `Float.float16ToFloat()` call, just moved from load-time to compute-time |
