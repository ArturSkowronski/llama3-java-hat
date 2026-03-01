# F16 vs F32 Weight Storage Benchmark Findings

**Date**: 2026-03-01
**Branch**: `feature/weight-storage-mode-benchmark`
**Model**: Llama 3.2 1B Instruct FP16 (`Llama-3.2-1B-Instruct-f16.gguf`)
**Machine**: macOS (Darwin 25.2.0), Java Sequential backend
**Prompt**: "Say hi" | Max tokens: 8

## Results

| Mode | Model Load | Inference | Tokens/sec | Relative Speed |
|------|-----------|-----------|------------|----------------|
| F16 (native) | 29.24s | 367.19s | 0.022 tok/s | 1.00x (baseline) |
| F32 (eager dequant) | 25.20s | 220.48s | 0.036 tok/s | **1.67x faster** |

## Key Observations

### 1. F32 inference is ~1.67x faster than F16

F32 mode completes inference in **220s** vs **367s** for F16 — a **40% reduction** in wall-clock time. This is expected: the F16 path calls `F16.f16ToFloat()` on every weight element during every GEMV dot product, while F32 mode pays the conversion cost once at load time and then operates on native floats throughout.

### 2. F32 loads ~14% faster

F32 load time is **25.2s** vs **29.2s** for F16. This is slightly counterintuitive since F32 mode eagerly dequantizes all weights (more work upfront). The difference likely comes from `mapTensor()` using `Float.float16ToFloat()` (JDK intrinsic) vs `mapTensorF16()` which copies into F16Array element-by-element via `buffer.array(i).value(short)`.

### 3. Both modes produce identical output

Both runs generated the same response: "Hi! How can I assist you today...". The `WeightStorageModeComparisonTest` integration test is designed to assert bit-identical logits.

## Analysis

The F16 native path trades **compute** (per-element f16-to-f32 conversion at every GEMV) for **memory** (~1.8 GB savings for Llama 3.2 1B weights). On this CPU-only Java Sequential backend:

- **F16 advantage**: ~1.8 GB less memory for weight buffers
- **F32 advantage**: 1.67x faster inference, 14% faster model load

The f16-to-f32 conversion overhead is substantial because GEMV is the dominant operation (~8 GEMV calls per transformer block, 16 layers, per token) and each call iterates over every weight element. With F32, these are direct float reads; with F16, each read goes through `F16.f16ToFloat()`.

## When to use which

| Scenario | Recommended Mode |
|----------|-----------------|
| Memory-constrained (e.g., <4GB heap) | F16 |
| Performance-critical / benchmarking | F32 |
| GPU/OpenCL backend (future) | F16 (GPU has native F16 ALUs) |
| Correctness comparison / regression testing | Either (identical output) |

## Raw TSV

```
timestamp	backend	load_sec	infer_sec	tok_per_sec	error
2026-03-01T12:25:28.036398Z	Weight Storage: F16 (native)	29.243395583	367.193990167	0.021786848952406863
2026-03-01T12:25:28.042754Z	Weight Storage: F32 (eager dequant)	25.201518625	220.476995625	0.03628496468451004
```
