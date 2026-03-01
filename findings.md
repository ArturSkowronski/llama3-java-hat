# F16 vs F32 Weight Storage Benchmark Findings

**Date**: 2026-03-01
**Branch**: `feature/weight-storage-mode-benchmark`
**Model**: Llama 3.2 1B Instruct FP16 (`Llama-3.2-1B-Instruct-f16.gguf`)
**Machine**: Apple M1 Pro (16 GPU cores), macOS Darwin 25.2.0
**Prompt**: "Say hi" | Max tokens: 8

## Results

| Backend | Mode | Model Load | Inference | Tokens/sec | Relative Speed |
|---------|------|-----------|-----------|------------|----------------|
| CPU Java Seq | F16 (native) | 29.84s | 371.24s | 0.022 tok/s | 1.00x (baseline) |
| CPU Java Seq | F32 (eager dequant) | 27.98s | 222.37s | 0.036 tok/s | **1.67x faster** |
| GPU OpenCL | F16 (native) | 30.01s | 349.98s | 0.023 tok/s | 1.06x faster |
| GPU OpenCL | F32 (eager dequant) | 25.25s | 220.90s | 0.036 tok/s | **1.68x faster** |

## Key Observations

### 1. F32 is ~1.67x faster than F16 on both CPU and GPU

The F32 advantage is consistent across backends: **222s vs 371s** on CPU, **221s vs 350s** on GPU. The F16 path calls `F16.f16ToFloat()` on every weight element during every GEMV dot product, while F32 pays the conversion cost once at load time.

### 2. GPU OpenCL provides no meaningful speedup

GPU OpenCL is essentially the same speed as CPU Java Sequential for both modes:
- F16: GPU 350s vs CPU 371s (only 6% faster)
- F32: GPU 221s vs CPU 222s (within noise)

This is expected — the HAT OpenCL backend dispatches kernels via `@Reflect` but the current `PlainJavaKernelFactory` doesn't use HAT-dispatched GEMV. The OpenCL backend is only exercising the infrastructure, not offloading the actual GEMV compute to GPU. A `HybridKernelFactory` with GPU-native GEMV would be needed to see real GPU acceleration.

### 3. F32 loads ~14% faster consistently

F32 load: **26-28s** vs F16 load: **30s** across both backends. The `mapTensor()` path uses `Float.float16ToFloat()` (JDK intrinsic) while `mapTensorF16()` copies element-by-element via `buffer.array(i).value(short)`.

### 4. Both modes produce identical output

All 4 runs generated the same response: "Hi! How can I assist you today...".

## Analysis

The F16 native path trades **compute** (per-element f16-to-f32 conversion at every GEMV) for **memory** (~1.8 GB savings for Llama 3.2 1B weights).

- **F16 advantage**: ~1.8 GB less memory for weight buffers
- **F32 advantage**: 1.67x faster inference, 14% faster model load

The f16-to-f32 conversion overhead is substantial because GEMV is the dominant operation (~8 GEMV calls per transformer block, 16 layers, per token) and each call iterates over every weight element. With F32, these are direct float reads; with F16, each goes through `F16.f16ToFloat()`.

The GPU showing no speedup confirms that the bottleneck is in the GEMV kernel implementation (plain Java loops), not the backend infrastructure. True GPU acceleration requires a HAT `@Reflect` GEMV kernel that offloads the matrix-vector multiply to OpenCL compute shaders — at which point F16 native storage becomes advantageous since GPUs have native F16 ALUs and the data is already in the right format.

## When to use which

| Scenario | Recommended Mode |
|----------|-----------------|
| Memory-constrained (e.g., <4GB heap) | F16 |
| Performance-critical / benchmarking (CPU) | F32 |
| GPU with native F16 GEMV kernel (future) | F16 (zero-copy to GPU, native F16 ALUs) |
| Correctness comparison / regression testing | Either (identical output) |

## Raw TSV

```
timestamp	backend	load_sec	infer_sec	tok_per_sec	error
2026-03-01T12:46:37.778952Z	CPU Java Seq: F16 (native)	29.840974625	371.237498917	0.021537083756498706
2026-03-01T12:46:37.783498Z	CPU Java Seq: F32 (eager dequant)	27.976752667	222.367655125	0.035979261098975624
2026-03-01T12:46:37.783647Z	GPU OpenCL: F16 (native)	30.010665583	349.976805917	0.022858654898218233
2026-03-01T12:46:37.783717Z	GPU OpenCL: F32 (eager dequant)	25.24756775	220.904247958	0.03621564027370754
```
