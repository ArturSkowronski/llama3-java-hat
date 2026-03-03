# F16 vs F32 Weight Storage Benchmark Findings

**Model**: Llama 3.2 1B Instruct FP16 (`Llama-3.2-1B-Instruct-f16.gguf`)
**Machine**: Apple M1 Pro (16 GPU cores), macOS Darwin 25.2.0
**Prompt**: "Say hi" | Max tokens: 8

> **Caveat**: 8 tokens is a very small sample — each run is ~4-7 minutes of wall clock
> time, so background OS activity significantly affects F16 numbers. F32 is stable
> (±1%) across all runs; F16 shows ±20% variance. The gap is real, the magnitude is approximate.

---

## Run 1 — Baseline (no GEMV optimization)

**Date**: 2026-03-01, branch `feature/weight-storage-mode-benchmark`

| Backend | Mode | Inference | Relative |
|---------|------|-----------|----------|
| CPU Java Seq | F16 (native) | 371s | 1.00x |
| CPU Java Seq | F32 (eager dequant) | 222s | **1.67x faster** |
| GPU OpenCL | F16 (native) | 350s | 1.06x faster |
| GPU OpenCL | F32 (eager dequant) | 221s | **1.68x faster** |

## Run 2 — After GEMV row-buffer optimization

**Date**: 2026-03-01, branch `feature/f16array-native-weights` @ `38c427a`

| Backend | Mode | Inference | Relative |
|---------|------|-----------|----------|
| CPU Java Seq | F16 (native) | 445s | 1.00x |
| CPU Java Seq | F32 (eager dequant) | 224s | **1.98x faster** |

**Observation**: The row-buffer optimization did not improve F16 inference time — it appears
slightly worse (445s vs 371s), though this is likely measurement noise given the high F16
variance. The optimization may not be effective because the dot-product pass (Pass 2) still
reads through `F32Array.array(c)` interface dispatch on the vector, preventing SIMD
vectorization of the inner loop. Only the weight reads land in plain `float[]`; the vector
reads remain interface-bound.

---

## Root Cause Analysis

The hot path is `GEMV.apply(F16Array, ...)` executing ~128× per token (8 calls × 16 layers).

**Three JVM-hostile operations interleaved per element:**

1. `matrix.array(long)` — returns `F16Impl`, an optkl iface-mapper struct proxy over `MemorySegment`
2. `F16.f16ToFloat(F16)` — virtual dispatch to get the `short`, then `Float.float16ToFloat()`
3. `vector.array(c)` — same interface dispatch on F32Array

The JIT cannot vectorize the dot product because neither operand is a plain `float[]`. The
F32 path has the same interface overhead for element access but avoids the f16→f32 conversion,
making it consistently ~1.67–2× faster.

**The row-buffer optimization (`38c427a`) isolates the weight conversion but not the vector
reads.** For full vectorization, both operands need to be `float[]`. That would require
also copying the input vector into a scratch buffer, doubling the memory pressure, which
may not be worth it.

## What would actually help

| Approach | Expected gain | Complexity |
|----------|--------------|------------|
| Copy vector into `float[]` scratch too (both operands plain) | ~1.5x | Low |
| Use `MemorySegment.copy` + `VectorAPI` for batch convert | ~2x | Medium |
| HAT `@Reflect` GEMV kernel on GPU (F16 native ALUs) | ~5-10x | High |

## When to use which

| Scenario | Recommended Mode |
|----------|-----------------|
| Memory-constrained (e.g., <4GB heap) | F16 |
| Performance-critical on CPU today | F32 |
| GPU with native F16 GEMV kernel (future) | F16 (zero-copy, native F16 ALUs) |

## Raw TSV

```
# Run 1 (baseline)
2026-03-01T12:46:37Z	CPU Java Seq: F16 (native)	29.84	371.24	0.0215
2026-03-01T12:46:37Z	CPU Java Seq: F32 (eager dequant)	27.98	222.37	0.0360
2026-03-01T12:46:37Z	GPU OpenCL: F16 (native)	30.01	349.98	0.0229
2026-03-01T12:46:37Z	GPU OpenCL: F32 (eager dequant)	25.25	220.90	0.0362

# Run 2 (after GEMV row-buffer optimization)
2026-03-01T~14:00Z	CPU Java Seq: F16 (native)	27.82	445.25	0.0180
2026-03-01T~14:00Z	CPU Java Seq: F32 (eager dequant)	24.69	224.46	0.0356
```
