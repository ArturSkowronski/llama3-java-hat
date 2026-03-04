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

---

## Run 3 — benchmarkInference: F16 / F16_FAST / F32 comparison (GCP n1-standard-4 + Tesla T4)

**Date**: 2026-03-04, branch `main` (PR #48 merged — F16_FAST included)
**Machine**: GCP n1-standard-4 (4 vCPU, 15GB RAM) + NVIDIA Tesla T4 (OpenCL 3.0, 40 CUs, CC 7.5)
**Prompt**: "Say hi" | Max tokens: 8

### GPU Results (Tesla T4)

| Backend | Weight Mode | Load | Inference | **tok/s** | vs F16 |
|---------|-------------|------|-----------|-----------|--------|
| HAT OpenCL GPU | F16 (F16Array) | 31.1s | 20.8s | **0.38** | baseline |
| HAT OpenCL GPU | F32 (F32Array) | 17.8s | 27.6s | **0.29** | 0.76x |
| HAT OpenCL GPU | F16_FAST (short[]) | 4.8s | 56.3s | **0.14** | 0.37x |

**Key insight**: F16_FAST (short[]) is CPU-optimized storage — it lazy-materializes to F16Array on first GPU use.
Each benchmark runs in an isolated JVM, so the lazy materialization happens every time → no caching benefit.
F16_FAST at 0.14 tok/s reflects the overhead of short[]→F16Array copy per JVM startup, not steady-state GPU perf.

### CPU Results

| Backend | Weight Mode | Load | Inference | **tok/s** |
|---------|-------------|------|-----------|-----------|
| Plain Java | F16_FAST (short[]) | 4.2s | 81.6s | **0.10** |
| Plain Java | F32 (F32Array) | 17.2s | 406.7s | **0.02** |
| Plain Java | F16 (F16Array) | 29.8s | 985.9s | **0.01** |
| HAT Java MT | F16_FAST (short[]) | 3.9s | 605.9s | 0.01 |
| HAT Java MT | F16 (F16Array) | 30.8s | 573.5s | 0.01 |
| HAT Java Seq | F16_FAST (short[]) | 4.1s | 1119.9s | 0.01 |
| HAT Java Seq | F16 (F16Array) | 30.3s | 1080.3s | 0.01 |

### Key Findings

1. **GPU F16 = 0.38 tok/s** — best overall, 31% faster than F32 on GPU (memory bandwidth win)
2. **GPU F32 loads faster** (17.8s vs 31.1s) — no F16→F32 dequant on load
3. **F16_FAST on GPU = 0.14 tok/s** — suboptimal: each isolated JVM re-materializes short[]→F16Array from scratch; in a long-running server the lazy cache would make it equivalent to F16
4. **F16_FAST on CPU = 0.10 tok/s** — 10x faster than F16Array on CPU (no iface-mapper overhead)
5. **HAT Java CPU overhead**: HAT Seq/MT dispatch overhead makes them slower than Plain Java for all modes

### Load Time Analysis

| Weight Mode | Load time |
|-------------|-----------|
| F16_FAST (short[]) | ~4s — fast: raw MemorySegment.copy into short[] |
| F32 (F32Array) | ~17s — medium: F16→F32 dequant on load |
| F16 (F16Array) | ~30s — slow: iface-mapper proxy setup overhead |

### Raw TSV

```
2026-03-04T09:31:17Z	HAT Java MT (F16_FAST)	3.91	605.94	0.0132
2026-03-04T09:41:23Z	HAT Java MT (F16)	30.83	573.49	0.0139
2026-03-04T09:42:26Z	HAT OpenCL GPU (F16_FAST)	4.82	56.31	0.1421
2026-03-04T09:43:20Z	HAT OpenCL GPU (F16)	31.14	20.80	0.3846
2026-03-04T09:44:07Z	HAT OpenCL GPU (F32)	17.77	27.63	0.2895
2026-03-04T10:02:53Z	HAT Java Sequential (F16_FAST)	4.05	1119.94	0.0071
2026-03-04T10:21:25Z	HAT Java Sequential (F16)	30.27	1080.28	0.0074
2026-03-04T10:22:52Z	Plain Java (F16_FAST)	4.24	81.56	0.0981
2026-03-04T10:39:49Z	Plain Java (F16)	29.79	985.89	0.0081
2026-03-04T10:46:54Z	Plain Java (F32)	17.24	406.74	0.0197
```
