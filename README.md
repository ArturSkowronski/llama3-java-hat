# Llama 3 HAT Implementation ![Java](https://img.shields.io/badge/Java-26_(Babylon)-orange) ![HAT Kernels](https://img.shields.io/badge/HAT_Kernels-6%2F6_(100%25)-brightgreen) ![Model](https://img.shields.io/badge/Model-Llama_3.2_1B_Instruct_FP16-blue)

[![CI](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/ci.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/ci.yml) 
[![E2E Integration Tests](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/chat-test.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/chat-test.yml)
[![Nightly E2E](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/nightly.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/nightly.yml)
[![Weekly Full Matrix](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/weekly-full-matrix.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/weekly-full-matrix.yml)


-----------
<table style="border: none;">
<tr style="border: none;">
<td style="width: 40%; vertical-align: middle; border: none;">
  <img src="media/llama3.png" />
</td><td>
  This is a from-scratch implementation of Llama 3.2 1B Instruct inference in Java, running on <a href="https://openjdk.org/projects/babylon/">Project Babylon</a> and its Hardware Accelerator Toolkit (HAT). The whole thing - GGUF model loading, BPE tokenization, a full 16-layer transformer forward pass with GQA attention, KV cache, and greedy token generation - sits in about 2,500 lines of Java 26 with preview features enabled.
</td>
</tr>
</table>

-----------

The reference implementation is [mukel/llama3.java](https://github.com/mukel/llama3.java) and [beehive-lab/GPULlama3.java](https://github.com/beehive-lab/GPULlama3.java). This project adapts it for HAT's `@Reflect` kernel dispatch, which (if you're not keeping up with Babylon) is a way to express GPU-friendly compute kernels in plain Java and have the runtime lower them to hardware-specific backends. Think of it as "what if Java had CUDA, but it was just Java."

The interesting part: **all six compute kernels run through HAT dispatch** - GEMV, RMSNorm, RoPE, SiLU, Softmax, and Attention. That's 100% kernel coverage, roughly 8,000 HAT dispatches per 32-token inference, producing output identical to the plain Java baseline. Character for character. The model tells the same bad programming joke either way.

## What actually works

The complete inference pipeline, end to end. You give it a GGUF file, it loads tensors (F16 and F32), builds a BPE tokenizer from the GGUF vocabulary metadata, formats your prompt using the Llama 3 Instruct chat template, runs it through 16 transformer layers with grouped-query attention (32 query heads, 8 KV heads, so a 4:1 GQA ratio), and generates tokens greedily until it hits EOS or the token limit.

The architecture uses a Strategy Pattern for kernel dispatch. An `IKernelFactory` interface produces kernel implementations, and you get two factories out of the box: `PlainJavaKernelFactory` (pure loops, no HAT, always works) and `HybridKernelFactory` (lets you enable HAT selectively, per kernel type). This means you can run with any combination - all plain Java, all HAT, or any mix in between. The factory design came from the need to debug HAT kernels one at a time, but it turned out to be a pretty clean separation regardless.

**The six kernels and their HAT dispatch patterns:**

| Kernel | What It Does | HAT Pattern |
|--------|-------------|-------------|
| GEMV | Matrix-vector multiply (~113 ops/token) | Row-parallel dispatch |
| RMSNorm | Layer normalization (~33 ops/token) | Hybrid: reduction in Java, normalize in HAT |
| RoPE | Rotary positional embeddings (~32 ops/token) | Head-parallel dispatch |
| SiLU | Activation function (~16 ops/token) | Pure element-wise dispatch |
| Softmax | Score normalization (~24 ops/token) | Hybrid: reduction in Java, normalize in HAT |
| Attention | Multi-head attention (~32 ops/token) | Two sequential dispatches per head |

The "Hybrid" pattern for RMSNorm and Softmax means the reduction phase runs in plain Java and only the normalization dispatches through HAT. This is a deliberate workaround for how HAT's dispatch model handles reductions - see [Why hybrid kernels?](#why-hybrid-kernels) below for the full explanation (and if I'm wrong, please let me know - I'm learner for life 😊).

## Verification pipelines

This project tracks upstream Babylon closely, so CI is split by purpose and cadence:

| | Workflow | Schedule |
|---|---|---|
| [![CI](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/ci.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/ci.yml) | Build + unit tests + plain integration smoke | Every push / PR |
| [![E2E](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/chat-test.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/chat-test.yml) | Manual/PR chat-focused integration scenarios | PR + manual |
| [![Nightly](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/nightly.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/nightly.yml) | Daily full integration + daily benchmarks | Daily 2 AM UTC |
| [![Weekly](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/weekly-full-matrix.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/weekly-full-matrix.yml) | Weekly baseline + benchmarks + regression + backend integration | Sunday 3 AM UTC |

### Benchmark pages (GitHub Pages)

- GitHub Actions (PoCL): https://arturskowronski.github.io/llama3-java-hat/benchmark-page/github-actions-pocl/
- GCP T4: https://arturskowronski.github.io/llama3-java-hat/benchmark-page/gcp-t4/
- Latest validated GCP T4 run (inference-only): https://github.com/ArturSkowronski/llama3-java-hat/actions/runs/22215044011

### GCP machine sizing finding

On February 20, 2026, we compared `n1-standard-4 + T4` vs a higher-tier `n1-standard-16 + T4` for `benchmarkInference`.
Result: the larger CPU/RAM machine did not provide enough benefit to justify the extra cost for this workload.
Current recommendation/default remains `n1-standard-4`.

### Tag and task model

- `plain-integration`: plain Java integration tests
- `hat-integration`: HAT backend integration tests (JavaMT/OpenCL + backend dispatch coverage)
- `benchmark`: benchmark tests
- `regression`: regression-only tests (kept out of daily runs)

Main Gradle tasks:

- `test`: unit-only
- `integrationTest`: all integration (`plain-integration` + `hat-integration`)
- `plainIntegrationTest`: plain integration subset
- `hatIntegrationTest`: HAT integration subset
- `benchmarkKernelAll`: per-kernel benchmark suite
- `regressionTest`: regression-only

## Building

You need a local build of the [Babylon JDK](https://github.com/openjdk/babylon) (the `code-reflection` branch). This isn't on Maven Central, it's not in SDKMAN -- you build it yourself. Point `JAVA_BABYLON_ROOT` at your clone:

```bash
export JAVA_BABYLON_ROOT=/path/to/your/babylon
./gradlew build
```

The build system looks for HAT artifacts in `$JAVA_BABYLON_ROOT/hat/build` (or `$JAVA_BABYLON_ROOT/Contents/Home/hat/build` on macOS). If that path doesn't exist, the build will fail with a perfectly unhelpful classpath error.

## Running

You need the Llama 3.2 1B Instruct model in FP16 GGUF format (~2.5 GB). A download script is included:

```bash
./scripts/download_llama_fp16.sh
```

Then:

```bash
./gradlew run --args="$(pwd)/Llama-3.2-1B-Instruct-f16.gguf"
```

## Tests

Unit tests (no model needed):

```bash
./gradlew test
```

Integration tests (requires model):

```bash
LLAMA_FP16_PATH=$(pwd)/Llama-3.2-1B-Instruct-f16.gguf ./gradlew integrationTest
```

Run only plain integration:

```bash
LLAMA_FP16_PATH=$(pwd)/Llama-3.2-1B-Instruct-f16.gguf ./gradlew plainIntegrationTest
```

Run only HAT/backend integration:

```bash
LLAMA_FP16_PATH=$(pwd)/Llama-3.2-1B-Instruct-f16.gguf ./gradlew hatIntegrationTest
```

Benchmarks:

```bash
LLAMA_FP16_PATH=$(pwd)/Llama-3.2-1B-Instruct-f16.gguf ./gradlew benchmarkDaily
```

Regression-only:

```bash
LLAMA_FP16_PATH=$(pwd)/Llama-3.2-1B-Instruct-f16.gguf ./gradlew regressionTest
```

## Current limitations

This is intentionally specialized. It does one thing and does it correctly.

**One model only.** Architecture constants (2048 hidden size, 16 layers, 32 heads, 8 KV heads) are hardcoded for Llama 3.2 1B. There's no model auto-detection, no config parsing from GGUF metadata. If you point it at a 3B or 8B model, it won't give you a helpful error - it'll just produce garbage (or crash, if you're lucky).

**FP16 and F32 tensors only.** No quantization support whatsoever. No `Q4_0`, no K-quants, nothing. F16 gets dequantized CPU-side via `Float.float16ToFloat()`. The model file is 2.5 GB because that's what FP16 costs you.

**Java sequential backend only.** HAT supports OpenCL and PTX backends for actual GPU execution. This project currently uses the Java sequential backend, which means HAT dispatch is running the kernels on CPU, in Java, sequentially. Yes, that means it's not faster than plain Java - the point (for now) is correctness verification, not performance. The architecture is ready for GPU backends - the kernel code won't need to change, I just need to run some additional tests.

**Greedy decoding only.** No top-k, no top-p, no temperature sampling. The model deterministically picks the most probable next token every time. This is fine for testing (reproducible output) but you wouldn't want it for creative text generation 😉. Additionally, that also means it is not a good target for benchmarks.

**No streaming.** The `chat()` method returns the complete response as a string. There's no token-by-token callback, no async generation. That is... problematic, as it forced me to do some magic with Github Actions Runners that were killing my sluggish (blame the author, not the HAT technology), non-optimized token generation tests.

## What's Next

Roughly in order of what would be most interesting to tackle:

**GPU backends.** The whole point of HAT is hardware acceleration, and the kernel code is already written in a dispatch-friendly way. Plugging in the OpenCL backend should (in theory) give immediate speedups on the GEMV kernel, which dominates inference time (~113 dispatches per token out of ~250 total). The hybrid kernels (RMSNorm, Softmax) will need their reduction steps reworked for GPU parallelism.

**Quantization.** Supporting `Q4_0` and `Q8_0` would make this practical for machines with less RAM. The K-quant formats (`Q4_K`, `Q6_K`) are more complex but also more accurate. Each quantization type has its own block structure and dequantization kernel -- and those kernels themselves could be HAT-dispatched.

**Model flexibility.** Reading architecture parameters from GGUF metadata instead of hardcoding them. This is mostly plumbing - the GGUF reader already parses metadata, the constants just need to flow through.

**Sampling strategies.** Top-k and top-p sampling, temperature control. Straightforward to add on top of the existing logits output.

**Kernel fusion.** The current pipeline dispatches each kernel separately. Fusing RMSNorm with the QKV projection (which is just three GEMV calls) would reduce memory round-trips significantly - assuming a GPU backend where that matters.

## On Coding Agents usage

This project was built with AI coding agents, and since transparency matters - here's what was used.

The daily driver was [Junie CLI](https://www.jetbrains.com/junie/) (JetBrains' coding agent) - it is premiere of Junie CLI that inspired me to go through process, as I needed something non-trivial to test it on. The default model for straightforward tasks was Gemini 2.5 Flash, which honestly deserves more attention than it gets. Do not sleep on the Gemini, folks - it's genuinely good for the kind of bread-and-butter coding work that makes up 80% of any project. For the heavier lifting (debugging HAT dispatch bugs across 16 transformer layers, designing the kernel factory architecture, figuring out why buffer bounds were being cached wrong), that was Claude Opus 4.6.

To be clear: the project is mine. The architecture decisions, design patterns, the kernel restoration strategy, the test structure, the PR workflow, the "let's try one kernel at a time and see what breaks" approach - that's all human judgment. The agents wrote code under direction, not the other way around. That said, coding with agents is genuinely pleasant. It's pair programming where your partner types faster than you and never gets bored of running integration tests.

P.S. [Gemini Code Assist](https://cloud.google.com/products/gemini-code-assist) was also in the mix, reviewing my PRs for potential mistakes. A solid number of its suggestions were useful and caught real issues. A few of them broke the build, but that's what CI is for 😅

## Findings

### Why hybrid kernels?

Both RMSNorm and Softmax have two phases: a reduction (sum of squares for RMSNorm, find-max-then-sum-exp for Softmax) that reads all elements to produce a single scalar, followed by a normalization that multiplies every element by that scalar.

HAT's `@Reflect` dispatch model works by giving each kernel invocation a single index via `KernelContext` - great for embarrassingly parallel work where each element is independent, but there's no built-in mechanism for cross-lane communication or shared accumulators. You can't have 2,048 kernel invocations all contributing to the same `float sum` without atomics or a reduction tree, and the Java sequential backend provides neither.

So the reduction runs as a plain Java loop (which is fine - it's a single pass over one vector), and then the normalization step dispatches through HAT where each element just gets multiplied by the precomputed scalar. It's pragmatic, not pretty, but it works and it'll map cleanly to GPU backends when those are ready -- the reduction phase will just need a proper parallel reduction tree.

---

*Based on [mukel/llama3.java](https://github.com/mukel/llama3.java) and [beehive-lab/GPULlama3.java](https://github.com/beehive-lab/GPULlama3.java). Built with [Project Babylon](https://openjdk.org/projects/babylon/) (Java 26, `code-reflection` branch). All the good ideas are theirs, all the bad code is mine 🥶.*
