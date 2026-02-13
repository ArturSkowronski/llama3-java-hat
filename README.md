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

The reference implementation is [mukel/llama3.java](https://github.com/mukel/llama3.java) and [beehive-lab/GPULlama3.java](https://github.com/beehive-lab/GPULlama3.java) . This project adapts it for HAT's `@Reflect` kernel dispatch, which (if you're not keeping up with Babylon) is a way to express GPU-friendly compute kernels in plain Java and have the runtime lower them to hardware-specific backends. Think of it as "what if Java had CUDA, but it was just Java."

The interesting part: **all six compute kernels run through HAT dispatch** - GEMV, RMSNorm, RoPE, SiLU, Softmax, and Attention. That's 100% kernel coverage, roughly 8,000 HAT dispatches per 32-token inference, producing output identical to the plain Java baseline. Character for character. The model tells the same bad programming joke either way.

## What actually works

The complete inference pipeline, end to end. You give it a GGUF file, it loads tensors (F16 and F32), builds a BPE tokenizer from the GGUF vocabulary metadata, formats your prompt using the Llama 3 Instruct chat template, runs it through 16 transformer layers with grouped-query attention (32 query heads, 8 KV heads, so a 4:1 GQA ratio), and generates tokens greedily until it hits EOS or the token limit.

The architecture uses a Strategy Pattern for kernel dispatch. An `IKernelFactory` interface produces kernel implementations, and you get two factories out of the box: `PlainJavaKernelFactory` (pure loops, no HAT, always works) and `HybridKernelFactory` (lets you enable HAT selectively, per kernel type). This means you can run with any combination -- all plain Java, all HAT, or any mix in between. The factory design came from the need to debug HAT kernels one at a time, but it turned out to be a pretty clean separation regardless.

**The six kernels and their HAT dispatch patterns:**

| Kernel | What It Does | HAT Pattern |
|--------|-------------|-------------|
| GEMV | Matrix-vector multiply (~113 ops/token) | Row-parallel dispatch |
| RMSNorm | Layer normalization (~33 ops/token) | Hybrid: reduction in Java, normalize in HAT |
| RoPE | Rotary positional embeddings (~32 ops/token) | Head-parallel dispatch |
| SiLU | Activation function (~16 ops/token) | Pure element-wise dispatch |
| Softmax | Score normalization (~24 ops/token) | Hybrid: reduction in Java, normalize in HAT |
| Attention | Multi-head attention (~32 ops/token) | Two sequential dispatches per head |

The hybrid pattern for RMSNorm and Softmax deserves a note. Both operations have two phases: a reduction (sum of squares for RMSNorm, find-max-then-sum-exp for Softmax) that reads all elements to produce a single scalar, followed by a normalization that multiplies every element by that scalar. HAT's `@Reflect` dispatch model works by giving each kernel invocation a single index via `KernelContext` -- great for embarrassingly parallel work where each element is independent, but there's no built-in mechanism for cross-lane communication or shared accumulators. You can't have 2,048 kernel invocations all contributing to the same `float sum` without atomics or a reduction tree, and the Java sequential backend provides neither. So the reduction runs as a plain Java loop (which is fine -- it's a single pass over one vector), and then the normalization step dispatches through HAT where each element just gets multiplied by the precomputed scalar. It's pragmatic, not pretty, but it works and it'll map cleanly to GPU backends when those are ready -- the reduction phase will just need a proper parallel reduction tree.

## Verification pipelines

This project depends on an upstream JDK that's under active development (the Babylon `code-reflection` branch), a HAT runtime that's pre-release, and kernel dispatch semantics that could change without warning. That combination means "it worked on my machine last Tuesday" isn't enough (I learned it the other way, nearly submiting bug report for issue that was already resolved at upstream). The CI strategy is built around one question: if something breaks, can we tell whether it's our code, HAT, or upstream Babylon?

Every workflow starts by building the Babylon JDK and HAT from source against the latest `code-reflection` HEAD. This is intentional - I want to catch upstream regressions (and benefits!) early, in async way.

| | Workflow | Schedule |
|---|---|---|
| [![CI](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/ci.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/ci.yml) | Build + Unit Tests + TinyLlama | Every push |
| [![E2E](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/chat-test.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/chat-test.yml) | E2E Integration Tests (FP16 + HAT) | Every PR + Manual |
| [![Nightly](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/nightly.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/nightly.yml) | Nightly E2E (Baseline + All-HAT) | Daily 2 AM UTC |
| [![Weekly](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/weekly-full-matrix.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/weekly-full-matrix.yml) | Weekly Full Matrix (6 individual HAT kernels) | Sunday 3 AM UTC |

**CI (every push)** - The fast feedback loop. Builds the Babylon JDK, compiles the project, runs unit tests, then downloads TinyLlama (a small quantized model) and runs integration tests against it. This validates that the GGUF reader, tokenizer, and basic inference pipeline haven't regressed. It doesn't touch the real Llama 3.2 FP16 model or any HAT dispatch -- the point is to catch compilation failures and logic bugs. Skips runs on docs-only changes because rebuilding an entire JDK to validate a typo in the README felt excessive 😉.

**E2E Integration Tests (every PR + manual)** - Runs the plain Java baseline test (`ChatIntegrationTest`) against the real Llama 3.2 1B Instruct FP16 model (2.5 GB, cached across runs) on every PR to main. This is the pre-merge gate - if the baseline inference is broken, you'll know before merging. Skips docs-only changes. On manual dispatch, you get tiered scopes: `baseline-only`, `all-hat` (adds the all-six-kernels-simultaneously test), and `full-matrix` (runs each of the six HAT kernels in isolation). The HAT tiers are manual-only because a full matrix run takes ~30 minutes of compute and you don't always need that level of granularity for a PR - most of the time, confirming baseline correctness is the merge gate, and HAT validation can wait for nightly.

**Nightly E2E (daily, 2 AM UTC)** - The upstream drift detector. Runs two jobs in parallel: unit tests with TinyLlama (same as CI, catching basic regressions) and the real FP16 model tests (plain Java baseline followed by all-HAT with 6/6 kernels). The rationale is simple: Babylon's `code-reflection` branch moves fast, and HAT's runtime semantics are still being stabilized. A nightly run against the latest upstream HEAD means we find out within 24 hours if a Babylon commit broke something, rather than discovering it during a weekend coding session. If the baseline passes but all-HAT fails, that points to a HAT runtime change. If both fail, it's likely an upstream JDK or compilation issue.

**Weekly Full Matrix (Sunday, 3 AM UTC)** - The exhaustive verification. This is the only workflow that tests each HAT kernel individually in isolation, running all six as separate matrix jobs (`ChatIntegrationTestWith{SiLU,RoPE,Softmax,RMSNorm,Attention,GEMV}HAT`) with `fail-fast: false` so every kernel gets its result regardless of whether others fail. After all individual tests and the baseline complete, it runs the all-HAT combined test as a final gate. The reason for running individual kernels weekly (and not daily) is that it's expensive - 8 separate E2E inference runs against a 2.5 GB model - but it answers a question the other pipelines can't: if the all-HAT test fails, *which specific kernel* broke? During initial development, this isolation was critical for debugging (the kernels were restored one by one precisely because combined failures were impossible to diagnose). Now it serves as a regression safety net.

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

Unit tests run without any model files:

```bash
./gradlew test
```

Integration tests need models. There are two tiers:

**TinyLlama tests of GGUF format** (lightweight, fast, good for CI):
```bash
./scripts/download_tinyllama.sh
TINY_LLAMA_PATH=$(pwd)/tinyllama-1.1b-chat-v1.0.Q2_K.gguf ./gradlew integrationTest
```

**Full Llama 3.2 FP16 tests** (the real thing, ~6 minutes each):
```bash
LLAMA_FP16_PATH=$(pwd)/Llama-3.2-1B-Instruct-f16.gguf ./gradlew integrationTest
```

The E2E tests are split per kernel. There's a `ChatIntegrationTestWith{Kernel}HAT` for each of the six kernels individually, plus `ChatIntegrationTestWithAllHAT` that enables all six simultaneously. Each test runs the same prompt ("Tell a joke about programming") and validates the output isn't gibberish using heuristics for repeated characters, non-ASCII ratio, and character diversity. The expected output (if you're curious): "Why did the programmer quit his job? Because he didn't get arrays."

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

**Model flexibility.** Reading architecture parameters from GGUF metadata instead of hardcoding them. This is mostly plumbing -- the GGUF reader already parses metadata, the constants just need to flow through.

**Sampling strategies.** Top-k and top-p sampling, temperature control. Straightforward to add on top of the existing logits output.

**Kernel fusion.** The current pipeline dispatches each kernel separately. Fusing RMSNorm with the QKV projection (which is just three GEMV calls) would reduce memory round-trips significantly - assuming a GPU backend where that matters.

## On Coding Agents usage

This project was built with AI coding agents, and since transparency matters - here's what was used.

The daily driver was [Junie CLI](https://www.jetbrains.com/junie/) (JetBrains' coding agent) - it is premiere of Junie CLI that inspired me to go through process, as I needed something non-trivial to test it on. The default model for straightforward tasks was Gemini 2.5 Flash, which honestly deserves more attention than it gets. Do not sleep on the Gemini, folks - it's genuinely good for the kind of bread-and-butter coding work that makes up 80% of any project. For the heavier lifting (debugging HAT dispatch bugs across 16 transformer layers, designing the kernel factory architecture, figuring out why buffer bounds were being cached wrong), that was Claude Opus 4.6.

To be clear: the project is mine. The architecture decisions, design patterns, the kernel restoration strategy, the test structure, the PR workflow, the "let's try one kernel at a time and see what breaks" approach - that's all human judgment. The agents wrote code under direction, not the other way around. That said, coding with agents is genuinely pleasant. It's pair programming where your partner types faster than you and never gets bored of running integration tests.

P.S. [Gemini Code Assist](https://cloud.google.com/products/gemini-code-assist) was also in the mix, reviewing my PRs for potential mistakes. A solid number of its suggestions were useful and caught real issues. A few of them broke the build, but that's what CI is for 😅

---

*Based on [mukel/llama3.java](https://github.com/mukel/llama3.java). Built with [Project Babylon](https://openjdk.org/projects/babylon/) (Java 26, code-reflection branch). All the good ideas are theirs, all the bad code is mine.*
