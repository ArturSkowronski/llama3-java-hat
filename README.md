# Llama 3 HAT Implementation ![Java](https://img.shields.io/badge/Java-26_(Babylon)-orange) ![HAT Kernels](https://img.shields.io/badge/HAT_Kernels-6%2F6_(100%25)-brightgreen) ![Model](https://img.shields.io/badge/Model-Llama_3.2_1B_Instruct_FP16-blue)

| | Workflow | Schedule |
|---|---|---|
| [![CI](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/ci.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/ci.yml) | Build + Unit Tests | Every push |
| [![E2E Integration Tests](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/chat-test.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/chat-test.yml) | E2E Integration Tests (FP16 + HAT) | Manual |
| [![Nightly E2E](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/nightly.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/nightly.yml) | Nightly E2E (Baseline + All-HAT) | Daily 2 AM UTC |
| [![Weekly Full Matrix](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/weekly-full-matrix.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/weekly-full-matrix.yml) | Weekly Full Matrix (6 individual HAT kernels) | Sunday 3 AM UTC |



---

## What Is This

This is a from-scratch implementation of Llama 3.2 1B Instruct inference in Java, running on [Project Babylon](https://openjdk.org/projects/babylon/) and its Hardware Accelerator Toolkit (HAT). The whole thing -- GGUF model loading, BPE tokenization, a full 16-layer transformer forward pass with GQA attention, KV cache, and greedy token generation -- sits in about 2,500 lines of Java 26 with preview features enabled.

The reference implementation is [mukel/llama3.java](https://github.com/mukel/llama3.java) and [beehive-lab/GPULlama3.java](https://github.com/beehive-lab/GPULlama3.java) . This project adapts it for HAT's `@Reflect` kernel dispatch, which (if you're not keeping up with Babylon) is a way to express GPU-friendly compute kernels in plain Java and have the runtime lower them to hardware-specific backends. Think of it as "what if Java had CUDA, but it was just Java."

The interesting part: **all six compute kernels run through HAT dispatch** -- GEMV, RMSNorm, RoPE, SiLU, Softmax, and Attention. That's 100% kernel coverage, roughly 8,000 HAT dispatches per 32-token inference, producing output identical to the plain Java baseline. Character for character. The model tells the same bad programming joke either way.

## What Actually Works

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

The hybrid pattern for RMSNorm and Softmax deserves a note: the reduction step (sum of squares, finding max) runs in plain Java because HAT's Java sequential backend doesn't parallelize reductions well. The normalization step dispatches through HAT. It's pragmatic, not pretty, but it works and it'll map cleanly to GPU backends when those are ready.

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
./gradlew run --args="path/to/Llama-3.2-1B-Instruct-f16.gguf"
```

## Tests

Unit tests run without any model files:

```bash
./gradlew test
```

Integration tests need models. There are two tiers:

**TinyLlama tests** (lightweight, fast, good for CI):
```bash
./scripts/download_tinyllama.sh
TINY_LLAMA_PATH=$(pwd)/tinyllama-1.1b-chat-v1.0.Q2_K.gguf ./gradlew integrationTest
```

**Full Llama 3.2 FP16 tests** (the real thing, ~6 minutes each):
```bash
LLAMA_FP16_PATH=/path/to/Llama-3.2-1B-Instruct-f16.gguf ./gradlew integrationTest
```

The E2E tests are split per kernel. There's a `ChatIntegrationTestWith{Kernel}HAT` for each of the six kernels individually, plus `ChatIntegrationTestWithAllHAT` that enables all six simultaneously. Each test runs the same prompt ("Tell a joke about programming") and validates the output isn't gibberish using heuristics for repeated characters, non-ASCII ratio, and character diversity. The expected output (if you're curious): "Why did the programmer quit his job? Because he didn't get arrays."

## Current Limitations

This is intentionally specialized. It does one thing and does it correctly.

**One model only.** Architecture constants (2048 hidden size, 16 layers, 32 heads, 8 KV heads) are hardcoded for Llama 3.2 1B. There's no model auto-detection, no config parsing from GGUF metadata. If you point it at a 3B or 8B model, it won't give you a helpful error -- it'll just produce garbage (or crash, if you're lucky).

**FP16 and F32 tensors only.** No quantization support whatsoever. No `Q4_0`, no K-quants, nothing. F16 gets dequantized CPU-side via `Float.float16ToFloat()`. The model file is 2.5 GB because that's what FP16 costs you.

**Java sequential backend only.** HAT supports OpenCL and PTX backends for actual GPU execution. This project currently uses the Java sequential backend, which means HAT dispatch is running the kernels on CPU, in Java, sequentially. Yes, that means it's not faster than plain Java -- the point (for now) is correctness verification, not performance. The architecture is ready for GPU backends; the kernel code won't need to change.

**Greedy decoding only.** No top-k, no top-p, no temperature sampling. The model deterministically picks the most probable next token every time. This is fine for testing (reproducible output) but you wouldn't want it for creative text generation.

**No streaming.** The `chat()` method returns the complete response as a string. There's no token-by-token callback, no async generation.

## What's Next

Roughly in order of what would be most interesting to tackle:

**GPU backends.** The whole point of HAT is hardware acceleration, and the kernel code is already written in a dispatch-friendly way. Plugging in the OpenCL backend should (in theory) give immediate speedups on the GEMV kernel, which dominates inference time (~113 dispatches per token out of ~250 total). The hybrid kernels (RMSNorm, Softmax) will need their reduction steps reworked for GPU parallelism.

**Quantization.** Supporting `Q4_0` and `Q8_0` would make this practical for machines with less RAM. The K-quant formats (`Q4_K`, `Q6_K`) are more complex but also more accurate. Each quantization type has its own block structure and dequantization kernel -- and those kernels themselves could be HAT-dispatched.

**Model flexibility.** Reading architecture parameters from GGUF metadata instead of hardcoding them. This is mostly plumbing -- the GGUF reader already parses metadata, the constants just need to flow through.

**Sampling strategies.** Top-k and top-p sampling, temperature control. Straightforward to add on top of the existing logits output.

**Kernel fusion.** The current pipeline dispatches each kernel separately. Fusing RMSNorm with the QKV projection (which is just three GEMV calls) would reduce memory round-trips significantly -- assuming a GPU backend where that matters.

---

*Based on [mukel/llama3.java](https://github.com/mukel/llama3.java). Built with [Project Babylon](https://openjdk.org/projects/babylon/) (Java 26, code-reflection branch).*
