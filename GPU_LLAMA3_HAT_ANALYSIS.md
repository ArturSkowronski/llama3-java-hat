### Analysis: Implementing GPULlama3 using HAT (Project Babylon)

This document analyzes the feasibility and steps required to implement a GPU-accelerated Llama3 inference engine (similar to [GPULlama3.java](https://www.tornadovm.org/gpullama3)) using the **Hardware Accelerator Toolkit (HAT)** and **Project Babylon**.

---

### 0. Target Model: Llama-3.2-1B-Instruct (F16)
The implementation will target the **Llama-3.2-1B-Instruct** model in GGUF format (specifically the F16 version).

**Key Specifications:**
- **Parameters:** ~1.2B
- **Hidden Size:** 2048
- **Intermediate Size:** 8192
- **Number of Layers:** 16
- **Attention Heads:** 32
- **KV Heads:** 8 (Grouped-Query Attention)
- **Vocabulary Size:** 128,256
- **Context Length:** 128k (standard)

**Next Steps (Model):**
1. Finalize target architecture constants in Java.
2. Prepare weight loading strategy for F16 tensors.

---

### 1. Conceptual Mapping: TornadoVM vs. HAT

| Feature | TornadoVM (GPULlama3) | HAT (Project Babylon) |
| --- | --- | --- |
| **Code Capture** | Custom bytecode analysis / Graal JIT hooks | **Code Reflection (`@Reflect`)** - embeds symbolic code models directly in class files. |
| **Execution Model** | `TaskGraph` / `TornadoTask` | `Accelerator.compute()` / `ComputeContext` / `KernelContext`. |
| **Data Management** | `MemorySegment` / `TornadoArray` | **Off-heap Buffers** (`S32Array`, `F32Array`, etc.) using `KernelBufferContext`. |
| **Kernel Fusion** | Manual at the Java source level (combining loops). | Manual at the Java source level (method inlining in Code Model). |
| **Backend Support** | OpenCL, PTX, SPIR-V | Java Sequential (current), OpenCL/PTX (in development/private). |

**Next Steps (Conceptual):**
1. Map TornadoVM `TaskGraph` logic to HAT `ComputeContext` dispatch sequences.
2. Design the buffer lifecycle for weight management in HAT.

---

### 2. What Works (Current State of HAT)

*   **Code Capture**: HAT successfully captures Java logic via `@Reflect`. We can express complex kernels like Matrix Multiplication or RMSNorm in pure Java.
*   **Off-heap Memory**: HAT provides specialized array types (`F32Array`, etc.) that are compatible with GPU memory layouts and accessible via `KernelContext`.
*   **ND-Range Dispatch**: HAT supports 1D, 2D, and 3D ranges, essential for mapping transformer operations (e.g., tokens x heads x dimensions).
*   **Loop Unrolling/Inlining**: Since Babylon provides the full Code Model, the HAT backend can perform optimizations like constant folding and inlining before generating native code.

**Next Steps (Capabilities):**
1. Benchmark simple kernels to verify backend optimization.
2. Test 2D ND-Range dispatch for GEMV operations.

---

### 3. Challenges & Gaps

*   **Quantization Support**: Llama3 typically uses `Q4_0` or `Q8_0` quantization. HAT currently focuses on standard types (`S32`, `F32`). Implementing custom bit-packing/unpacking in HAT kernels will require careful bitwise operation handling in the Code Model.
*   **Kernel Fusion Efficiency**: TornadoVM allows building complex `TaskGraphs`. In HAT, fusion must be done by writing "mega-kernels" or by the backend automatically inlining `@Reflect` methods.
*   **GGUF Parsing**: The logic to parse GGUF files and map them to HAT's `KernelBuffer` needs to be implemented.
*   **Native Backends**: While HAT has an OpenCL backend, it may not yet be as optimized as TornadoVM's highly tuned PTX/OpenCL generators for specific NVIDIA/AMD architectures.

**Next Steps (Gaps):**
1. Prototype `Q4_0` dequantization logic using bitwise operators in a `@Reflect` method.
2. Investigate OpenCL backend performance for complex math.

---

### 4. Implementation Plan

#### Phase 1: Foundation (Current Status: ✅)
- [x] Verify Babylon/HAT runtime environment.
- [x] Implement basic "Vector Add" to test dispatch and memory.

#### Phase 2: Primitive Kernels (Current Status: ✅)
- [x] Implement `Matrix-Vector Multiplication` (GEMV) kernel in HAT.
- [x] Implement `RMSNorm` and `SiLU` activation kernels.
- [x] Implement `RoPE` (Rotary Positional Embedding) kernel.
- [x] Implement `Attention` kernel (GQA).

#### Phase 3: Data Management & GGUF (Current Status: 🚧)
- [x] Implement initial GGUF Metadata Reader using Java Foreign Memory API.
- [x] Verify GGUF reader with real-world model (TinyLlama-1.1B-Chat).
- [x] Extend GGUF reader to parse tensor information and map to HAT buffers.
- [ ] Implement dequantization kernels for `Q2_K`, `Q4_K`, etc., to run on GPU.
- [ ] Implement KV-Cache management using HAT buffers.

#### Phase 4: Integration
- [ ] Implement Llama 3.2 1B specialized inference loop using `Accelerator.compute()`.
- [ ] Implement Kernel Fusion: combine RMSNorm + QKV Projection into a single `@Reflect` method.

---

### 17. Next Step: Implementing LlamaModel (2026-02-07)

The next major milestone is to implement the `LlamaModel` class, which serves as the central orchestrator for Llama 3.2 1B Instruct (FP16) inference using HAT.

#### What is LlamaModel?

`LlamaModel` is the core class that bridges the GGUF file format with HAT's GPU acceleration capabilities. It is responsible for:

1.  **Weight Loading**: Reading tensor data from the GGUF file and mapping it into HAT-compatible buffers (`F32Array`).
2.  **Dequantization**: Converting F16 (half-precision) weights to F32 (single-precision) for computation, either on CPU during load or via GPU kernels.
3.  **Inference Orchestration**: Coordinating the execution of transformer layers by dispatching HAT kernels (`GEMV`, `RMSNorm`, `SiLU`, `RoPE`, `Attention`) in the correct sequence.
4.  **KV-Cache Management**: Allocating and managing Key-Value cache buffers for efficient autoregressive generation.

#### Why is LlamaModel the Next Step?

| Reason | Explanation |
|--------|-------------|
| **Foundation is Ready** | The GGUF reader (`GGUFReader`) is fully functional and verified. It can parse metadata and tensor information from real model files. |
| **Kernels are Implemented** | Primitive kernels (`GEMV`, `RMSNorm`, `SiLU`, `RoPE`) have been implemented and verified with the HAT Java sequential backend. |
| **Infrastructure is Stable** | Build system, CI/CD (GitHub Actions), and testing suites (unit + integration) are operational. |
| **Clear Target** | The project is specialized for Llama 3.2 1B Instruct (FP16), so `LlamaModel` can use hardcoded architecture constants without over-generalization. |

#### What Needs to Be Done Now?

1.  **Create `LlamaModel.java`**:
    *   Define architecture constants: `HIDDEN_SIZE=2048`, `INTERMEDIATE_SIZE=8192`, `NUM_LAYERS=16`, `NUM_HEADS=32`, `NUM_KV_HEADS=8`, `HEAD_DIM=64`.
    *   Implement a constructor that takes a `Path` to the GGUF file and uses `GGUFReader` to load metadata.

2.  **Implement Tensor Mapping**:
    *   Create a `mapTensor(GGUFTensorInfo)` method that reads raw bytes from the GGUF file and populates an `F32Array`.
    *   For F16 tensors: implement `Float.float16ToFloat()` dequantization (CPU-side initially).
    *   For F32 tensors: direct copy into the buffer.

3.  **Implement Single Transformer Block**:
    *   Chain kernels: `RMSNorm` → `GEMV` (Q, K, V projections) → `RoPE` → `Attention` → `GEMV` (output projection) → `RMSNorm` → `FFN` (SwiGLU).
    *   Use `Accelerator.compute()` to dispatch each kernel.

4.  **Implement Attention Mechanism**:
    *   Grouped-Query Attention (GQA) with 32 query heads and 8 KV heads.
    *   Softmax kernel for attention scores.
    *   Weighted sum of values.

5.  **Implement KV-Cache**:
    *   Allocate `F32Array` buffers for K and V caches: `NUM_LAYERS × MAX_SEQ_LEN × NUM_KV_HEADS × HEAD_DIM`.
    *   Update cache during each forward pass.

6.  **Add Integration Tests**:
    *   Extend `TinyLlamaIntegrationTest` (or create a new suite) to verify tensor loading and single-layer forward pass.

#### Implementation Order (Recommended)

| Step | Task | Complexity | Prerequisite |
|------|------|------------|--------------|
| 1 | Create `LlamaModel` skeleton with constants | Low | None |
| 2 | Implement `mapTensor` for F32/F16 | Medium | Step 1 |
| 3 | Implement Softmax kernel | Low | None |
| 4 | Implement Attention kernel (GQA) | High | Step 3 |
| 5 | Implement single transformer block | High | Steps 2, 4 |
| 6 | Implement KV-Cache | Medium | Step 5 |
| 7 | Implement full forward pass (16 layers) | High | Step 6 |
| 8 | Implement tokenizer (BPE) | Medium | None |
| 9 | End-to-end text generation | High | Steps 7, 8 |

---

### 14. Verification of Build and Run Scripts (2026-02-06)

I have performed a complete verification of the project's build and execution infrastructure to ensure that the instructions provided in the `README.md` are accurate and functional.

**Verification Steps & Results:**
1.  **Unit Tests**: Ran `./gradlew clean test` with `JAVA_BABYLON_HOME` set. All tests passed, confirming the core logic (GGUF metadata parsing) is intact.
2.  **Integration Tests**: Ran `./gradlew cleanIntegrationTest integrationTest` with both `JAVA_BABYLON_HOME` and `TINY_LLAMA_PATH` set. The tests successfully verified the header of the real `tinyllama-1.1b-chat-v1.0.Q2_K.gguf` file.
3.  **Application Execution**: Ran the main application using `./gradlew run --args="tinyllama-1.1b-chat-v1.0.Q2_K.gguf"`. The application correctly parsed and printed the metadata and tensor information for the 1.1B model, demonstrating that the `GGUFReader` is robust and the Gradle `run` task is correctly configured.
4.  **Environment Validation**: Confirmed that the build fails gracefully with a clear error message when `JAVA_BABYLON_HOME` is missing.

**Conclusion:**
The project infrastructure is stable, and the documented build/run cycles are fully operational. This solid foundation allows us to proceed with the specialized Llama 3.2 1B implementation and high-performance kernel development.

---

### 6. Phase 3 Detail: GGUF Reader Verification (2026-02-06)

I have successfully performed an end-to-end verification of the GGUF reader logic using a real-world model file: `tinyllama-1.1b-chat-v1.0.Q2_K.gguf`.

**Findings:**
- **Robustness**: The `MemorySegment` based reader correctly handles large files (460MB+) and complex GGUF metadata structures.
- **Model Metadata**: Successfully extracted Llama-specific keys (embedding length, head count, etc.) which will be used to initialize the HAT `Accelerator` and `NDRange` configurations.
- **Tensor Discovery**: Parsed all 201 tensors, providing the necessary offsets and shapes for the next phase of mapping weights to `F32Array` or `S32Array` HAT buffers.

**Next Steps (GGUF):**
1. Implement `GGUFTensorInfo` extraction.
2. Map tensor memory segments to HAT `F32Array`.

---

### 7. Proposed Next Steps (Updated 2026-02-06)

1.  **High-Performance Dequantization**: Implement HAT kernels for GGML quantization formats (`Q2_K`, `Q4_K`, etc.). This is critical because currently dequantization happens on the CPU during model loading, which is slow and memory-intensive. Moving dequantization to the GPU (via HAT kernels) will allow loading quantized weights directly into GPU memory.
2.  **Transformer Loop Integration**: Start implementing the main transformer block logic. This involves chaining the implemented kernels (`RMSNorm` -> `GEMV` (QKV) -> `RoPE` -> `Attention` -> `GEMV` (Output) -> `RMSNorm` -> `FFN`).
3.  **KV-Cache Implementation**: Design and implement the KV-cache using HAT `F32Array` buffers to support multi-token generation efficiently.
4.  **Backend Optimization**: Investigate the "IllegalStateException: Why don't we have enough captures" error observed in complex lambdas and ensure compatibility with more advanced HAT backends (like OpenCL) for real GPU execution.

---

### 8. Accomplishments in Phase 2 & 3 (Summary)

1.  **Primitive Kernels Completed**:
    *   Implemented and verified `GEMV` (Matrix-Vector Multiplication).
    *   Implemented and verified `RMSNorm` (Root Mean Square Layer Normalization).
    *   Implemented and verified `SiLU` (Sigmoid Linear Unit) activation.
    *   Implemented `RoPE` (Rotary Positional Embedding) kernel.
2.  **Verification**:
    *   All kernels were tested with the HAT Java sequential backend.
    *   The `RMSNorm` and `SiLU` kernels were integrated into the `LlamaModel` and verified with sample data.
3.  **Code Improvement**:
    *   Fixed a HAT capture issue ("IllegalStateException: Why don't we have enough captures") by wrapping `accelerator.compute` calls in instance methods, preventing incorrect variable capture in nested lambdas.

### 9. Focusing on Llama-3.2-1B-Instruct (F16) (Updated 2026-02-06)

Following the requirement to avoid overly generalized solutions, the implementation is now strictly focused on **Llama-3.2-1B-Instruct (FP16, ~2.5 GB)**.

**Model Specifics for Llama-3.2-1B-Instruct:**
- **Architecture**: Standard Transformer with Grouped-Query Attention (GQA).
- **Hidden Size**: 2048
- **Intermediate Size (FFN)**: 8192
- **Number of Layers**: 16
- **Attention Heads**: 32 (Query), 8 (Key/Value)
- **Vocabulary Size**: 128,256
- **Tensor Types (FP16 version)**: Primarily `F16` (type 1) for weights, `F32` (type 0) for norms.
- **Dequantization**: Focus on `F16` to `F32` dequantization kernels on GPU.

**Implementation Adjustments:**
1.  **Constants**: Hardcode architecture constants for 1.2B parameters to optimize kernel dispatch (e.g., block sizes, unrolling factors).
2.  **Memory Layout**: Optimize `F32Array` buffer allocations specifically for 2048 hidden dimension.
3.  **Kernel Optimization**: Tailor `GEMV`, `RMSNorm`, and `RoPE` kernels for the specific dimensions of the 1B model (rows=2048, heads=32, head_size=64).
4.  **Loading**: Optimize the GGUF loader to prioritize memory mapping of F16 tensors directly into GPU-accessible buffers where possible.

### 10. Process Knowledge & Guidelines (2026-02-06)

I have established new guidelines for the development process to ensure knowledge preservation and code quality.

**Accomplishments:**
- **Created `Agent.md`**: Tracks high-level goals, current status, key decisions/rationales, and next steps. This file serves as the "brain" of the project across sessions.
- **Created `Claude.md`**: Defines implementation principles (minimalism, modern Java, specialization) and HAT-specific usage rules (e.g., avoiding capture issues).
- **Established Testing Protocol**: Integrated JUnit 6 and added `GGUFReaderTest` to verify metadata parsing and tensor discovery.
- **Improved Security/Privacy**: Updated `.gitignore` to prevent session-specific knowledge files (`Agent.md`, `Claude.md`) from being tracked if they contain sensitive or temporary state (though they are primarily for knowledge preservation).

**Rationale:**
- **Why `Agent.md`?** To provide a persistent context for future sessions, ensuring the next agent knows exactly where we left off and why certain paths were chosen.
- **Why `Claude.md`?** To enforce a constant coding style and technical strategy (specialization on Llama 3.2 1B).
- **Why JUnit 6?** To maintain high reliability as the complexity of the inference engine grows.

---

### 11. Infrastructure & Build Portability (2026-02-06)

I have improved the build system to remove hardcoded paths and ensure portability across different development environments, including platform-specific JDK path resolution.

**Accomplishments:**
- **Environment Variable Integration**: Replaced `JAVA_BABYLON_HOME` with `JAVA_BABYLON_ROOT` in `build.gradle.kts`.
- **macOS Path Resolution**: Added logic to `build.gradle.kts` to automatically append `/Contents/Home` to `JAVA_BABYLON_ROOT` when running on macOS, ensuring the correct JDK home is used.
- **Build Pre-check**: Implemented a check in Gradle that fails the build early if `JAVA_BABYLON_ROOT` is not defined.
- **Documentation**: Updated `README.md` detailing the requirement of setting `JAVA_BABYLON_ROOT` and explaining the macOS path resolution.

**Rationale:**
- **Why `JAVA_BABYLON_ROOT`?** To provide a consistent root directory for Project Babylon across different OSs, while handling the unique directory structure of JDKs on macOS (`/Contents/Home`) automatically.
- **Why a pre-check?** To provide clear, actionable feedback when the environment is not correctly configured.

---

### 12. Integration Testing Suite (2026-02-06)

I have introduced a dedicated integration testing suite for TinyLlama to separate heavy model-based tests from unit tests.

**Accomplishments:**
- **Dedicated Test Class**: Created `TinyLlamaIntegrationTest` using JUnit 6 and `@Tag("integration")`.
- **Focused Verification**: The test currently focuses on verifying the GGUF Header reading for TinyLlama, as requested.
- **Environment-Driven Tests**: The integration tests are gated by the `TINY_LLAMA_PATH` environment variable, ensuring they only run when the model is available.
- **Gradle Integration**: Added an `integrationTest` task to `build.gradle.kts` to run these tests specifically.
- **Model Downloader**: Provided `download_tinyllama.sh` to automate the acquisition of the required TinyLlama GGUF model from HuggingFace.

**Rationale:**
- **Why a separate suite?** Model files are large and tests involving them are slow. Decoupling them allows for fast unit testing during development while still enabling thorough integration checks.
- **Why focus on Header reading?** Following the latest requirement to assume no LLAMA implementation yet, the integration test ensures that the foundational GGUF reading logic is robust for the targeted model before proceeding to complex inference logic.
- **Why `@Tag("integration")`?** It provides a clean way to filter tests at the build system level.
- **Why a download script?** To make the project reproducible and reduce the friction of setting up a new development environment.

### 14. Repository Reorganization (2026-02-06)

Organized scripts and test outputs to maintain a cleaner project structure.
- Moved `download_tinyllama.sh` to the `scripts/` directory.
- Moved `integration_test_output.txt` to the `results/` directory.
- Updated `.gitignore` to exclude the `results/` directory.

### 15. Continuous Integration with GitHub Actions (2026-02-06)

Implemented a GitHub Actions workflow to automate the building of Project Babylon and HAT from source.
- **Babylon Build**: The workflow clones the `openjdk/babylon` repository, configures it, and builds the JDK images.
- **HAT Build**: Automatically builds HAT artifacts using the newly built Babylon JDK.
- **Caching Strategy**: Uses GitHub Actions cache to store the Babylon JDK and HAT build artifacts, significantly reducing subsequent run times.
- **Test Automation**: Automatically runs unit tests and integration tests (including TinyLlama header verification) on every push and pull request.
- **Artifact Preservation**: Preserves the built Babylon JDK and HAT artifacts for future use or manual inspection.

### 16. Verification and Monitoring (2026-02-06)

I have performed a final verification of the entire build and test pipeline to ensure the project is in a stable state for further development.

**Verification Results:**
1.  **Application Execution**: Verified that `./gradlew run --args="tinyllama-1.1b-chat-v1.0.Q2_K.gguf"` correctly parses and displays the TinyLlama model's metadata and tensors.
2.  **Full Build Cycle**: Confirmed that `./gradlew clean build test integrationTest` passes successfully with both `JAVA_BABYLON_HOME` and `TINY_LLAMA_PATH` set.
3.  **CI Readiness**: The successful execution of the full suite locally validates the logic implemented in the GitHub Actions workflow.

**Monitoring Points:**
- **Model Integrity**: The integration test now correctly fails if the model file is missing or corrupted, which was previously a silent failure point (empty file).
- **Environment Stability**: The build pre-check ensures that any environment misconfiguration (missing `JAVA_BABYLON_HOME`) is caught immediately.

The project is now monitored and verified as stable.

---

### 18. Current Limitations (2026-02-07)

This implementation is **specialized for Llama 3.2 1B Instruct (FP16)** and intentionally avoids over-generalization.

#### Model Support Limitations
| Limitation | Description |
|------------|-------------|
| **Target Model Only** | Llama 3.2 1B Instruct in FP16 GGUF format (~2.5 GB) |
| **No K-Quant Support** | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` formats not implemented |
| **No Legacy Quantization** | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` formats not supported |
| **Hardcoded Architecture** | Model constants fixed for 1B model (2048 hidden, 16 layers, 32 heads) |

#### Tensor Type Support
| Type | Status | Notes |
|------|--------|-------|
| F32 (type 0) | ✅ Supported | Direct copy to HAT buffer |
| F16 (type 1) | ✅ Supported | CPU-side `Float.float16ToFloat()` dequantization |
| Q4_0 (type 2) | ❌ Not supported | Would need bitwise unpacking kernel |
| Q4_1 (type 3) | ❌ Not supported | Would need bitwise unpacking kernel |
| Q2_K (type 10) | ❌ Not supported | Complex super-block structure |
| Q3_K (type 11) | ❌ Not supported | Complex super-block structure |
| Q4_K (type 12) | ❌ Not supported | Complex super-block structure |
| Q5_K (type 13) | ❌ Not supported | Complex super-block structure |
| Q6_K (type 14) | ❌ Not supported | Complex super-block structure |

#### Backend Limitations
- **Java Sequential Only**: Uses HAT's Java sequential backend for correctness verification
- **No GPU Acceleration**: OpenCL/PTX backends not yet integrated
- **No Kernel Fusion**: Each kernel dispatched separately

#### Inference Limitations
- **No KV-Cache**: Multi-token generation not implemented
- **No Tokenizer**: BPE encoding/decoding not implemented
- **No Attention**: Softmax and GQA attention kernels pending
- **No End-to-End**: Cannot generate text yet

---

### 19. For Future: Generalized Implementation Checklist (2026-02-07)

Checklist for extending this implementation to support more models and quantization formats.

#### Quantization Support
- [ ] Implement `Q4_0` dequantization kernel (block size 32, 18 bytes/block)
- [ ] Implement `Q8_0` dequantization kernel (block size 32, 34 bytes/block)
- [ ] Implement `Q2_K` dequantization kernel (super-block 256, 84 bytes)
- [ ] Implement `Q3_K` dequantization kernel (super-block 256, 110 bytes)
- [ ] Implement `Q4_K` dequantization kernel (super-block 256, 144 bytes)
- [ ] Implement `Q5_K` dequantization kernel (super-block 256, 176 bytes)
- [ ] Implement `Q6_K` dequantization kernel (super-block 256, 210 bytes)
- [ ] Move dequantization from CPU to GPU via HAT `@Reflect` kernels

#### Model Flexibility
- [ ] Make architecture constants configurable from GGUF metadata
- [ ] Support variable hidden sizes (1024, 2048, 4096, 8192)
- [ ] Support variable layer counts (16, 32, 40, 80)
- [ ] Support variable head configurations (MHA, GQA, MQA)
- [ ] Add model auto-detection from GGUF `general.architecture` key

#### Inference Pipeline
- [ ] Implement Softmax kernel
- [ ] Implement Grouped-Query Attention (GQA) kernel
- [ ] Implement KV-Cache management with HAT buffers
- [ ] Implement single transformer block (chain all kernels)
- [ ] Implement full forward pass loop
- [ ] Implement BPE tokenizer (from GGUF vocabulary)
- [ ] Implement token sampling (greedy, top-k, top-p)

#### Performance Optimization
- [ ] Integrate OpenCL backend for GPU execution
- [ ] Integrate PTX backend for NVIDIA GPUs
- [ ] Implement kernel fusion (RMSNorm + QKV projection)
- [ ] Optimize memory layout for coalesced GPU access
- [ ] Add batch processing support

#### Testing & Validation
- [ ] Add unit tests for each dequantization kernel
- [ ] Add integration tests for multiple model sizes (1B, 3B, 8B)
- [ ] Add performance benchmarks (tokens/second)
- [ ] Add numerical accuracy tests against reference implementation (llama.cpp)

---

### 20. Tensor Mapping Implementation (2026-02-07)

Implemented Step 2 of the LlamaModel implementation plan: tensor mapping from GGUF to HAT buffers.

**Accomplishments:**
1.  **`mapTensor()` Method**: Loads tensors from GGUF file into HAT `F32Array` buffers.
    *   F32 tensors: direct copy from memory-mapped file
    *   F16 tensors: dequantization via `Float.float16ToFloat()`
    *   Caching: tensors are cached after first load to avoid redundant I/O
2.  **Helper Methods**: Added `hasTensor()` and `getTensorInfo()` for tensor discovery.
3.  **Test Infrastructure**: Extended `MinimalGGUFGenerator` to create GGUF files with llama architecture and F32 tensors for unit testing.
4.  **Unit Tests**: Added 3 new tests for tensor mapping (F32 loading, caching, error handling).

**Implementation Details:**
```java
public F32Array mapTensor(String tensorName) throws IOException {
    // 1. Check cache
    // 2. Find tensor info in metadata
    // 3. Validate type (F32 or F16 only)
    // 4. Memory-map the tensor data region
    // 5. Copy/dequantize into F32Array
    // 6. Cache and return
}
```

**PR**: https://github.com/ArturSkowronski/llama3-java-hat/pull/6

**Next Step**: Step 3 - Implement Softmax kernel (prerequisite for Attention)

---

### 21. Softmax Kernel Implementation (2026-02-07)

Implemented Step 3 of the LlamaModel implementation plan: Softmax kernel for attention mechanism.

**PR**: https://github.com/ArturSkowronski/llama3-java-hat/pull/7

#### Files Created
- **`Softmax.java`** (164 lines): Numerically stable softmax implementation
- **`SoftmaxTest.java`** (145 lines): 5 unit tests

#### Implementation Details
```java
// Numerically stable: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
public void apply(F32Array input, int size)      // Full array softmax
public void applyRow(F32Array input, int offset, int size)  // Row-wise for attention
```

#### Tests
| Test | Purpose |
|------|---------|
| `testSoftmaxSumsToOne` | Outputs sum to 1.0 |
| `testSoftmaxValues` | Exact value verification |
| `testSoftmaxMonotonicity` | Ordering preserved |
| `testSoftmaxRowApply` | Row-wise application |
| `testSoftmaxNumericalStability` | No NaN/Inf with large inputs |

#### Code Review Feedback (Gemini Code Assist)

**Review Summary**: "Code is well-structured and tests are comprehensive. Review focuses on improving consistency and robustness."

**Issue 1: Mixed CPU/GPU Operations in `apply()` (Medium Priority)**
- **Location**: `Softmax.java`, line 58
- **Problem**: Steps 1-2 (find max, compute exp/sum) run on CPU, but Step 3 (normalize) dispatches to GPU. This is inefficient due to data transfers and kernel dispatch overhead.
- **Suggestion**: Make `apply()` fully CPU-based like `applyRow()` for consistency. A fully GPU-accelerated version can be implemented later.
- **Recommended Fix**:
```java
if (sum > 0.0f) {
    final float invSum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        input.array(i, input.array(i) * invSum);
    }
}
```

**Issue 2: Division by Zero in `applyRow()` (Medium Priority)**
- **Location**: `Softmax.java`, line 102
- **Problem**: If all inputs are `Float.NEGATIVE_INFINITY`, sum could be zero, causing `1.0f / sum` to produce `Infinity` and subsequent `NaN` results.
- **Suggestion**: Guard against division by zero for robustness.
- **Recommended Fix**:
```java
if (sum > 0.0f) {
    float invSum = 1.0f / sum;
    for (int i = 0; i < rowSize; i++) {
        input.array(rowOffset + i, input.array(rowOffset + i) * invSum);
    }
}
```

#### Action Items from Review
- [ ] Remove GPU dispatch from `apply()`, make fully CPU-based for now
- [ ] Add `sum > 0.0f` guard in both `apply()` and `applyRow()`
- [ ] Plan future GPU-accelerated version for both methods

**Next Step**: Step 4 - Implement Attention kernel (GQA) using this Softmax kernel

---

### 22. PR #8: Softmax Kernel and CI Resilience (2026-02-08)

**PR**: https://github.com/ArturSkowronski/llama3-java-hat/pull/8

This PR combines the Softmax kernel implementation with CI build resilience improvements backported from the conference-jvm-in-age-ai-2026 repository.

#### Changes Summary
| File | Additions | Deletions | Description |
|------|-----------|-----------|-------------|
| `.github/workflows/ci.yml` | 32 | 9 | CI self-healing mechanisms |
| `Softmax.java` | 163 | 0 | Numerically stable softmax kernel |
| `SoftmaxTest.java` | 144 | 0 | 5 unit tests for softmax |

#### CI Resilience Improvements

**1. Boot JDK Update**
- Updated from JDK 23 to JDK 25 for better compatibility with Babylon

**2. Improved Caching Strategy**
```yaml
key: ${{ runner.os }}-babylon-hat-${{ hashFiles('.github/workflows/ci.yml') }}-${{ hashFiles('babylon/.git/HEAD') }}
restore-keys: |
  ${{ runner.os }}-babylon-hat-${{ hashFiles('.github/workflows/ci.yml') }}-
  ${{ runner.os }}-babylon-hat-
```
- Cache key now includes workflow file hash for invalidation on CI changes
- Added restore-keys for partial cache hits

**3. Self-Healing Build Mechanism**
- Added `continue-on-error: true` to initial build step
- Implemented automatic reconfiguration and retry on failure:
```yaml
- name: Retry Build if failed
  if: steps.build-jdk.outcome == 'failure'
  run: |
    cd babylon
    make reconfigure JOBS=$(nproc) CONF=linux-x86_64-server-release
    make images JOBS=$(nproc) CONF=linux-x86_64-server-release
```

**4. Incremental Build Support**
- Checks if configuration exists before full reconfigure
- Uses `make check-conf` to detect stale configurations
- Only reconfigures when necessary

**5. Explicit Branch Reference**
- Added `ref: code-reflection` to Babylon checkout for reproducibility

#### Softmax Kernel (Same as PR #7)
- Numerically stable implementation: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`
- GPU-accelerated normalization via HAT `@Reflect`
- Row-wise application for attention scores
- 5 comprehensive unit tests

#### Code Review Feedback (Gemini Code Assist)

**Review Summary**: "Code is well-structured and tests are comprehensive."

**Performance Issues Identified (All Medium Priority)**:

| Line | Location | Issue |
|------|----------|-------|
| 44 | `apply()` - find max | CPU-side max finding is a performance bottleneck for large inputs |
| 52 | `apply()` - exp/sum | Computing `exp(x - max)` and sum on CPU negates accelerator benefits |
| 88 | `applyRow()` - find max | Repeated pattern of CPU-side max finding |
| 96 | `applyRow()` - exp/sum | Same CPU bottleneck for row-wise operations |
| 102 | `applyRow()` - normalize | Normalization on CPU instead of accelerator |
| 150 | `main()` - test values | Hardcoded expected values; suggest programmatic calculation |

**Recommendations**:
1. Implement GPU-accelerated max-reduction kernel
2. Move `exp(x - max)` and sum computation to accelerator
3. Use accelerator for normalization in `applyRow()` (like `apply()` does)
4. Consider programmatic expected value calculation for floating-point tests

**Action Items for Future Optimization**:
- [ ] Implement GPU max-reduction kernel for `apply()` and `applyRow()`
- [ ] Move exp/sum computation to GPU
- [ ] Unify normalization approach (both methods should use accelerator)
- [ ] Add division-by-zero guards (`sum > 0.0f` check)

---

### 23. Attention and RMSNorm/SiLU Implementation (2026-02-11)

Implemented the core kernels for Llama 3.2 1B inference.

#### 1. Attention Kernel (GQA)
- **File**: `Attention.java`
- **Functionality**: Implements Scaled Dot-Product Attention for one query head.
- **Support**: Designed for Grouped-Query Attention (GQA) with 32 query heads and 8 KV heads.
- **Kernels**: `computeScores` (Q*K^T / scale) and `computeValues` (Scores * V).
- **PR**: https://github.com/ArturSkowronski/llama3-java-hat/pull/9

#### 2. RMSNorm and SiLU Kernels
- **Files**: `RMSNorm.java`, `SiLU.java`
- **RMSNorm**: Implements Root Mean Square Layer Normalization. Uses hybrid approach (CPU for sum of squares, GPU for normalization).
- **SiLU**: Implements Sigmoid Linear Unit activation function. Fully GPU-accelerated.
- **PR**: https://github.com/ArturSkowronski/llama3-java-hat/pull/10

#### Verification
All kernels have been verified with comprehensive unit tests:
- `AttentionTest.java`
- `RMSNormTest.java`
- `SiLUTest.java`

Verified that all tests pass using `./gradlew test`.
