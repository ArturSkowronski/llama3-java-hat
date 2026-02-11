# Agent Knowledge & Progress

### 1. Project Goal
Implement a GPU-accelerated Llama 3.2 1B Instruct (FP16) inference engine using Java, Project Babylon, and the Hardware Accelerator Toolkit (HAT).

### 2. Current Status
*   **GGUF Reader**: Functional for metadata and tensor info extraction. Supports `F32` and `F16` dequantization.
*   **LlamaModel**: Removed. Implementation will be restarted from scratch to focus on Llama 3.2 1B Instruct (FP16) specifically.
*   **Kernels**: `GEMV`, `RMSNorm`, `SiLU`, and `RoPE` implemented and verified with HAT Java sequential backend.
*   **Testing**: 
    *   Unit tests: `GGUFReaderTest` (JUnit 6). Verified and passing.
    *   Integration tests: `TinyLlamaIntegrationTest` (JUnit 6), tagged with `@Tag("integration")`. Currently verifies GGUF Header reading for TinyLlama. Verified and passing.
    *   Automation: `download_tinyllama.sh` script (moved to `scripts/`) to fetch test models.
*   **Infrastructure**: Decoupled build from hardcoded paths by using `JAVA_BABYLON_ROOT` environment variable (which points to the Babylon root; on macOS, it automatically adds `/Contents/Home` to resolve the JDK home). 
*   **Verification**: All Gradle scripts from `README.md` (`test`, `integrationTest`, `run`) have been verified to work properly. Full build and test cycle verified as stable.
*   **Fixes**: Resolved "IllegalStateException: Why don't we have enough captures" by avoiding nested lambdas in `accelerator.compute`. Corrected the main class in `build.gradle.kts`.

### 3. Decisions & Rationale
*   **Specialization over Generalization**: Focused on Llama 3.2 1B FP16 (~2.5GB) to optimize performance and reduce complexity. Supporting all GGUF quantization levels (K-Quants) on GPU is a massive task deferred for later.
*   **Modern Java**: Utilizing `java.lang.foreign` (Foreign Function & Memory API) for efficient GGUF parsing and buffer mapping.
*   **HAT `@Reflect`**: Leveraging Babylon's code reflection to write GPU kernels in pure Java.

### 4. Next Steps (Updated 2026-02-11)

**Immediate Next Step: Single Transformer Block**

The RMSNorm and SiLU kernels (PR #10) have been implemented. The next step is to combine all implemented kernels (`RMSNorm`, `Attention`, `SiLU`, `GEMV`, `RoPE`) into a **single transformer block** execution logic.

**Implementation Order:**
| Step | Task | Status | PR |
|------|------|--------|----|
| 1 | LlamaModel skeleton with constants | ✅ Done | #5 |
| 2 | mapTensor for F32/F16 loading | ✅ Done | #6 |
| 3 | Softmax kernel + CI resilience | ✅ Merged | #8 |
| 4 | Attention kernel (GQA) | ✅ Done | #9 |
| 5 | RMSNorm and SiLU kernels | ✅ Done | #10 |
| 6 | GEMV and RoPE kernels | ✅ Done | #11 |
| 7 | **Single transformer block** | ✅ Done | #11 |
| 8 | KV-Cache management | Pending | - |
| 9 | Full forward pass (16 layers) | Pending | - |
| 10 | BPE Tokenizer | Pending | - |
| 11 | End-to-end text generation | Pending | - |

**Attention Kernel Requirements:**
*   Grouped-Query Attention: 32 query heads, 8 KV heads (4:1 ratio)
*   Head dimension: 64
*   Operations: Q×K^T → scale → Softmax → ×V
*   Use existing Softmax kernel for attention scores

**Deferred Tasks:**
*   **GPU Backend**: Transition from Java sequential to OpenCL/PTX (after correctness verified)
*   **Kernel Fusion**: RMSNorm + QKV projection (optimization phase)
*   **Code Review Fixes for Softmax**: Address division-by-zero guard and CPU-only normalization

### 5. GitHub Actions CI (2026-02-07)
*   **CI Pipeline**: Successfully implemented and verified GitHub Actions workflow that:
    *   Builds Project Babylon from source and caches the artifacts.
    *   Builds HAT using the Babylon JDK.
    *   Runs unit tests and TinyLlama integration tests.
*   **Key Fix**: Artifact extraction loses executable permissions. Fixed by adding `chmod +x` for JDK binaries (`bin/`) and lib executables (`jspawnhelper`, `*.so`).
*   **Environment**: Uses `JAVA_BABYLON_ROOT` pointing to the artifact root, with JDK at `$JAVA_BABYLON_ROOT/build/linux-x86_64-server-release/images/jdk` and HAT at `$JAVA_BABYLON_ROOT/hat/build`.

### 6. Recent Progress (2026-02-11)

**Completed PRs:**
1.  **PR #5 - LlamaModel Skeleton**: Minimal skeleton with hardcoded Llama 3.2 1B constants and GGUF validation.
2.  **PR #6 - Tensor Mapping**: `mapTensor()` method for loading F32/F16 tensors into HAT `F32Array` buffers with caching.
3.  **PR #7 - Softmax Kernel**: Numerically stable softmax with 5 unit tests.
4.  **PR #9 - Attention Kernel**: Grouped-Query Attention (GQA) implementation with unit tests.
5.  **PR #10 - RMSNorm & SiLU Kernels**: RMSNorm and SiLU activation kernels with unit tests.
6.  **PR #11 - GEMV, RoPE & TransformerBlock**: Matrix-vector multiplication, Rotary Position Embeddings, and the orchestrator for a single transformer layer.

**Code Review Action Items (from PR #7):**
- [ ] Remove GPU dispatch from `Softmax.apply()`, make fully CPU-based
- [ ] Add `sum > 0.0f` guard in `apply()` and `applyRow()` to prevent NaN
- [ ] Plan future GPU-accelerated Softmax version

### 7. Session Handoff Notes

For the next session, the developer should:
1.  **Complete GQA implementation**: The `TransformerBlock.forward` currently has a placeholder for the full GQA attention mechanism. This needs to be implemented using the `Attention` and `Softmax` kernels, including KV-cache updates.
2.  **Verify Forward Pass**: Add more detailed tests for `TransformerBlock` that verify numerical correctness of the full forward pass against a known baseline.
3.  **Full Model Forward**: Once a single block is verified, implement the full 16-layer forward pass in `LlamaModel`.
