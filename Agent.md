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

### 4. Next Steps
*   **Confirm Integration Test Solution**: Waiting for user confirmation on the JUnit 6 based integration suite.
*   **Inference Loop**: Chain kernels to implement the full transformer layer.
*   **KV-Cache**: Implement efficient management for Key-Value pairs in HAT buffers.
*   **GPU Backend**: Transition from Java sequential to OpenCL/PTX for performance.
*   **GitHub Actions**: Finalized and local verification of the CI workflow logic for building Babylon and running tests.
