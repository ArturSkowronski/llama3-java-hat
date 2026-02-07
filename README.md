### Llama 3 HAT Implementation

[![Build Babylon and Run Tests](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/ci.yml/badge.svg)](https://github.com/ArturSkowronski/llama3-java-hat/actions/workflows/ci.yml)

This project implements Llama 3.2 1B Instruct (FP16) inference using Project Babylon and HAT (Hardware Accelerator Toolkit).

#### Prerequisites

Before building or running the project, you must set the `JAVA_BABYLON_ROOT` environment variable to the root directory of your local Babylon repository clone.

```bash
export JAVA_BABYLON_ROOT=/path/to/your/babylon
```

The build system expects the HAT artifacts to be present in `$JAVA_BABYLON_ROOT/hat/build` (or `$JAVA_BABYLON_ROOT/Contents/Home/hat/build` on macOS).

#### Building

```bash
./gradlew build
```

#### Running

```bash
./gradlew run --args="path/to/your/model.gguf"
```

## Running Integration Tests

Integration tests require the TinyLlama model. You can download it using the provided script:

```bash
./scripts/download_tinyllama.sh
```

Then run the integration tests via Gradle:

```bash
JAVA_BABYLON_ROOT=/path/to/babylon TINY_LLAMA_PATH=$(pwd)/tinyllama-1.1b-chat-v1.0.Q2_K.gguf ./gradlew integrationTest
```

---

### Current Limitations

This implementation is **specialized for Llama 3.2 1B Instruct (FP16)** and intentionally avoids over-generalization. The following limitations apply:

#### Model Support
- **Target Model Only**: Llama 3.2 1B Instruct in FP16 GGUF format (~2.5 GB)
- **No Quantization Support**: K-Quant formats (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`) are not supported
- **No Legacy Quantization**: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` formats are not implemented
- **Hardcoded Architecture**: Model constants (hidden size, layers, heads) are fixed for 1B model

#### Tensor Types
- **F32**: Fully supported (direct copy)
- **F16**: Supported via CPU-side `Float.float16ToFloat()` dequantization
- **Quantized Types**: Not supported (will return size 0 or fail)

#### Backend
- **Java Sequential Only**: Currently uses HAT's Java sequential backend for correctness verification
- **No GPU Acceleration**: OpenCL/PTX backends are not yet integrated
- **No Kernel Fusion**: Each kernel dispatched separately (no RMSNorm+QKV fusion)

#### Inference
- **No KV-Cache**: Multi-token generation not yet implemented
- **No Tokenizer**: BPE encoding/decoding not implemented
- **No Attention**: Softmax and GQA attention kernels pending

---

### For Future

Checklist for a more generalized Llama implementation:

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

#### Performance
- [ ] Integrate OpenCL backend for GPU execution
- [ ] Integrate PTX backend for NVIDIA GPUs
- [ ] Implement kernel fusion (RMSNorm + QKV projection)
- [ ] Optimize memory layout for coalesced GPU access
- [ ] Add batch processing support

#### Testing
- [ ] Add unit tests for each dequantization kernel
- [ ] Add integration tests for multiple model sizes (1B, 3B, 8B)
- [ ] Add performance benchmarks (tokens/second)
- [ ] Add numerical accuracy tests against reference implementation
