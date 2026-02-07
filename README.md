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
