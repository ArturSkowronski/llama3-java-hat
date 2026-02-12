#!/bin/bash
set -euo pipefail

# Download Llama 3.2 1B Instruct FP16 GGUF for integration tests
# Source: bartowski/Llama-3.2-1B-Instruct-GGUF on HuggingFace
# Size: ~2.48 GB

MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf"
MODEL_FILE="Llama-3.2-1B-Instruct-f16.gguf"

if [ -f "$MODEL_FILE" ]; then
    echo "Model $MODEL_FILE already exists. Skipping download."
else
    echo "Downloading Llama 3.2 1B Instruct FP16 (~2.48 GB)..."
    curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"
    echo "Download successful."
fi

export LLAMA_FP16_PATH="$(pwd)/$MODEL_FILE"
echo "LLAMA_FP16_PATH is set to $LLAMA_FP16_PATH"
