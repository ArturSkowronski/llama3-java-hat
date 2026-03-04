#!/usr/bin/env bash

# Script to download Tiny Llama model for integration tests
# Model: tinyllama-1.1b-chat-v1.0.Q2_K.gguf
# Source: HuggingFace (TheBloke)

MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
MODEL_FILE="tinyllama-1.1b-chat-v1.0.Q2_K.gguf"

if [ -f "$MODEL_FILE" ]; then
  echo "Model $MODEL_FILE already exists. Skipping download."
else
  echo "Downloading TinyLlama model..."
  if curl -L -o "$MODEL_FILE" "$MODEL_URL"; then
    echo "Download successful."
  else
    echo "Download failed."
    exit 1
  fi
fi

# Set environment variable for current session (if sourced)
TINY_LLAMA_PATH="$(pwd)/$MODEL_FILE"
export TINY_LLAMA_PATH
echo "TINY_LLAMA_PATH is set to $TINY_LLAMA_PATH"
