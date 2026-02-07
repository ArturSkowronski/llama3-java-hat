package com.arturskowronski.llama3babylon.hat;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Minimal LlamaModel skeleton for Llama 3.2 1B Instruct (FP16).
 * 
 * This class is intentionally specialized and does NOT support:
 * - Quantized models (Q2_K, Q4_K, etc.)
 * - Other Llama variants (7B, 13B, 70B)
 * - Dynamic architecture detection
 */
public class LlamaModel {

    // Llama 3.2 1B Instruct architecture constants (hardcoded)
    public static final int HIDDEN_SIZE = 2048;
    public static final int INTERMEDIATE_SIZE = 8192;
    public static final int NUM_LAYERS = 16;
    public static final int NUM_HEADS = 32;
    public static final int NUM_KV_HEADS = 8;
    public static final int HEAD_DIM = HIDDEN_SIZE / NUM_HEADS; // 64
    public static final int VOCAB_SIZE = 128256;

    private final GGUFReader.GGUFMetadata metadata;
    private final Path modelPath;

    public LlamaModel(Path ggufPath) throws IOException {
        this.modelPath = ggufPath;
        this.metadata = GGUFReader.readMetadata(ggufPath);
        validateModel();
    }

    private void validateModel() {
        Map<String, Object> meta = metadata.metadata();
        
        String arch = (String) meta.get("general.architecture");
        if (arch == null || !arch.equals("llama")) {
            throw new IllegalArgumentException("Expected 'llama' architecture, got: " + arch);
        }

        // Verify tensor count is reasonable for a Llama model
        if (metadata.tensorCount() < 100) {
            throw new IllegalArgumentException("Too few tensors for Llama model: " + metadata.tensorCount());
        }
    }

    public GGUFReader.GGUFMetadata getMetadata() {
        return metadata;
    }

    public Path getModelPath() {
        return modelPath;
    }

    public int getHiddenSize() {
        return HIDDEN_SIZE;
    }

    public int getNumLayers() {
        return NUM_LAYERS;
    }

    public int getNumHeads() {
        return NUM_HEADS;
    }

    public int getNumKvHeads() {
        return NUM_KV_HEADS;
    }

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.out.println("Usage: LlamaModel <path_to_gguf>");
            return;
        }

        Path path = Path.of(args[0]);
        LlamaModel model = new LlamaModel(path);

        System.out.println("=== Llama 3.2 1B Instruct Model ===");
        System.out.println("Model Path: " + model.getModelPath());
        System.out.println("GGUF Version: " + model.getMetadata().version());
        System.out.println("Tensor Count: " + model.getMetadata().tensorCount());
        System.out.println();
        System.out.println("Architecture Constants:");
        System.out.println("  Hidden Size: " + HIDDEN_SIZE);
        System.out.println("  Intermediate Size: " + INTERMEDIATE_SIZE);
        System.out.println("  Num Layers: " + NUM_LAYERS);
        System.out.println("  Num Heads: " + NUM_HEADS);
        System.out.println("  Num KV Heads: " + NUM_KV_HEADS);
        System.out.println("  Head Dim: " + HEAD_DIM);
        System.out.println("  Vocab Size: " + VOCAB_SIZE);
    }
}
