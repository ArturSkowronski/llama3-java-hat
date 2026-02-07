package com.arturskowronski.llama3babylon.hat;

import hat.Accelerator;
import hat.buffer.F32Array;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
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
    private final Accelerator accelerator;
    private final Map<String, F32Array> tensors = new HashMap<>();

    public LlamaModel(Path ggufPath) throws IOException {
        this(ggufPath, true);
    }

    /**
     * Constructor with optional strict validation.
     * @param ggufPath path to GGUF file
     * @param strictValidation if true, validates tensor count >= 100 (for real models)
     */
    LlamaModel(Path ggufPath, boolean strictValidation) throws IOException {
        this.modelPath = ggufPath;
        this.metadata = GGUFReader.readMetadata(ggufPath);
        this.accelerator = new Accelerator(MethodHandles.lookup());
        validateModel(strictValidation);
    }

    private void validateModel(boolean strictValidation) {
        Map<String, Object> meta = metadata.metadata();
        
        String arch = (String) meta.get("general.architecture");
        if (arch == null || !arch.equals("llama")) {
            throw new IllegalArgumentException("Expected 'llama' architecture, got: " + arch);
        }

        // Verify tensor count is reasonable for a Llama model (skip for unit tests)
        if (strictValidation && metadata.tensorCount() < 100) {
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

    public Accelerator getAccelerator() {
        return accelerator;
    }

    /**
     * Maps a tensor from the GGUF file into a HAT F32Array buffer.
     * Supports F32 (type 0) and F16 (type 1) tensors only.
     * 
     * @param tensorName the name of the tensor to load
     * @return F32Array containing the dequantized tensor data
     * @throws IOException if tensor not found or unsupported type
     */
    public F32Array mapTensor(String tensorName) throws IOException {
        if (tensors.containsKey(tensorName)) {
            return tensors.get(tensorName);
        }

        GGUFReader.GGUFTensorInfo tensorInfo = metadata.tensors().stream()
                .filter(t -> t.name().equals(tensorName))
                .findFirst()
                .orElseThrow(() -> new IOException("Tensor not found: " + tensorName));

        int type = tensorInfo.type();
        if (type != 0 && type != 1) {
            throw new IOException("Unsupported tensor type: " + type + " for tensor: " + tensorName + 
                    ". Only F32 (0) and F16 (1) are supported.");
        }

        long elementCount = 1;
        for (long dim : tensorInfo.shape()) {
            elementCount *= dim;
        }

        F32Array buffer = F32Array.create(accelerator, (int) elementCount);
        long absoluteOffset = metadata.dataStartOffset() + tensorInfo.offset();

        try (FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ);
             Arena arena = Arena.ofConfined()) {
            
            long dataSize = tensorInfo.size();
            MemorySegment segment = channel.map(FileChannel.MapMode.READ_ONLY, absoluteOffset, dataSize, arena);

            if (type == 0) {
                // F32: direct copy
                for (int i = 0; i < elementCount; i++) {
                    buffer.array(i, segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, (long) i * 4));
                }
            } else {
                // F16: dequantize to F32
                for (int i = 0; i < elementCount; i++) {
                    short f16 = segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, (long) i * 2);
                    buffer.array(i, Float.float16ToFloat(f16));
                }
            }
        }

        tensors.put(tensorName, buffer);
        return buffer;
    }

    /**
     * Returns the tensor info for a given tensor name.
     */
    public GGUFReader.GGUFTensorInfo getTensorInfo(String tensorName) {
        return metadata.tensors().stream()
                .filter(t -> t.name().equals(tensorName))
                .findFirst()
                .orElse(null);
    }

    /**
     * Checks if a tensor exists in the model.
     */
    public boolean hasTensor(String tensorName) {
        return metadata.tensors().stream().anyMatch(t -> t.name().equals(tensorName));
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

        // Demo: try to map a tensor if available
        System.out.println();
        System.out.println("Tensor Mapping Demo:");
        for (var tensor : model.getMetadata().tensors()) {
            if (tensor.type() == 0 || tensor.type() == 1) {
                System.out.println("  Loading: " + tensor.name() + " (type=" + tensor.type() + ")");
                try {
                    F32Array arr = model.mapTensor(tensor.name());
                    System.out.println("    Loaded " + arr.length() + " floats");
                    if (arr.length() > 0) {
                        System.out.println("    First value: " + arr.array(0));
                    }
                    break; // Just demo one tensor
                } catch (IOException e) {
                    System.out.println("    Failed: " + e.getMessage());
                }
            }
        }
    }
}
