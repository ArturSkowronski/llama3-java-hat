package com.arturskowronski.llama3babylon.hat;

import hat.Accelerator;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.types.F16;
import com.arturskowronski.llama3babylon.hat.kernels.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/**
 * End-to-end inference pipeline for Llama 3.2 1B Instruct.
 *
 * Pipeline: embedding lookup → 16 transformer layers → final RMSNorm → classifier → logits
 */
public class LlamaInference {

    private final LlamaModel model;
    private final Tokenizer tokenizer;
    private final ChatFormat chatFormat;
    private final TransformerBlock[] layers;
    private final F32Array[] kCaches;
    private final F32Array[] vCaches;

    private final WeightStorageMode weightMode;
    private final Object tokenEmbedding;   // F16Array or F32Array depending on mode
    private final F32Array outputNormWeight;
    private final Object outputWeight;     // F16Array or F32Array depending on mode

    private final IRMSNorm rmsNorm;
    private final IGEMV gemv;

    private final F32Array x;
    private final F32Array logits;

    public LlamaInference(Path ggufPath) throws IOException {
        this(ggufPath, new PlainJavaKernelFactory());
    }

    public LlamaInference(Path ggufPath, IKernelFactory factory) throws IOException {
        this(ggufPath, factory, BackendType.JAVA_SEQ);
    }

    public LlamaInference(Path ggufPath, IKernelFactory factory, BackendType backendType) throws IOException {
        this(ggufPath, factory, backendType, WeightStorageMode.F16);
    }

    /**
     * Creates a LlamaInference instance with full configuration.
     *
     * @param ggufPath path to GGUF model file
     * @param factory kernel factory for creating kernel implementations
     * @param backendType HAT backend to use for acceleration
     * @param weightMode how to store F16 weight tensors in memory
     */
    public LlamaInference(Path ggufPath, IKernelFactory factory, BackendType backendType,
                           WeightStorageMode weightMode) throws IOException {
        this.model = new LlamaModel(ggufPath, backendType);
        this.weightMode = weightMode;
        Accelerator acc = model.getAccelerator();

        // Load global weights
        // Llama 3.2 1B uses tied embeddings: output classifier shares token_embd.weight
        this.tokenEmbedding = mapProjectionWeight("token_embd.weight");
        this.outputNormWeight = model.mapTensor("output_norm.weight");
        this.outputWeight = model.hasTensor("output.weight")
                ? mapProjectionWeight("output.weight")
                : tokenEmbedding;

        // Initialize kernels using factory
        this.rmsNorm = factory.createRMSNorm(acc);
        this.gemv = factory.createGEMV(acc);

        // Allocate working buffers
        this.x = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.logits = F32Array.create(acc, LlamaModel.VOCAB_SIZE);

        // Allocate KV caches (one pair per layer)
        int kvDim = LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM;
        this.kCaches = new F32Array[LlamaModel.NUM_LAYERS];
        this.vCaches = new F32Array[LlamaModel.NUM_LAYERS];
        for (int l = 0; l < LlamaModel.NUM_LAYERS; l++) {
            kCaches[l] = F32Array.create(acc, LlamaModel.MAX_SEQ_LEN * kvDim);
            vCaches[l] = F32Array.create(acc, LlamaModel.MAX_SEQ_LEN * kvDim);
        }

        // Create transformer blocks
        this.layers = new TransformerBlock[LlamaModel.NUM_LAYERS];
        for (int l = 0; l < LlamaModel.NUM_LAYERS; l++) {
            layers[l] = new TransformerBlock(model, l, factory, weightMode);
        }

        // Initialize tokenizer and chat format from GGUF metadata
        this.tokenizer = Tokenizer.fromGGUFMetadata(model.getMetadata().metadata());
        this.chatFormat = new ChatFormat(tokenizer);
    }

    private Object mapProjectionWeight(String tensorName) throws IOException {
        return switch (weightMode) {
            case F16 -> model.mapTensorF16(tensorName);
            case F32 -> model.mapTensor(tensorName);
        };
    }

    /**
     * Forward pass for a single token at a given position.
     *
     * @param token input token ID
     * @param pos position in the sequence
     * @return logits array [VOCAB_SIZE]
     */
    public float[] forward(int token, int pos) {
        int hiddenSize = LlamaModel.HIDDEN_SIZE;
        int vocabSize = LlamaModel.VOCAB_SIZE;

        // 1. Embedding lookup
        int offset = token * hiddenSize;
        switch (tokenEmbedding) {
            case F16Array f16 -> {
                for (int i = 0; i < hiddenSize; i++) {
                    x.array(i, F16.f16ToFloat(f16.array(offset + i)));
                }
            }
            case F32Array f32 -> {
                for (int i = 0; i < hiddenSize; i++) {
                    x.array(i, f32.array(offset + i));
                }
            }
            default -> throw new IllegalStateException("Unexpected embedding type: " + tokenEmbedding.getClass());
        }

        // 2. Transformer layers
        for (int l = 0; l < LlamaModel.NUM_LAYERS; l++) {
            layers[l].forward(x, pos, kCaches[l], vCaches[l]);
        }

        // 3. Final RMSNorm
        rmsNorm.apply(x, outputNormWeight, hiddenSize);

        // 4. Classifier GEMV (outputWeight @ x → logits)
        switch (outputWeight) {
            case F16Array f16 -> gemv.apply(f16, x, logits, vocabSize, hiddenSize);
            case F32Array f32 -> gemv.apply(f32, x, logits, vocabSize, hiddenSize);
            default -> throw new IllegalStateException("Unexpected output weight type: " + outputWeight.getClass());
        }

        // 5. Copy to plain float[]
        float[] result = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            result[i] = logits.array(i);
        }
        return result;
    }

    /**
     * Generate tokens from a prompt using greedy decoding.
     *
     * @param promptTokens input token IDs
     * @param maxNewTokens maximum number of tokens to generate
     * @return generated token IDs (excluding prompt)
     */
    public int[] generate(int[] promptTokens, int maxNewTokens) {
        return generate(promptTokens, maxNewTokens, Set.of(128001)); // EOS_TOKEN
    }

    /**
     * Generate tokens with custom stop token set.
     */
    public int[] generate(int[] promptTokens, int maxNewTokens, Set<Integer> stopTokens) {
        int[] result = new int[maxNewTokens];
        int generated = 0;
        boolean isCI = System.getenv("CI") != null;

        // Prefill: process all prompt tokens
        float[] lastLogits = null;
        for (int i = 0; i < promptTokens.length; i++) {
            lastLogits = forward(promptTokens[i], i);
            if (isCI) {
                System.out.print("p");
                System.out.flush();
            }
        }

        // First generated token from last prefill logits
        int nextToken = argmax(lastLogits);
        result[0] = nextToken;
        generated = 1;

        // Auto-regressive generation
        while (generated < maxNewTokens && !stopTokens.contains(nextToken)) {
            lastLogits = forward(nextToken, promptTokens.length + generated - 1);
            nextToken = argmax(lastLogits);
            result[generated] = nextToken;
            generated++;

            if (isCI) {
                System.out.print(".");
                System.out.flush();
            }
        }
        if (isCI) {
            System.out.println(); // newline after progress
        }

        return Arrays.copyOf(result, generated);
    }

    /**
     * Generate a response to a user prompt using the Instruct chat format.
     *
     * @param systemPrompt system instructions
     * @param userPrompt the user's message
     * @param maxNewTokens maximum tokens to generate
     * @return decoded text response
     */
    public String chat(String systemPrompt, String userPrompt, int maxNewTokens) {
        List<ChatFormat.Message> dialog = new ArrayList<>();
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            dialog.add(new ChatFormat.Message(ChatFormat.Role.SYSTEM, systemPrompt));
        }
        dialog.add(new ChatFormat.Message(ChatFormat.Role.USER, userPrompt));

        List<Integer> promptTokens = chatFormat.encodeDialogPrompt(dialog);
        int[] promptArray = promptTokens.stream().mapToInt(Integer::intValue).toArray();

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        int[] generatedIds = generate(promptArray, maxNewTokens, stopTokens);

        // Decode generated tokens, excluding stop tokens
        List<Integer> tokenList = new ArrayList<>();
        for (int id : generatedIds) {
            if (stopTokens.contains(id)) break;
            tokenList.add(id);
        }
        return tokenizer.decode(tokenList);
    }

    /**
     * Returns the index of the maximum value in the array.
     */
    public static int argmax(float[] values) {
        int maxIdx = 0;
        float maxVal = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > maxVal) {
                maxVal = values[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
