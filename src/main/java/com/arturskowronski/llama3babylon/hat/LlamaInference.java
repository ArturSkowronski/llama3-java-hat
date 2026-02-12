package com.arturskowronski.llama3babylon.hat;

import hat.Accelerator;
import hat.buffer.F32Array;
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

    // Global weights
    private final F32Array tokenEmbedding;
    private final F32Array outputNormWeight;
    private final F32Array outputWeight;

    // Kernels
    private final IRMSNorm rmsNorm;
    private final IGEMV gemv;

    // Working buffers
    private final F32Array x;
    private final F32Array logits;

    /**
     * Creates a LlamaInference instance with a plain Java kernel factory (backward compatible).
     */
    public LlamaInference(Path ggufPath) throws IOException {
        this(ggufPath, new PlainJavaKernelFactory());
    }

    /**
     * Creates a LlamaInference instance with a custom kernel factory.
     *
     * @param ggufPath path to GGUF model file
     * @param factory kernel factory for creating kernel implementations
     */
    public LlamaInference(Path ggufPath, IKernelFactory factory) throws IOException {
        this.model = new LlamaModel(ggufPath);
        Accelerator acc = model.getAccelerator();

        // Load global weights
        // Llama 3.2 1B uses tied embeddings: output classifier shares token_embd.weight
        this.tokenEmbedding = model.mapTensor("token_embd.weight");
        this.outputNormWeight = model.mapTensor("output_norm.weight");
        this.outputWeight = model.hasTensor("output.weight")
                ? model.mapTensor("output.weight")
                : tokenEmbedding;

        // Initialize kernels using factory
        this.rmsNorm = factory.createRMSNorm(acc);
        this.gemv = factory.createGEMV(acc);

        // Allocate working buffers
        this.x = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.logits = F32Array.create(acc, LlamaModel.VOCAB_SIZE);

        // Prime GEMV with classifier matrix (128256x2048 — largest in model)
        // This forces HAT backend to cache buffer bounds large enough for all later dispatches
        gemv.apply(outputWeight, x, logits, LlamaModel.VOCAB_SIZE, LlamaModel.HIDDEN_SIZE);

        // Allocate KV caches (one pair per layer)
        int kvDim = LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM;
        this.kCaches = new F32Array[LlamaModel.NUM_LAYERS];
        this.vCaches = new F32Array[LlamaModel.NUM_LAYERS];
        for (int l = 0; l < LlamaModel.NUM_LAYERS; l++) {
            kCaches[l] = F32Array.create(acc, LlamaModel.MAX_SEQ_LEN * kvDim);
            vCaches[l] = F32Array.create(acc, LlamaModel.MAX_SEQ_LEN * kvDim);
        }

        // Create transformer blocks (after GEMV priming)
        this.layers = new TransformerBlock[LlamaModel.NUM_LAYERS];
        for (int l = 0; l < LlamaModel.NUM_LAYERS; l++) {
            layers[l] = new TransformerBlock(model, l, factory);
        }

        // Initialize tokenizer and chat format from GGUF metadata
        this.tokenizer = Tokenizer.fromGGUFMetadata(model.getMetadata().metadata());
        this.chatFormat = new ChatFormat(tokenizer);
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

        // 1. Embedding lookup: copy row from embedding table
        int offset = token * hiddenSize;
        for (int i = 0; i < hiddenSize; i++) {
            x.array(i, tokenEmbedding.array(offset + i));
        }

        // 2. Transformer layers
        for (int l = 0; l < LlamaModel.NUM_LAYERS; l++) {
            layers[l].forward(x, pos, kCaches[l], vCaches[l]);
        }

        // 3. Final RMSNorm
        rmsNorm.apply(x, outputNormWeight, hiddenSize);

        // 4. Classifier GEMV (outputWeight @ x → logits)
        gemv.apply(outputWeight, x, logits, vocabSize, hiddenSize);

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

        // Prefill: process all prompt tokens
        float[] lastLogits = null;
        for (int i = 0; i < promptTokens.length; i++) {
            lastLogits = forward(promptTokens[i], i);
        }

        // First generated token from last prefill logits
        int nextToken = argmax(lastLogits);
        result[0] = nextToken;
        generated = 1;

        // Auto-regressive generation
        boolean isCI = System.getenv("CI") != null;
        while (generated < maxNewTokens && !stopTokens.contains(nextToken)) {
            lastLogits = forward(nextToken, promptTokens.length + generated - 1);
            nextToken = argmax(lastLogits);
            result[generated] = nextToken;
            generated++;

            // Print progress on CI to prevent GitHub Actions no-output timeout (~10 min)
            // Use stderr because Gradle buffers stdout until test completion
            if (isCI && generated % 4 == 0) {
                System.err.print(".");
                System.err.flush();
            }
        }
        if (isCI && generated > 0) {
            System.err.println(); // newline after dots
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

    public Tokenizer getTokenizer() { return tokenizer; }
    public ChatFormat getChatFormat() { return chatFormat; }

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
