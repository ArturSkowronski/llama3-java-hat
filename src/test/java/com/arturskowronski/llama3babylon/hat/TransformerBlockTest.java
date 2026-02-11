package com.arturskowronski.llama3babylon.hat;

import com.arturskowronski.llama3babylon.hat.kernels.GEMV;
import com.arturskowronski.llama3babylon.hat.utils.MinimalGGUFGenerator;
import hat.buffer.F32Array;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import hat.Accelerator;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TransformerBlockTest {

    @TempDir
    Path tempDir;

    @Test
    public void testTransformerBlockInitialization() throws IOException {
        Path ggufPath = tempDir.resolve("model.gguf");
        
        // Define all necessary tensors for a single layer
        String[] tensorNames = {
            "layers.0.attention_norm.weight",
            "layers.0.attention.wq.weight",
            "layers.0.attention.wk.weight",
            "layers.0.attention.wv.weight",
            "layers.0.attention.wo.weight",
            "layers.0.ffn_norm.weight",
            "layers.0.feed_forward.w1.weight",
            "layers.0.feed_forward.w2.weight",
            "layers.0.feed_forward.w3.weight"
        };
        
        float[][] tensorData = new float[tensorNames.length][];
        // Populate with dummy data of correct sizes for 1B model
        tensorData[0] = new float[LlamaModel.HIDDEN_SIZE]; // attn_norm
        tensorData[1] = new float[LlamaModel.HIDDEN_SIZE * LlamaModel.HIDDEN_SIZE]; // wq
        tensorData[2] = new float[LlamaModel.HIDDEN_SIZE * (LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM)]; // wk
        tensorData[3] = new float[LlamaModel.HIDDEN_SIZE * (LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM)]; // wv
        tensorData[4] = new float[LlamaModel.HIDDEN_SIZE * LlamaModel.HIDDEN_SIZE]; // wo
        tensorData[5] = new float[LlamaModel.HIDDEN_SIZE]; // ffn_norm
        tensorData[6] = new float[LlamaModel.INTERMEDIATE_SIZE * LlamaModel.HIDDEN_SIZE]; // w1
        tensorData[7] = new float[LlamaModel.HIDDEN_SIZE * LlamaModel.INTERMEDIATE_SIZE]; // w2
        tensorData[8] = new float[LlamaModel.INTERMEDIATE_SIZE * LlamaModel.HIDDEN_SIZE]; // w3

        MinimalGGUFGenerator.generateLlamaWithTensors(ggufPath, tensorNames, tensorData);

        LlamaModel model = new LlamaModel(ggufPath, false);
        TransformerBlock block = new TransformerBlock(model, 0);
        
        assertNotNull(block);
    }

    @Test
    public void testForwardPassWithGQA() throws IOException {
        Path ggufPath = tempDir.resolve("model.gguf");

        String[] tensorNames = {
            "layers.0.attention_norm.weight",
            "layers.0.attention.wq.weight",
            "layers.0.attention.wk.weight",
            "layers.0.attention.wv.weight",
            "layers.0.attention.wo.weight",
            "layers.0.ffn_norm.weight",
            "layers.0.feed_forward.w1.weight",
            "layers.0.feed_forward.w2.weight",
            "layers.0.feed_forward.w3.weight"
        };

        // Fill weights with small random values so output is non-trivial
        Random rng = new Random(42);
        float[][] tensorData = new float[tensorNames.length][];
        tensorData[0] = randomArray(rng, LlamaModel.HIDDEN_SIZE, 0.1f);
        tensorData[1] = randomArray(rng, LlamaModel.HIDDEN_SIZE * LlamaModel.HIDDEN_SIZE, 0.01f);
        tensorData[2] = randomArray(rng, LlamaModel.HIDDEN_SIZE * (LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM), 0.01f);
        tensorData[3] = randomArray(rng, LlamaModel.HIDDEN_SIZE * (LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM), 0.01f);
        tensorData[4] = randomArray(rng, LlamaModel.HIDDEN_SIZE * LlamaModel.HIDDEN_SIZE, 0.01f);
        tensorData[5] = randomArray(rng, LlamaModel.HIDDEN_SIZE, 0.1f);
        tensorData[6] = randomArray(rng, LlamaModel.INTERMEDIATE_SIZE * LlamaModel.HIDDEN_SIZE, 0.01f);
        tensorData[7] = randomArray(rng, LlamaModel.HIDDEN_SIZE * LlamaModel.INTERMEDIATE_SIZE, 0.01f);
        tensorData[8] = randomArray(rng, LlamaModel.INTERMEDIATE_SIZE * LlamaModel.HIDDEN_SIZE, 0.01f);

        MinimalGGUFGenerator.generateLlamaWithTensors(ggufPath, tensorNames, tensorData);

        LlamaModel model = new LlamaModel(ggufPath, false);
        Accelerator acc = model.getAccelerator();

        // Prime GEMV with the largest matrix size used in a TransformerBlock
        // (INTERMEDIATE_SIZE × HIDDEN_SIZE = 16M elements) — HAT bug workaround.
        // Must happen before any TransformerBlock creation or forward() calls.
        F32Array primingMatrix = F32Array.create(acc, LlamaModel.INTERMEDIATE_SIZE * LlamaModel.HIDDEN_SIZE);
        F32Array primingInput = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        F32Array primingOutput = F32Array.create(acc, LlamaModel.INTERMEDIATE_SIZE);
        new GEMV(acc).apply(primingMatrix, primingInput, primingOutput,
                LlamaModel.INTERMEDIATE_SIZE, LlamaModel.HIDDEN_SIZE);

        TransformerBlock block = new TransformerBlock(model, 0);

        F32Array x = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        for (int i = 0; i < LlamaModel.HIDDEN_SIZE; i++) {
            x.array(i, rng.nextFloat() * 0.1f);
        }

        int kvDim = LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM;
        F32Array kCache = F32Array.create(acc, LlamaModel.MAX_SEQ_LEN * kvDim);
        F32Array vCache = F32Array.create(acc, LlamaModel.MAX_SEQ_LEN * kvDim);

        // Forward pass at position 0 — completes without exception
        block.forward(x, 0, kCache, vCache);

        // Output should be finite and non-trivial
        boolean allZero = true;
        for (int i = 0; i < LlamaModel.HIDDEN_SIZE; i++) {
            float val = x.array(i);
            assertFalse(Float.isNaN(val), "Output contains NaN at index " + i);
            assertFalse(Float.isInfinite(val), "Output contains Inf at index " + i);
            if (val != 0.0f) allZero = false;
        }
        assertFalse(allZero, "Output should not be all zeros");

        // Note: KV cache population and output-differs-from-input are validated
        // by the integration test with a real model. The HAT sequential backend
        // does not reliably sync intermediate buffer results when multiple
        // kernel types (RMSNorm, GEMV, RoPE, SiLU) are dispatched in sequence.
    }

    private static float[] randomArray(Random rng, int size, float scale) {
        float[] arr = new float[size];
        for (int i = 0; i < size; i++) {
            arr[i] = (rng.nextFloat() - 0.5f) * 2 * scale;
        }
        return arr;
    }
}
