package com.arturskowronski.llama3babylon.hat;

import com.arturskowronski.llama3babylon.hat.utils.MinimalGGUFGenerator;
import hat.buffer.F32Array;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class TransformerBlockTest {

    @TempDir
    Path tempDir;

    @Test
    public void testTransformerBlockInitialization() throws IOException {
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

        float[][] tensorData = new float[tensorNames.length][];
        tensorData[0] = new float[LlamaModel.HIDDEN_SIZE];
        tensorData[1] = new float[LlamaModel.HIDDEN_SIZE * LlamaModel.HIDDEN_SIZE];
        tensorData[2] = new float[LlamaModel.HIDDEN_SIZE * (LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM)];
        tensorData[3] = new float[LlamaModel.HIDDEN_SIZE * (LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM)];
        tensorData[4] = new float[LlamaModel.HIDDEN_SIZE * LlamaModel.HIDDEN_SIZE];
        tensorData[5] = new float[LlamaModel.HIDDEN_SIZE];
        tensorData[6] = new float[LlamaModel.INTERMEDIATE_SIZE * LlamaModel.HIDDEN_SIZE];
        tensorData[7] = new float[LlamaModel.HIDDEN_SIZE * LlamaModel.INTERMEDIATE_SIZE];
        tensorData[8] = new float[LlamaModel.INTERMEDIATE_SIZE * LlamaModel.HIDDEN_SIZE];

        MinimalGGUFGenerator.generateLlamaWithTensors(ggufPath, tensorNames, tensorData);

        LlamaModel model = new LlamaModel(ggufPath, false);
        TransformerBlock block = new TransformerBlock(model, 0);

        assertNotNull(block);
    }

    @Test
    public void testTransformerBlockForwardPass() throws IOException {
        Path ggufPath = tempDir.resolve("model_forward.gguf");

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
        tensorData[0] = createArray(LlamaModel.HIDDEN_SIZE, 1.0f);              // attn_norm
        tensorData[1] = createArray(LlamaModel.HIDDEN_SIZE * LlamaModel.HIDDEN_SIZE, 0.01f);  // wq
        tensorData[2] = createArray(LlamaModel.HIDDEN_SIZE * (LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM), 0.01f); // wk
        tensorData[3] = createArray(LlamaModel.HIDDEN_SIZE * (LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM), 0.01f); // wv
        tensorData[4] = createArray(LlamaModel.HIDDEN_SIZE * LlamaModel.HIDDEN_SIZE, 0.01f);  // wo
        tensorData[5] = createArray(LlamaModel.HIDDEN_SIZE, 1.0f);              // ffn_norm
        tensorData[6] = createArray(LlamaModel.INTERMEDIATE_SIZE * LlamaModel.HIDDEN_SIZE, 0.01f); // w1
        tensorData[7] = createArray(LlamaModel.HIDDEN_SIZE * LlamaModel.INTERMEDIATE_SIZE, 0.01f); // w2
        tensorData[8] = createArray(LlamaModel.INTERMEDIATE_SIZE * LlamaModel.HIDDEN_SIZE, 0.01f); // w3

        MinimalGGUFGenerator.generateLlamaWithTensors(ggufPath, tensorNames, tensorData);

        LlamaModel model = new LlamaModel(ggufPath, false);
        TransformerBlock block = new TransformerBlock(model, 0);

        // Create input and cache buffers
        F32Array x = F32Array.create(model.getAccelerator(), LlamaModel.HIDDEN_SIZE);
        F32Array kCache = F32Array.create(model.getAccelerator(), 2048 * LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM);
        F32Array vCache = F32Array.create(model.getAccelerator(), 2048 * LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM);

        // Initialize input
        for (int i = 0; i < LlamaModel.HIDDEN_SIZE; i++) {
            x.array(i, 0.1f);
        }

        // Run forward pass at position 0
        block.forward(x, 0, kCache, vCache);

        // Verify output is finite and non-zero
        float sum = 0.0f;
        boolean allFinite = true;
        for (int i = 0; i < LlamaModel.HIDDEN_SIZE; i++) {
            float val = x.array(i);
            if (Float.isNaN(val) || Float.isInfinite(val)) {
                allFinite = false;
                break;
            }
            sum += Math.abs(val);
        }

        assertTrue(allFinite, "Forward pass should produce finite values");
        assertTrue(sum > 0.0f, "Forward pass should produce non-zero output");
    }

    private float[] createArray(int size, float value) {
        float[] arr = new float[size];
        for (int i = 0; i < size; i++) {
            arr[i] = value;
        }
        return arr;
    }
}
