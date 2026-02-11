package com.arturskowronski.llama3babylon.hat;

import com.arturskowronski.llama3babylon.hat.utils.MinimalGGUFGenerator;
import hat.buffer.F32Array;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

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
}
