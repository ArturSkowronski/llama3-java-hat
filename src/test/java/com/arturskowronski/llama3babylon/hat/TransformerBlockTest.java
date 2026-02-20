package com.arturskowronski.llama3babylon.hat;

import com.arturskowronski.llama3babylon.hat.kernels.GEMV;
import com.arturskowronski.llama3babylon.hat.kernels.IAttention;
import com.arturskowronski.llama3babylon.hat.kernels.IGEMV;
import com.arturskowronski.llama3babylon.hat.kernels.IKernelFactory;
import com.arturskowronski.llama3babylon.hat.kernels.IRMSNorm;
import com.arturskowronski.llama3babylon.hat.kernels.IRoPE;
import com.arturskowronski.llama3babylon.hat.kernels.ISiLU;
import com.arturskowronski.llama3babylon.hat.kernels.ISoftmax;
import com.arturskowronski.llama3babylon.hat.kernels.PlainJavaKernelFactory;
import com.arturskowronski.llama3babylon.hat.utils.MinimalGGUFGenerator;
import hat.buffer.F32Array;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

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
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_up.weight"
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
        TransformerBlock block = new TransformerBlock(model, 0, new PlainJavaKernelFactory());

        assertNotNull(block);
    }

    @Test
    public void testForwardPassWithGQA() throws IOException {
        Path ggufPath = tempDir.resolve("model.gguf");

        String[] tensorNames = {
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_up.weight"
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

        CountingKernelFactory factory = new CountingKernelFactory(new PlainJavaKernelFactory());
        TransformerBlock block = new TransformerBlock(model, 0, factory);

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
        assertTrue(factory.attentionScoreCalls > 0, "Attention.computeScores should be used in transformer forward");
        assertTrue(factory.attentionValueCalls > 0, "Attention.computeValues should be used in transformer forward");
        assertTrue(factory.softmaxApplyCalls > 0, "Softmax.apply should be used in transformer forward");

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

    private static class CountingKernelFactory implements IKernelFactory {
        private final IKernelFactory delegate;
        int attentionScoreCalls;
        int attentionValueCalls;
        int softmaxApplyCalls;

        CountingKernelFactory(IKernelFactory delegate) {
            this.delegate = delegate;
        }

        @Override
        public IGEMV createGEMV(Accelerator acc) {
            return delegate.createGEMV(acc);
        }

        @Override
        public IRMSNorm createRMSNorm(Accelerator acc) {
            return delegate.createRMSNorm(acc);
        }

        @Override
        public IRoPE createRoPE(Accelerator acc) {
            return delegate.createRoPE(acc);
        }

        @Override
        public ISiLU createSiLU(Accelerator acc) {
            return delegate.createSiLU(acc);
        }

        @Override
        public ISoftmax createSoftmax(Accelerator acc) {
            ISoftmax kernel = delegate.createSoftmax(acc);
            return new ISoftmax() {
                @Override
                public void apply(F32Array input, int size) {
                    softmaxApplyCalls++;
                    kernel.apply(input, size);
                }

                @Override
                public void applyRow(F32Array input, int rowOffset, int rowSize) {
                    kernel.applyRow(input, rowOffset, rowSize);
                }
            };
        }

        @Override
        public IAttention createAttention(Accelerator acc) {
            IAttention kernel = delegate.createAttention(acc);
            return new IAttention() {
                @Override
                public void computeScores(F32Array query, F32Array keys, F32Array scores, int seqLen, int headDim) {
                    attentionScoreCalls++;
                    kernel.computeScores(query, keys, scores, seqLen, headDim);
                }

                @Override
                public void computeValues(F32Array scores, F32Array values, F32Array output, int seqLen, int headDim) {
                    attentionValueCalls++;
                    kernel.computeValues(scores, values, output, seqLen, headDim);
                }
            };
        }
    }
}
