package com.arturskowronski.llama3babylon.hat;

import hat.Accelerator;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import com.arturskowronski.llama3babylon.hat.kernels.*;

import java.io.IOException;

/**
 * Orchestrates a single transformer block for Llama 3.2 1B Instruct.
 *
 * Flow:
 * 1. RMSNorm (attn_norm)
 * 2. QKV Projection (GEMV x 3)
 * 3. RoPE (Q and K)
 * 4. Multi-head Attention (GQA)
 * 5. Output Projection (GEMV)
 * 6. Residual Add (x = x + attn_out)
 * 7. RMSNorm (ffn_norm)
 * 8. Feed-Forward (SwiGLU: GEMV x 3 + SiLU)
 * 9. Residual Add (x = x + ffn_out)
 */
public class TransformerBlock {

    // Kernels
    private final IRMSNorm rmsNorm;
    private final IGEMV gemv;
    private final IRoPE rope;
    private final IAttention attention;
    private final ISoftmax softmax;
    private final ISiLU silu;

    private final WeightStorageMode weightMode;

    // Weights (mapped from model) — either F16Array or F32Array depending on mode
    private final F32Array attnNormWeight;  // F32 in GGUF — norm weights always F32
    private final Object wq;
    private final Object wk;
    private final Object wv;
    private final Object wo;

    private final F32Array ffnNormWeight;   // F32 in GGUF — norm weights always F32
    private final Object w1;
    private final Object w2;
    private final Object w3;

    // Buffers for intermediate results
    private final F32Array q;
    private final F32Array k;
    private final F32Array v;
    private final F32Array attnOut;
    private final F32Array qHead;
    private final F32Array keyHeadCache;
    private final F32Array valueHeadCache;
    private final F32Array attnScores;
    private final F32Array attnHeadOut;
    private final F32Array ffn1Out;
    private final F32Array ffn3Out;
    private final F32Array ffnOut;
    private final F32Array residual;

    public TransformerBlock(LlamaModel model, int layerIdx, IKernelFactory factory) throws IOException {
        this(model, layerIdx, factory, WeightStorageMode.F16);
    }

    public TransformerBlock(LlamaModel model, int layerIdx, IKernelFactory factory,
                            WeightStorageMode weightMode) throws IOException {
        Accelerator acc = model.getAccelerator();
        this.weightMode = weightMode;

        // Initialize Kernels using factory
        this.rmsNorm = factory.createRMSNorm(acc);
        this.gemv = factory.createGEMV(acc);
        this.rope = factory.createRoPE(acc);
        this.attention = factory.createAttention(acc);
        this.softmax = factory.createSoftmax(acc);
        this.silu = factory.createSiLU(acc);

        // Map Weights (GGUF standard naming: blk.{N}.*)
        // Norm weights are F32 in GGUF; projection/FFN weights are F16 on disk
        String prefix = "blk." + layerIdx + ".";
        this.attnNormWeight = model.mapTensor(prefix + "attn_norm.weight");
        this.wq = mapProjectionWeight(model, prefix + "attn_q.weight");
        this.wk = mapProjectionWeight(model, prefix + "attn_k.weight");
        this.wv = mapProjectionWeight(model, prefix + "attn_v.weight");
        this.wo = mapProjectionWeight(model, prefix + "attn_output.weight");

        this.ffnNormWeight = model.mapTensor(prefix + "ffn_norm.weight");
        this.w1 = mapProjectionWeight(model, prefix + "ffn_gate.weight");
        this.w2 = mapProjectionWeight(model, prefix + "ffn_down.weight");
        this.w3 = mapProjectionWeight(model, prefix + "ffn_up.weight");

        // Pre-allocate Intermediate Buffers
        this.q = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.k = F32Array.create(acc, LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM);
        this.v = F32Array.create(acc, LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM);
        this.attnOut = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.qHead = F32Array.create(acc, LlamaModel.HEAD_DIM);
        this.keyHeadCache = F32Array.create(acc, LlamaModel.MAX_SEQ_LEN * LlamaModel.HEAD_DIM);
        this.valueHeadCache = F32Array.create(acc, LlamaModel.MAX_SEQ_LEN * LlamaModel.HEAD_DIM);
        this.attnScores = F32Array.create(acc, LlamaModel.MAX_SEQ_LEN);
        this.attnHeadOut = F32Array.create(acc, LlamaModel.HEAD_DIM);
        this.ffn1Out = F32Array.create(acc, LlamaModel.INTERMEDIATE_SIZE);
        this.ffn3Out = F32Array.create(acc, LlamaModel.INTERMEDIATE_SIZE);
        this.ffnOut = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.residual = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
    }

    private Object mapProjectionWeight(LlamaModel model, String tensorName) throws IOException {
        return switch (weightMode) {
            case F16 -> model.mapTensorF16(tensorName);
            case F32 -> model.mapTensor(tensorName);
        };
    }

    /**
     * Executes the transformer block for a single token.
     *
     * @param x input hidden state [HIDDEN_SIZE] (modified in-place by residual adds)
     * @param pos current token position
     * @param kCache Key Cache [MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM]
     * @param vCache Value Cache [MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM]
     */
    public void forward(F32Array x, int pos, F32Array kCache, F32Array vCache) {
        int hiddenSize = LlamaModel.HIDDEN_SIZE;
        int intermediateSize = LlamaModel.INTERMEDIATE_SIZE;
        int numHeads = LlamaModel.NUM_HEADS;
        int numKvHeads = LlamaModel.NUM_KV_HEADS;
        int headDim = LlamaModel.HEAD_DIM;
        float ropeTheta = 500000.0f; // Llama 3 default

        // Save residual for Step 6
        copy(x, residual, hiddenSize);

        // 1. RMSNorm (attn_norm)
        rmsNorm.apply(x, attnNormWeight, hiddenSize);

        // 2. QKV Projection
        gemvApply(wq, x, q, hiddenSize, hiddenSize);
        gemvApply(wk, x, k, numKvHeads * headDim, hiddenSize);
        gemvApply(wv, x, v, numKvHeads * headDim, hiddenSize);

        // 3. RoPE
        rope.apply(q, pos, numHeads, headDim, ropeTheta);
        rope.apply(k, pos, numKvHeads, headDim, ropeTheta);

        // 4. Update KV cache and compute multi-head attention via selected kernels.
        int kvDim = numKvHeads * headDim;
        int kvMul = numHeads / numKvHeads;

        // Store k, v into KV caches at current position
        int cacheOffset = pos * kvDim;
        for (int i = 0; i < kvDim; i++) {
            kCache.array(cacheOffset + i, k.array(i));
            vCache.array(cacheOffset + i, v.array(i));
        }

        int seqLen = pos + 1;
        for (int h = 0; h < numHeads; h++) {
            int kvHead = h / kvMul;
            int qOffset = h * headDim;
            int kvHeadOffset = kvHead * headDim;

            for (int d = 0; d < headDim; d++) {
                qHead.array(d, q.array(qOffset + d));
            }

            // Gather one KV head across sequence into contiguous scratch buffers.
            for (int t = 0; t < seqLen; t++) {
                int kvCacheOffset = t * kvDim + kvHeadOffset;
                int headOffset = t * headDim;
                for (int d = 0; d < headDim; d++) {
                    keyHeadCache.array(headOffset + d, kCache.array(kvCacheOffset + d));
                    valueHeadCache.array(headOffset + d, vCache.array(kvCacheOffset + d));
                }
            }

            attention.computeScores(qHead, keyHeadCache, attnScores, seqLen, headDim);
            softmax.apply(attnScores, seqLen);
            attention.computeValues(attnScores, valueHeadCache, attnHeadOut, seqLen, headDim);

            int attnOffset = h * headDim;
            for (int d = 0; d < headDim; d++) {
                attnOut.array(attnOffset + d, attnHeadOut.array(d));
            }
        }

        // 5. Output Projection
        gemvApply(wo, attnOut, x, hiddenSize, hiddenSize);

        // 6. Residual Add
        add(x, residual, hiddenSize);

        // Save residual for Step 9
        copy(x, residual, hiddenSize);

        // 7. RMSNorm (ffn_norm)
        rmsNorm.apply(x, ffnNormWeight, hiddenSize);

        // 8. Feed-Forward (SwiGLU)
        gemvApply(w1, x, ffn1Out, intermediateSize, hiddenSize);
        gemvApply(w3, x, ffn3Out, intermediateSize, hiddenSize);
        silu.apply(ffn1Out, intermediateSize);
        elementWiseMul(ffn1Out, ffn3Out, intermediateSize);
        gemvApply(w2, ffn1Out, ffnOut, hiddenSize, intermediateSize);

        // 9. Residual Add (use saved residual, not norm'd x)
        for (int i = 0; i < hiddenSize; i++) {
            x.array(i, residual.array(i) + ffnOut.array(i));
        }
    }

    private void gemvApply(Object weight, F32Array input, F32Array output, int rows, int cols) {
        switch (weight) {
            case F16Array f16 -> gemv.apply(f16, input, output, rows, cols);
            case F32Array f32 -> gemv.apply(f32, input, output, rows, cols);
            default -> throw new IllegalStateException("Unexpected weight type: " + weight.getClass());
        }
    }

    private void copy(F32Array src, F32Array dst, int size) {
        for (int i = 0; i < size; i++) {
            dst.array(i, src.array(i));
        }
    }

    private void add(F32Array a, F32Array b, int size) {
        for (int i = 0; i < size; i++) {
            a.array(i, a.array(i) + b.array(i));
        }
    }

    private void elementWiseMul(F32Array a, F32Array b, int size) {
        for (int i = 0; i < size; i++) {
            a.array(i, a.array(i) * b.array(i));
        }
    }
}
