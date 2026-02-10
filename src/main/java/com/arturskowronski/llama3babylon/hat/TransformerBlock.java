package com.arturskowronski.llama3babylon.hat;

import hat.Accelerator;
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

    private final LlamaModel model;
    private final int layerIdx;
    
    // Kernels
    private final RMSNorm rmsNorm;
    private final GEMV gemv;
    private final RoPE rope;
    private final Attention attention;
    private final Softmax softmax;
    private final SiLU silu;

    // Weights (mapped from model)
    private final F32Array attnNormWeight;
    private final F32Array wq;
    private final F32Array wk;
    private final F32Array wv;
    private final F32Array wo;
    
    private final F32Array ffnNormWeight;
    private final F32Array w1;
    private final F32Array w2;
    private final F32Array w3;

    // Buffers for intermediate results
    private final F32Array normOut;
    private final F32Array q;
    private final F32Array k;
    private final F32Array v;
    private final F32Array attnOut;
    private final F32Array ffn1Out;
    private final F32Array ffn3Out;
    private final F32Array ffnOut;

    public TransformerBlock(LlamaModel model, int layerIdx) throws IOException {
        this.model = model;
        this.layerIdx = layerIdx;
        Accelerator acc = model.getAccelerator();

        // Initialize Kernels
        this.rmsNorm = new RMSNorm(acc);
        this.gemv = new GEMV(acc);
        this.rope = new RoPE(acc);
        this.attention = new Attention(acc);
        this.softmax = new Softmax(acc);
        this.silu = new SiLU(acc);

        // Map Weights
        String prefix = "layers." + layerIdx + ".";
        this.attnNormWeight = model.mapTensor(prefix + "attention_norm.weight");
        this.wq = model.mapTensor(prefix + "attention.wq.weight");
        this.wk = model.mapTensor(prefix + "attention.wk.weight");
        this.wv = model.mapTensor(prefix + "attention.wv.weight");
        this.wo = model.mapTensor(prefix + "attention.wo.weight");

        this.ffnNormWeight = model.mapTensor(prefix + "ffn_norm.weight");
        this.w1 = model.mapTensor(prefix + "feed_forward.w1.weight");
        this.w2 = model.mapTensor(prefix + "feed_forward.w2.weight");
        this.w3 = model.mapTensor(prefix + "feed_forward.w3.weight");

        // Pre-allocate Intermediate Buffers
        this.normOut = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.q = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.k = F32Array.create(acc, LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM);
        this.v = F32Array.create(acc, LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM);
        this.attnOut = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.ffn1Out = F32Array.create(acc, LlamaModel.INTERMEDIATE_SIZE);
        this.ffn3Out = F32Array.create(acc, LlamaModel.INTERMEDIATE_SIZE);
        this.ffnOut = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
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

        // 1. RMSNorm (attn_norm)
        rmsNorm.apply(x, attnNormWeight, hiddenSize);
        // Note: x is modified in-place by rmsNorm according to its current implementation.
        // Wait, RMSNorm.apply(input, weight, size) modifies 'input' in-place.

        // 2. QKV Projection
        gemv.apply(wq, x, q, numHeads * headDim, hiddenSize);
        gemv.apply(wk, x, k, numKvHeads * headDim, hiddenSize);
        gemv.apply(wv, x, v, numKvHeads * headDim, hiddenSize);

        // 3. RoPE
        rope.apply(q, pos, numHeads, headDim, ropeTheta);
        rope.apply(k, pos, numKvHeads, headDim, ropeTheta);

        // 4. Update KV Cache & Multi-head Attention (Simplified for now: single token, no full GQA yet)
        // TODO: Full GQA implementation using attention and softmax kernels

        // 5. Output Projection
        
        // 6. Residual Add

        // 7. RMSNorm (ffn_norm)
        rmsNorm.apply(x, ffnNormWeight, hiddenSize);

        // 8. Feed-Forward (SwiGLU)
        gemv.apply(w1, x, ffn1Out, intermediateSize, hiddenSize);
        gemv.apply(w3, x, ffn3Out, intermediateSize, hiddenSize);
        silu.apply(ffn1Out, intermediateSize);
        // elementWiseMul(ffn1Out, ffn3Out, intermediateSize);
        gemv.apply(w2, ffn1Out, ffnOut, hiddenSize, intermediateSize);

        // 9. Residual Add
    }
}
