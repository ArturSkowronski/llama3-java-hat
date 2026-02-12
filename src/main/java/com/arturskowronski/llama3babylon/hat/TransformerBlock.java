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
    private final F32Array residual;

    // Plain Java array for per-head attention scores (avoids HAT buffer caching bug)
    private final float[] att;

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

        // Map Weights (GGUF standard naming: blk.{N}.*)
        String prefix = "blk." + layerIdx + ".";
        this.attnNormWeight = model.mapTensor(prefix + "attn_norm.weight");
        this.wq = model.mapTensor(prefix + "attn_q.weight");
        this.wk = model.mapTensor(prefix + "attn_k.weight");
        this.wv = model.mapTensor(prefix + "attn_v.weight");
        this.wo = model.mapTensor(prefix + "attn_output.weight");

        this.ffnNormWeight = model.mapTensor(prefix + "ffn_norm.weight");
        this.w1 = model.mapTensor(prefix + "ffn_gate.weight");
        this.w2 = model.mapTensor(prefix + "ffn_down.weight");
        this.w3 = model.mapTensor(prefix + "ffn_up.weight");

        // Pre-allocate Intermediate Buffers
        this.normOut = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.q = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.k = F32Array.create(acc, LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM);
        this.v = F32Array.create(acc, LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM);
        this.attnOut = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.ffn1Out = F32Array.create(acc, LlamaModel.INTERMEDIATE_SIZE);
        this.ffn3Out = F32Array.create(acc, LlamaModel.INTERMEDIATE_SIZE);
        this.ffnOut = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.residual = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.att = new float[LlamaModel.MAX_SEQ_LEN];
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
        gemv.apply(wq, x, q, hiddenSize, hiddenSize);
        gemv.apply(wk, x, k, numKvHeads * headDim, hiddenSize);
        gemv.apply(wv, x, v, numKvHeads * headDim, hiddenSize);

        // 3. RoPE
        rope.apply(q, pos, numHeads, headDim, ropeTheta);
        rope.apply(k, pos, numKvHeads, headDim, ropeTheta);

        // 4. Update KV Cache & Multi-head Attention (plain Java, no HAT dispatch)
        int kvDim = numKvHeads * headDim;
        int kvMul = numHeads / numKvHeads;

        // Store k, v into KV caches at current position
        int cacheOffset = pos * kvDim;
        for (int i = 0; i < kvDim; i++) {
            kCache.array(cacheOffset + i, k.array(i));
            vCache.array(cacheOffset + i, v.array(i));
        }

        float scale = 1.0f / (float) Math.sqrt(headDim);

        // GQA: for each query head, compute scaled dot-product attention
        for (int h = 0; h < numHeads; h++) {
            int kvHead = h / kvMul;
            int qOffset = h * headDim;

            // Compute attention scores against all cached keys (0..pos inclusive)
            for (int t = 0; t <= pos; t++) {
                float score = 0.0f;
                int kOffset = t * kvDim + kvHead * headDim;
                for (int d = 0; d < headDim; d++) {
                    score += q.array(qOffset + d) * kCache.array(kOffset + d);
                }
                att[t] = score * scale;
            }

            // Softmax over scores
            softmaxInPlace(att, pos + 1);

            // Weighted sum of cached values → write to attnOut
            int attnOffset = h * headDim;
            for (int d = 0; d < headDim; d++) {
                float sum = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    sum += att[t] * vCache.array(t * kvDim + kvHead * headDim + d);
                }
                attnOut.array(attnOffset + d, sum);
            }
        }

        // 5. Output Projection
        gemv.apply(wo, attnOut, x, hiddenSize, hiddenSize);
        
        // 6. Residual Add
        add(x, residual, hiddenSize);

        // Save residual for Step 9
        copy(x, residual, hiddenSize);

        // 7. RMSNorm (ffn_norm)
        rmsNorm.apply(x, ffnNormWeight, hiddenSize);

        // 8. Feed-Forward (SwiGLU)
        gemv.apply(w1, x, ffn1Out, intermediateSize, hiddenSize);
        gemv.apply(w3, x, ffn3Out, intermediateSize, hiddenSize);
        silu.apply(ffn1Out, intermediateSize);
        elementWiseMul(ffn1Out, ffn3Out, intermediateSize);
        gemv.apply(w2, ffn1Out, ffnOut, hiddenSize, intermediateSize);

        // 9. Residual Add (use saved residual, not norm'd x)
        for (int i = 0; i < hiddenSize; i++) {
            x.array(i, residual.array(i) + ffnOut.array(i));
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

    private void softmaxInPlace(float[] values, int size) {
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            if (values[i] > maxVal) maxVal = values[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            values[i] = (float) Math.exp(values[i] - maxVal);
            sum += values[i];
        }
        float invSum = 1.0f / sum;
        for (int i = 0; i < size; i++) {
            values[i] *= invSum;
        }
    }

    private void elementWiseMul(F32Array a, F32Array b, int size) {
        for (int i = 0; i < size; i++) {
            a.array(i, a.array(i) * b.array(i));
        }
    }
}
