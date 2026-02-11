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
 * 4. Update KV Cache
 * 5. Multi-head Grouped Query Attention (GQA) — offset-based, following mukel/llama3.java
 * 6. Output Projection (GEMV)
 * 7. Residual Add (x = x + attn_out)
 * 8. RMSNorm (ffn_norm)
 * 9. Feed-Forward (SwiGLU: GEMV x 3 + SiLU)
 * 10. Residual Add (x = x + ffn_out)
 */
public class TransformerBlock {

    private final LlamaModel model;
    private final int layerIdx;

    // Kernels
    private final RMSNorm rmsNorm;
    private final GEMV gemv;
    private final RoPE rope;
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
    private final F32Array xb;       // RMSNorm output / attention weighted sum
    private final F32Array xb2;      // output projection result
    private final F32Array q;
    private final F32Array k;
    private final F32Array v;
    private final F32Array att;      // attention scores [NUM_HEADS * MAX_SEQ_LEN]
    private final F32Array hb;       // FFN hidden buffer 1
    private final F32Array hb2;      // FFN hidden buffer 2

    private static final int MAX_SEQ_LEN = 2048;

    public TransformerBlock(LlamaModel model, int layerIdx) throws IOException {
        this.model = model;
        this.layerIdx = layerIdx;
        Accelerator acc = model.getAccelerator();

        // Initialize Kernels
        this.rmsNorm = new RMSNorm(acc);
        this.gemv = new GEMV(acc);
        this.rope = new RoPE(acc);
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

        // Pre-allocate Intermediate Buffers (following mukel/llama3.java naming)
        this.xb = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.xb2 = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.q = F32Array.create(acc, LlamaModel.HIDDEN_SIZE);
        this.k = F32Array.create(acc, LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM);
        this.v = F32Array.create(acc, LlamaModel.NUM_KV_HEADS * LlamaModel.HEAD_DIM);
        this.att = F32Array.create(acc, LlamaModel.NUM_HEADS * MAX_SEQ_LEN);
        this.hb = F32Array.create(acc, LlamaModel.INTERMEDIATE_SIZE);
        this.hb2 = F32Array.create(acc, LlamaModel.INTERMEDIATE_SIZE);

        // HAT workaround: The Java Sequential Backend caches buffer bounds from the
        // first GEMV kernel dispatch. We must prime the kernel with the largest matrix
        // (w1/w3: INTERMEDIATE_SIZE x HIDDEN_SIZE) so subsequent calls with smaller
        // matrices succeed. See: https://github.com/openjdk/babylon/issues/XXX
        gemv.apply(w1, xb, hb, LlamaModel.INTERMEDIATE_SIZE, LlamaModel.HIDDEN_SIZE);
    }

    /**
     * Executes the transformer block for a single token.
     *
     * @param x input hidden state [HIDDEN_SIZE] (modified in-place via residual adds)
     * @param pos current token position
     * @param kCache Key Cache [MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM]
     * @param vCache Value Cache [MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM]
     */
    public void forward(F32Array x, int pos, F32Array kCache, F32Array vCache) {
        int dim = LlamaModel.HIDDEN_SIZE;
        int hiddenDim = LlamaModel.INTERMEDIATE_SIZE;
        int numHeads = LlamaModel.NUM_HEADS;
        int numKvHeads = LlamaModel.NUM_KV_HEADS;
        int headDim = LlamaModel.HEAD_DIM;
        int kvDim = numKvHeads * headDim;
        int kvMul = numHeads / numKvHeads; // GQA multiplier (4 for Llama 3.2 1B)
        float sqrtHeadSize = (float) Math.sqrt(headDim);
        float ropeTheta = 500000.0f;

        // 1. RMSNorm (attn_norm) -> xb, preserving x for residual
        rmsnorm(xb, x, attnNormWeight, dim);

        // 2. QKV Projection
        gemv.apply(wq, xb, q, dim, dim);
        gemv.apply(wk, xb, k, kvDim, dim);
        gemv.apply(wv, xb, v, kvDim, dim);

        // 3. RoPE
        rope.apply(q, pos, numHeads, headDim, ropeTheta);
        rope.apply(k, pos, numKvHeads, headDim, ropeTheta);

        // 4. Update KV Cache
        for (int i = 0; i < kvDim; i++) {
            kCache.array(pos * kvDim + i, k.array(i));
            vCache.array(pos * kvDim + i, v.array(i));
        }

        // 5. Multi-head Grouped Query Attention (offset-based, following mukel/llama3.java)
        for (int h = 0; h < numHeads; h++) {
            int qOffset = h * headDim;
            int attOffset = h * MAX_SEQ_LEN;

            // Compute attention scores: (Q_h · K_t) / sqrt(d_k)
            for (int t = 0; t <= pos; t++) {
                int keyCacheOffset = t * kvDim + (h / kvMul) * headDim;
                float score = dot(q, qOffset, kCache, keyCacheOffset, headDim);
                att.array(attOffset + t, score / sqrtHeadSize);
            }

            // Softmax over scores [0..pos]
            softmaxInPlace(att, attOffset, pos + 1);

            // Weighted sum of values -> xb[h*headDim .. (h+1)*headDim]
            int xbOffset = h * headDim;
            for (int i = 0; i < headDim; i++) {
                xb.array(xbOffset + i, 0.0f);
            }
            for (int t = 0; t <= pos; t++) {
                int vCacheOffset = t * kvDim + (h / kvMul) * headDim;
                float a = att.array(attOffset + t);
                for (int i = 0; i < headDim; i++) {
                    xb.array(xbOffset + i, xb.array(xbOffset + i) + a * vCache.array(vCacheOffset + i));
                }
            }
        }

        // 6. Output Projection: wo * xb -> xb2
        gemv.apply(wo, xb, xb2, dim, dim);

        // 7. Residual Add: x += xb2
        for (int i = 0; i < dim; i++) {
            x.array(i, x.array(i) + xb2.array(i));
        }

        // 8. RMSNorm (ffn_norm) -> xb, preserving x for residual
        rmsnorm(xb, x, ffnNormWeight, dim);

        // 9. Feed-Forward (SwiGLU): w2(silu(w1(xb)) * w3(xb))
        gemv.apply(w1, xb, hb, hiddenDim, dim);
        gemv.apply(w3, xb, hb2, hiddenDim, dim);
        silu.apply(hb, hiddenDim);
        elementWiseMul(hb, hb2, hiddenDim);
        gemv.apply(w2, hb, xb, dim, hiddenDim);

        // 10. Residual Add: x += xb
        for (int i = 0; i < dim; i++) {
            x.array(i, x.array(i) + xb.array(i));
        }
    }

    /**
     * RMSNorm into a separate output buffer, preserving the input for residual.
     */
    private void rmsnorm(F32Array out, F32Array x, F32Array weight, int size) {
        float ss = 0.0f;
        for (int i = 0; i < size; i++) {
            float xi = x.array(i);
            ss += xi * xi;
        }
        ss = 1.0f / (float) Math.sqrt(ss / size + 1e-5f);
        for (int i = 0; i < size; i++) {
            out.array(i, weight.array(i) * (ss * x.array(i)));
        }
    }

    /**
     * Dot product between two arrays at given offsets.
     */
    private float dot(F32Array a, int aOffset, F32Array b, int bOffset, int size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += a.array(aOffset + i) * b.array(bOffset + i);
        }
        return sum;
    }

    /**
     * In-place softmax over a slice of the array.
     */
    private void softmaxInPlace(F32Array arr, int offset, int size) {
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            float val = arr.array(offset + i);
            if (val > maxVal) maxVal = val;
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            float expVal = (float) Math.exp(arr.array(offset + i) - maxVal);
            arr.array(offset + i, expVal);
            sum += expVal;
        }
        float invSum = 1.0f / sum;
        for (int i = 0; i < size; i++) {
            arr.array(offset + i, arr.array(offset + i) * invSum);
        }
    }

    private void elementWiseMul(F32Array a, F32Array b, int size) {
        for (int i = 0; i < size; i++) {
            a.array(i, a.array(i) * b.array(i));
        }
    }
}
