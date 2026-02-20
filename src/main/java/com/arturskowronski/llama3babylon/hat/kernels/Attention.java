package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.buffer.F32Array;

/**
 * Attention kernel for Llama 3.2 1B Instruct (FP16).
 *
 * Specifically implements Scaled Dot-Product Attention for one query head:
 * Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
 *
 * Constraints:
 * - Designed for Llama 3.2 1B (HIDDEN_SIZE=2048, NUM_HEADS=32, HEAD_DIM=64)
 * - Single-head query, multiple KV heads (GQA support)
 * - FP32 buffers (dequantized from FP16)
 */
public class Attention implements IAttention {

    public Attention(Accelerator accelerator) {
        // Kept for factory symmetry with HAT implementation.
    }

    /**
     * Computes raw attention scores (Q * K^T) / sqrt(d_k)
     * 
     * @param query head query vector [HEAD_DIM]
     * @param keys keys buffer [seq_len, HEAD_DIM]
     * @param scores output scores buffer [seq_len]
     * @param seqLen current sequence length
     * @param headDim dimension of each head (64 for Llama 3.2 1B)
     */
    public void computeScores(F32Array query, F32Array keys, F32Array scores, int seqLen, int headDim) {
        float scale = 1.0f / (float) Math.sqrt(headDim);
        for (int t = 0; t < seqLen; t++) {
            float sum = 0.0f;
            int keyOffset = t * headDim;
            for (int i = 0; i < headDim; i++) {
                sum += query.array(i) * keys.array(keyOffset + i);
            }
            scores.array(t, sum * scale);
        }
    }

    /**
     * Computes weighted sum of values: AttentionScores * V
     * 
     * @param scores softmaxed attention scores [seq_len]
     * @param values values buffer [seq_len, HEAD_DIM]
     * @param output output vector [HEAD_DIM]
     * @param seqLen current sequence length
     * @param headDim dimension of each head
     */
    public void computeValues(F32Array scores, F32Array values, F32Array output, int seqLen, int headDim) {
        for (int i = 0; i < headDim; i++) {
            float sum = 0.0f;
            for (int t = 0; t < seqLen; t++) {
                sum += scores.array(t) * values.array(t * headDim + i);
            }
            output.array(i, sum);
        }
    }

}
