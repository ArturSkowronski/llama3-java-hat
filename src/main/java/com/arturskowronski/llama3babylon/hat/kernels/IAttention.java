package com.arturskowronski.llama3babylon.hat.kernels;

import hat.buffer.F32Array;

/**
 * Interface for Attention kernel implementations.
 */
public interface IAttention {

    /**
     * Computes raw attention scores (Q * K^T) / sqrt(d_k)
     *
     * @param query head query vector [HEAD_DIM]
     * @param keys keys buffer [seq_len, HEAD_DIM]
     * @param scores output scores buffer [seq_len]
     * @param seqLen current sequence length
     * @param headDim dimension of each head (64 for Llama 3.2 1B)
     */
    void computeScores(F32Array query, F32Array keys, F32Array scores, int seqLen, int headDim);

    /**
     * Computes weighted sum of values: AttentionScores * V
     *
     * @param scores softmaxed attention scores [seq_len]
     * @param values values buffer [seq_len, HEAD_DIM]
     * @param output output vector [HEAD_DIM]
     * @param seqLen current sequence length
     * @param headDim dimension of each head
     */
    void computeValues(F32Array scores, F32Array values, F32Array output, int seqLen, int headDim);
}
