package com.arturskowronski.llama3babylon.hat.kernels;

import hat.buffer.F32Array;

/**
 * Interface for RoPE (Rotary Positional Embedding) kernel implementations.
 */
public interface IRoPE {

    /**
     * Applies RoPE to a vector (Q or K).
     *
     * @param vec input/output vector [num_heads, head_dim] (modified in-place)
     * @param pos current token position in sequence
     * @param numHeads number of heads
     * @param headDim dimension of each head
     * @param theta base for frequency calculation (typically 10000.0 or 500000.0 for Llama 3)
     */
    void apply(F32Array vec, int pos, int numHeads, int headDim, float theta);
}
