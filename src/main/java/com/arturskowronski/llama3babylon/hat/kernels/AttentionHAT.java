package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;
import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.RW;
import static optkl.ifacemapper.MappableIface.WO;

/**
 * Attention kernel using HAT @Reflect dispatch.
 * <p>
 * Fifth kernel to be tested with HAT dispatch in real 16-layer inference.
 * Unlike previous kernels, Attention already used HAT dispatch in the base
 * implementation, so this version validates that the existing HAT usage works
 * correctly in the full inference pipeline.
 * <p>
 * Implements Scaled Dot-Product Attention:
 * Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
 * <p>
 * Two-step computation (both using HAT dispatch):
 * 1. Compute scores: Q * K^T / sqrt(d_k) → parallelized over sequence length
 * 2. Compute values: Scores * V → parallelized over head dimension
 * <p>
 * Note: Softmax is applied between steps 1 and 2 (uses SoftmaxHAT).
 */
public class AttentionHAT implements IAttention {

    private final Accelerator accelerator;

    public AttentionHAT(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    /**
     * Computes raw attention scores (Q * K^T) / sqrt(d_k) using HAT dispatch.
     * <p>
     * Parallelized over sequence length: each work item computes dot product
     * of query with one key vector.
     *
     * @param query head query vector [HEAD_DIM]
     * @param keys keys buffer [seq_len, HEAD_DIM]
     * @param scores output scores buffer [seq_len]
     * @param seqLen current sequence length
     * @param headDim dimension of each head (64 for Llama 3.2 1B)
     */
    @Override
    public void computeScores(F32Array query, F32Array keys, F32Array scores, int seqLen, int headDim) {
        float scale = 1.0f / (float) Math.sqrt(headDim);
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
            dispatchScores(cc, query, keys, scores, seqLen, headDim, scale)
        );
    }

    @Reflect
    public static void dispatchScores(@RO ComputeContext cc, @RO F32Array query, @RO F32Array keys, @WO F32Array scores, @RO int seqLen, @RO int headDim, @RO float scale) {
        cc.dispatchKernel(NDRange.of1D(seqLen), kc -> scoresKernel(kc, query, keys, scores, headDim, scale));
    }

    @Reflect
    public static void scoresKernel(@RO KernelContext kc, @RO F32Array query, @RO F32Array keys, @WO F32Array scores, @RO int headDim, @RO float scale) {
        int t = kc.gix; // target token index in sequence
        float sum = 0.0f;
        int keyOffset = t * headDim;
        for (int i = 0; i < headDim; i++) {
            sum += query.array(i) * keys.array(keyOffset + i);
        }
        scores.array(t, sum * scale);
    }

    /**
     * Computes weighted sum of values: AttentionScores * V using HAT dispatch.
     *
     * Parallelized over head dimension: each work item computes weighted sum
     * for one output dimension across all sequence positions.
     *
     * @param scores softmaxed attention scores [seq_len]
     * @param values values buffer [seq_len, HEAD_DIM]
     * @param output output vector [HEAD_DIM]
     * @param seqLen current sequence length
     * @param headDim dimension of each head
     */
    @Override
    public void computeValues(F32Array scores, F32Array values, F32Array output, int seqLen, int headDim) {
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
            dispatchValues(cc, scores, values, output, seqLen, headDim)
        );
    }

    @Reflect
    public static void dispatchValues(@RO ComputeContext cc, @RO F32Array scores, @RO F32Array values, @WO F32Array output, @RO int seqLen, @RO int headDim) {
        cc.dispatchKernel(NDRange.of1D(headDim), kc -> valuesKernel(kc, scores, values, output, seqLen, headDim));
    }

    @Reflect
    public static void valuesKernel(@RO KernelContext kc, @RO F32Array scores, @RO F32Array values, @WO F32Array output, @RO int seqLen, @RO int headDim) {
        int i = kc.gix; // index in head_dim
        float sum = 0.0f;
        for (int t = 0; t < seqLen; t++) {
            sum += scores.array(t) * values.array(t * headDim + i);
        }
        output.array(i, sum);
    }
}
