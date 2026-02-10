package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;

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
public class Attention {

    private final Accelerator accelerator;

    public Attention(Accelerator accelerator) {
        this.accelerator = accelerator;
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
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
                dispatchScores(cc, query, keys, scores, seqLen, headDim, scale)
        );
    }

    @Reflect
    public static void scoresKernel(KernelContext kc, F32Array query, F32Array keys, F32Array scores, int headDim, float scale) {
        int t = kc.gix; // target token index in sequence
        float sum = 0.0f;
        int keyOffset = t * headDim;
        for (int i = 0; i < headDim; i++) {
            sum += query.array(i) * keys.array(keyOffset + i);
        }
        scores.array(t, sum * scale);
    }

    @Reflect
    public static void dispatchScores(ComputeContext cc, F32Array query, F32Array keys, F32Array scores, int seqLen, int headDim, float scale) {
        cc.dispatchKernel(NDRange.of1D(seqLen), kc -> scoresKernel(kc, query, keys, scores, headDim, scale));
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
        accelerator.compute((Accelerator.@Reflect Compute) cc ->
                dispatchValues(cc, scores, values, output, seqLen, headDim)
        );
    }

    @Reflect
    public static void valuesKernel(KernelContext kc, F32Array scores, F32Array values, F32Array output, int seqLen, int headDim) {
        int i = kc.gix; // index in head_dim
        float sum = 0.0f;
        for (int t = 0; t < seqLen; t++) {
            sum += scores.array(t) * values.array(t * headDim + i);
        }
        output.array(i, sum);
    }

    @Reflect
    public static void dispatchValues(ComputeContext cc, F32Array scores, F32Array values, F32Array output, int seqLen, int headDim) {
        cc.dispatchKernel(NDRange.of1D(headDim), kc -> valuesKernel(kc, scores, values, output, seqLen, headDim));
    }

    // ========== Test/Demo ==========

    public static void main(String[] args) {
        System.out.println("=== Attention Kernel Test ===");

        Accelerator accelerator = new Accelerator(MethodHandles.lookup());
        Attention attention = new Attention(accelerator);
        Softmax softmax = new Softmax(accelerator);

        int headDim = 64;
        int seqLen = 10;

        F32Array query = F32Array.create(accelerator, headDim);
        F32Array keys = F32Array.create(accelerator, seqLen * headDim);
        F32Array values = F32Array.create(accelerator, seqLen * headDim);
        F32Array scores = F32Array.create(accelerator, seqLen);
        F32Array output = F32Array.create(accelerator, headDim);

        // Initialize with dummy data
        for (int i = 0; i < headDim; i++) query.array(i, 0.1f);
        for (int i = 0; i < seqLen * headDim; i++) {
            keys.array(i, 0.01f);
            values.array(i, (float) (i % headDim));
        }

        System.out.println("Step 1: Computing scores...");
        attention.computeScores(query, keys, scores, seqLen, headDim);

        System.out.println("Step 2: Applying softmax...");
        softmax.apply(scores, seqLen);

        System.out.println("Step 3: Computing weighted values...");
        attention.computeValues(scores, values, output, seqLen, headDim);

        System.out.println("Output (first 5 elements):");
        for (int i = 0; i < 5; i++) {
            System.out.printf("  [%d] = %.4f%n", i, output.array(i));
        }

        // Verification:
        // Query is all 0.1, Keys are all 0.01.
        // Dot product = 64 * (0.1 * 0.01) = 64 * 0.001 = 0.064
        // scale = 1/sqrt(64) = 1/8 = 0.125
        // scaled score = 0.064 * 0.125 = 0.008
        // All scores are 0.008. Softmax of N equal values is 1/N.
        // softmax score = 1/10 = 0.1 for all t.
        // Output[i] = sum_t (0.1 * values[t, i])
        // Since values[t, i] = i for all t (because i < headDim and i % headDim = i)
        // Output[i] = 10 * (0.1 * i) = i
        
        boolean passed = true;
        for (int i = 0; i < 5; i++) {
            if (Math.abs(output.array(i) - i) > 1e-5) {
                System.out.printf("Mismatch at [%d]: expected %.4f, got %.4f%n", i, (float)i, output.array(i));
                passed = false;
            }
        }

        if (passed) {
            System.out.println("Attention test PASSED ✅");
        } else {
            System.out.println("Attention test FAILED ❌");
        }
    }
}
