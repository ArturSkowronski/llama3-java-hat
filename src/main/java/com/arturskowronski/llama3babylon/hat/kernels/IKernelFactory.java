package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;

/**
 * Factory interface for creating kernel implementations.
 * Allows switching between plain Java and HAT implementations.
 */
public interface IKernelFactory {

    /**
     * Creates a GEMV (Matrix-Vector Multiplication) kernel.
     */
    IGEMV createGEMV(Accelerator acc);

    /**
     * Creates an RMSNorm (Root Mean Square Layer Normalization) kernel.
     */
    IRMSNorm createRMSNorm(Accelerator acc);

    /**
     * Creates a RoPE (Rotary Positional Embedding) kernel.
     */
    IRoPE createRoPE(Accelerator acc);

    /**
     * Creates a SiLU (Sigmoid Linear Unit) kernel.
     */
    ISiLU createSiLU(Accelerator acc);

    /**
     * Creates a Softmax kernel.
     */
    ISoftmax createSoftmax(Accelerator acc);

    /**
     * Creates an Attention kernel.
     */
    IAttention createAttention(Accelerator acc);
}
