package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;

/**
 * Factory that creates plain Java kernel implementations.
 * This is the current working implementation (all kernels use plain Java loops).
 */
public class PlainJavaKernelFactory implements IKernelFactory {

    @Override
    public IGEMV createGEMV(Accelerator acc) {
        return new GEMV(acc);
    }

    @Override
    public IRMSNorm createRMSNorm(Accelerator acc) {
        return new RMSNorm(acc);
    }

    @Override
    public IRoPE createRoPE(Accelerator acc) {
        return new RoPE(acc);
    }

    @Override
    public ISiLU createSiLU(Accelerator acc) {
        return new SiLU(acc);
    }

    @Override
    public ISoftmax createSoftmax(Accelerator acc) {
        return new Softmax(acc);
    }

    @Override
    public IAttention createAttention(Accelerator acc) {
        return new Attention(acc);
    }
}
