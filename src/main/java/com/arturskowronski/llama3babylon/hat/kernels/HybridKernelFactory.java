package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;

import java.util.EnumSet;
import java.util.Set;

/**
 * Factory that allows selective enablement of HAT kernels.
 * Each kernel type can be individually toggled between HAT dispatch and plain Java.
 */
public class HybridKernelFactory implements IKernelFactory {

    /**
     * Kernel types that can be selectively enabled for HAT dispatch.
     */
    public enum KernelType {
        GEMV,
        RMSNORM,
        ROPE,
        SILU,
        SOFTMAX,
        ATTENTION
    }

    private final Set<KernelType> enableHAT;

    /**
     * Creates a hybrid factory with no HAT kernels enabled (all plain Java).
     */
    public HybridKernelFactory() {
        this.enableHAT = EnumSet.noneOf(KernelType.class);
    }

    /**
     * Creates a hybrid factory with specified kernels enabled for HAT dispatch.
     *
     * @param enableHAT set of kernel types to use HAT dispatch for
     */
    public HybridKernelFactory(Set<KernelType> enableHAT) {
        this.enableHAT = EnumSet.copyOf(enableHAT);
    }

    @Override
    public IGEMV createGEMV(Accelerator acc) {
        if (enableHAT.contains(KernelType.GEMV)) {
            // TODO: Create GEMVHAT when implemented
            throw new UnsupportedOperationException("GEMV HAT not yet implemented");
        }
        return new GEMV(acc);
    }

    @Override
    public IRMSNorm createRMSNorm(Accelerator acc) {
        if (enableHAT.contains(KernelType.RMSNORM)) {
            return new RMSNormHAT(acc);
        }
        return new RMSNorm(acc);
    }

    @Override
    public IRoPE createRoPE(Accelerator acc) {
        if (enableHAT.contains(KernelType.ROPE)) {
            return new RoPEHAT(acc);
        }
        return new RoPE(acc);
    }

    @Override
    public ISiLU createSiLU(Accelerator acc) {
        if (enableHAT.contains(KernelType.SILU)) {
            return new SiLUHAT(acc);
        }
        return new SiLU(acc);
    }

    @Override
    public ISoftmax createSoftmax(Accelerator acc) {
        if (enableHAT.contains(KernelType.SOFTMAX)) {
            return new SoftmaxHAT(acc);
        }
        return new Softmax(acc);
    }

    @Override
    public IAttention createAttention(Accelerator acc) {
        if (enableHAT.contains(KernelType.ATTENTION)) {
            return new AttentionHAT(acc);
        }
        return new Attention(acc);
    }
}
