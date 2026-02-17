package com.arturskowronski.llama3babylon.hat;

import hat.backend.Backend;

import java.util.function.Predicate;

/**
 * HAT backend types for Accelerator creation.
 */
public enum BackendType {
    /** Java Sequential backend (single-threaded, default). */
    JAVA_SEQ(be -> be.getName().contains("JavaSequential")),

    /** Java Multi-Threaded backend (CPU parallel). */
    JAVA_MT(be -> be.getName().contains("MultiThreaded")),

    /** OpenCL GPU backend via FFI. */
    OPENCL(be -> be.getName().contains("OpenCL"));

    private final Predicate<Backend> predicate;

    BackendType(Predicate<Backend> predicate) {
        this.predicate = predicate;
    }

    public Predicate<Backend> predicate() {
        return predicate;
    }
}
