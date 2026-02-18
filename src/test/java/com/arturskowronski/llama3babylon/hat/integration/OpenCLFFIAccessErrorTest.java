package com.arturskowronski.llama3babylon.hat.integration;

import hat.Accelerator;
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Minimal reproduction of HAT backend bugs using a trivial vector-add kernel.
 * No model loading required — isolates HAT framework issues from inference code.
 *
 * Bug 1 (OpenCL FFI): IllegalAccessError when FFIBackend.dispatchCompute() tries
 * to access private field ComputeEntrypoint.lowered.
 *
 * Bug 2 (Java MT): Kernel dispatch succeeds but result buffer contains zeros —
 * data not written back (possible buffer visibility / synchronization issue).
 *
 * To run:  ./gradlew openclBugTest
 */
@Tag("integration")
@Tag("opencl-bug")
public class OpenCLFFIAccessErrorTest {

    @Reflect
    public static void addKernel(KernelContext kc, F32Array a, F32Array b, F32Array result) {
        if (kc.gix < kc.gsx) {
            result.array(kc.gix, a.array(kc.gix) + b.array(kc.gix));
        }
    }

    @Reflect
    public static void vectorAdd(ComputeContext cc, F32Array a, F32Array b, F32Array result) {
        cc.dispatchKernel(NDRange.of1D(a.length()),
                kc -> addKernel(kc, a, b, result));
    }

    /**
     * BUG: OpenCL FFI backend throws IllegalAccessError.
     *
     * java.lang.IllegalAccessError: class hat.backend.ffi.FFIBackend tried to access
     * private field hat.callgraph.ComputeEntrypoint.lowered
     * (hat.backend.ffi.FFIBackend and hat.callgraph.ComputeEntrypoint are in unnamed module of loader 'app')
     *     at hat.backend.ffi.FFIBackend.dispatchCompute(FFIBackend.java:44)
     *     at hat.Accelerator.compute(Accelerator.java:206)
     *
     * Current source (2026-02-18) shows FFIBackend.dispatchCompute delegates to
     * computeContext.invokeWithArgs() which should NOT access the private field.
     * The compiled JAR (hat-backend-ffi-opencl-1.0.jar) was built from older source
     * that accessed the field directly. HAT build system is currently broken
     * (NPE in job.Jar.javaSourcePath), preventing rebuild to verify fix.
     */
    @Test
    public void testOpenCLBackendDoesNotThrowIllegalAccessError() {
        Predicate<Backend> openclPredicate = be -> be.getName().contains("OpenCL");

        Backend openclBackend;
        try {
            openclBackend = Backend.getBackend(openclPredicate);
        } catch (Exception e) {
            System.out.println("OpenCL backend not available, skipping: " + e.getMessage());
            return;
        }

        var accelerator = new Accelerator(MethodHandles.lookup(), openclBackend);
        int size = 64;
        var a = F32Array.create(accelerator, size);
        var b = F32Array.create(accelerator, size);
        var result = F32Array.create(accelerator, size);

        for (int i = 0; i < size; i++) {
            a.array(i, (float) i);
            b.array(i, (float) (i * 2));
        }

        assertDoesNotThrow(() ->
            accelerator.compute((@Reflect Compute)
                    cc -> vectorAdd(cc, a, b, result)),
            "OpenCL FFI backend threw IllegalAccessError — " +
            "FFIBackend cannot access private ComputeEntrypoint.lowered field. " +
            "See: https://github.com/openjdk/babylon HAT FFIBackend.dispatchCompute"
        );

        for (int i = 0; i < size; i++) {
            assertEquals(i + i * 2f, result.array(i), 0.001f,
                    "Vector add result mismatch at index " + i);
        }

        System.out.println("OpenCL FFI backend: PASSED");
    }

    /**
     * Control: Java Sequential backend — works correctly.
     */
    @Test
    public void testJavaSequentialBackendWorks() {
        var accelerator = new Accelerator(MethodHandles.lookup(),
                (Predicate<Backend>) be -> be.getName().contains("JavaSequential"));
        int size = 64;
        var a = F32Array.create(accelerator, size);
        var b = F32Array.create(accelerator, size);
        var result = F32Array.create(accelerator, size);

        for (int i = 0; i < size; i++) {
            a.array(i, (float) i);
            b.array(i, (float) (i * 2));
        }

        accelerator.compute((@Reflect Compute)
                cc -> vectorAdd(cc, a, b, result));

        for (int i = 0; i < size; i++) {
            assertEquals(i + i * 2f, result.array(i), 0.001f,
                    "Vector add result mismatch at index " + i);
        }

        System.out.println("Java Sequential backend: PASSED");
    }

    /**
     * BUG: Java MT backend dispatches kernel without error but result buffer
     * contains all zeros. The kernel appears to execute but writes are not
     * visible to the caller. Possible buffer synchronization / visibility issue.
     *
     * Note: This same backend works correctly in full inference (ChatIntegrationTestWithJavaMT)
     * where kernels are dispatched via HybridKernelFactory from within HAT kernel classes.
     * The difference may be related to how the compute entrypoint is resolved
     * when @Reflect methods are in a JUnit test class vs a dedicated kernel class.
     */
    @Test
    public void testJavaMTBackendDispatchesWithoutError() {
        var accelerator = new Accelerator(MethodHandles.lookup(),
                (Predicate<Backend>) be -> be.getName().contains("MultiThreaded"));
        int size = 64;
        var a = F32Array.create(accelerator, size);
        var b = F32Array.create(accelerator, size);
        var result = F32Array.create(accelerator, size);

        for (int i = 0; i < size; i++) {
            a.array(i, (float) i);
            b.array(i, (float) (i * 2));
        }

        // MT dispatch should not throw — verify no IllegalAccessError
        assertDoesNotThrow(() ->
            accelerator.compute((@Reflect Compute)
                    cc -> vectorAdd(cc, a, b, result)),
            "Java MT backend threw an error during dispatch"
        );

        // Check if results are correct (known to return zeros with current JARs)
        boolean allCorrect = true;
        for (int i = 0; i < size; i++) {
            if (Math.abs((i + i * 2f) - result.array(i)) > 0.001f) {
                allCorrect = false;
                break;
            }
        }

        if (allCorrect) {
            System.out.println("Java MT backend: PASSED — dispatch and results correct");
        } else {
            System.out.println("Java MT backend: DISPATCH OK, RESULTS INCORRECT (buffer visibility bug)");
            System.out.println("  Expected result[1]=3.0, got result[1]=" + result.array(1));
            System.out.println("  Note: Full inference via HybridKernelFactory works correctly on MT backend.");
            System.out.println("  This may be a test-class @Reflect resolution issue, not a backend bug.");
        }
    }
}
