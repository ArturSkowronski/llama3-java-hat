package com.arturskowronski.llama3babylon.samples;

import jdk.incubator.code.Reflect;
import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.S32Array;
import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.WO;
import java.lang.invoke.MethodHandles;

public class HATSample {

    @Reflect
    public static void vectorAdd(@RO KernelContext kc, @RO S32Array a, @RO S32Array b, @WO S32Array c) {
        int i = kc.gix;
        c.array(i, a.array(i) + b.array(i));
    }

    @Reflect
    public static void computeAdd(@RO ComputeContext cc, @RO S32Array a, @RO S32Array b, @WO S32Array c, @RO int size) {
        System.out.println("Inside computeAdd");
        cc.dispatchKernel(NDRange.of1D(size), kcx -> vectorAdd(kcx, a, b, c));
    }

    public static void main(String[] args) {
        System.out.println("=== HAT Sample ===");

        Accelerator accelerator = new Accelerator(MethodHandles.lookup());

        int size = 10;
        S32Array a = S32Array.create(accelerator, size);
        S32Array b = S32Array.create(accelerator, size);
        S32Array c = S32Array.create(accelerator, size);

        for (int i = 0; i < size; i++) {
            a.array(i, i);
            b.array(i, i * 2);
        }

        System.out.println("Dispatching HAT kernel via reflected compute method...");
        accelerator.compute((Accelerator.@Reflect Compute) cc -> computeAdd(cc, a, b, c, size));

        System.out.println("Verifying results...");
        boolean success = true;
        for (int i = 0; i < size; i++) {
            int expected = i + (i * 2);
            int actual = c.array(i);
            if (actual != expected) {
                System.out.println("Error at " + i + ": expected " + expected + ", got " + actual);
                success = false;
            }
        }

        if (success) {
            System.out.println("HAT vectorAdd successful! ✅");
        } else {
            System.out.println("HAT vectorAdd failed! ❌");
        }
    }
}
