package com.arturskowronski.llama3babylon;

import com.arturskowronski.llama3babylon.samples.HATSample;

public class RuntimeCheck {
    public static void main(String[] args) {
        System.out.println("=== Babylon Runtime Check ===");
        System.out.println("Java Version:    " + System.getProperty("java.version"));
        System.out.println("Java Vendor:     " + System.getProperty("java.vendor"));
        System.out.println("Runtime Version: " + System.getProperty("java.runtime.version"));
        System.out.println("OS:              " + System.getProperty("os.name") + " (" + System.getProperty("os.arch") + ")");

        System.out.println("\n=== Code Reflection Check ===");
        boolean hasCodeReflection = ModuleLayer.boot().findModule("jdk.incubator.code").isPresent() ||
                                     ModuleLayer.boot().findModule("java.lang.reflect.code").isPresent();

        if (hasCodeReflection) {
            System.out.println("Code Reflection Module Present: true ✅");
            System.out.println("Babylon is ready for use!");

            System.out.println("\n=== HAT Verification ===");
            try {
                HATSample.main(new String[0]);
                System.out.println("\nHAT Verification Successful! ✅");
            } catch (Exception e) {
                System.out.println("\nHAT Verification Failed! ❌");
                e.printStackTrace();
            }
        } else {
            System.out.println("Code Reflection Module Present: false ⚠️");
            System.out.println("You need Babylon JDK for Code Reflection features.");
            System.out.println("See: https://jdk.java.net/babylon/");
        }
    }
}
