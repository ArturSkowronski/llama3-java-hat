package com.arturskowronski.llama3babylon.hat;

import jdk.incubator.code.Op;
import jdk.incubator.code.interpreter.Interpreter;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;

public class CodeReflectionSample {

    @Reflect
    public static int add(int a, int b) {
        return a + b;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== Code Reflection Sample ===");

        Method method = CodeReflectionSample.class.getDeclaredMethod("add", int.class, int.class);

        CoreOp.FuncOp addOp = Op.ofMethod(method).orElseThrow(() -> new RuntimeException("No code model found for 'add' method. Make sure it's annotated with @CodeReflection and compiled with --enable-preview."));

        System.out.println("Code Model for 'add' method:");
        System.out.println(addOp.toText());

        System.out.println("\nInterpreting the code model...");
        int result = (int) Interpreter.invoke(MethodHandles.lookup(), addOp, 10, 20);
        System.out.println("Result of 10 + 20 interpreted: " + result);

        if (result == 30) {
            System.out.println("Code Reflection interpretation successful! ✅");
        } else {
            System.out.println("Code Reflection interpretation failed! ❌");
        }
    }
}
