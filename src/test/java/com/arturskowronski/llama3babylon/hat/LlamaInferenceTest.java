package com.arturskowronski.llama3babylon.hat;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class LlamaInferenceTest {

    @Test
    public void testArgmaxPositiveValues() {
        float[] values = {1.0f, 3.0f, 2.0f, 0.5f};
        assertEquals(1, LlamaInference.argmax(values));
    }

    @Test
    public void testArgmaxNegativeValues() {
        float[] values = {-5.0f, -1.0f, -3.0f, -2.0f};
        assertEquals(1, LlamaInference.argmax(values));
    }

    @Test
    public void testArgmaxMixed() {
        float[] values = {-1.0f, 0.0f, 5.0f, -3.0f, 2.0f};
        assertEquals(2, LlamaInference.argmax(values));
    }

    @Test
    public void testArgmaxLastElement() {
        float[] values = {0.0f, 0.0f, 0.0f, 1.0f};
        assertEquals(3, LlamaInference.argmax(values));
    }

    @Test
    public void testArgmaxSingleElement() {
        float[] values = {42.0f};
        assertEquals(0, LlamaInference.argmax(values));
    }
}
