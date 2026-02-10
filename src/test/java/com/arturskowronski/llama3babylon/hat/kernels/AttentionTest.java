package com.arturskowronski.llama3babylon.hat.kernels;

import hat.Accelerator;
import hat.buffer.F32Array;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AttentionTest {

    @Test
    public void testAttentionMechanism() {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup());
        Attention attention = new Attention(accelerator);
        Softmax softmax = new Softmax(accelerator);

        int headDim = 64;
        int seqLen = 4;

        F32Array query = F32Array.create(accelerator, headDim);
        F32Array keys = F32Array.create(accelerator, seqLen * headDim);
        F32Array values = F32Array.create(accelerator, seqLen * headDim);
        F32Array scores = F32Array.create(accelerator, seqLen);
        F32Array output = F32Array.create(accelerator, headDim);

        // Initialize: query=0.1, keys=0.01, values[t, i]=i
        for (int i = 0; i < headDim; i++) query.array(i, 0.1f);
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < headDim; i++) {
                keys.array(t * headDim + i, 0.01f);
                values.array(t * headDim + i, (float) i);
            }
        }

        // 1. Scores
        attention.computeScores(query, keys, scores, seqLen, headDim);
        
        // Dot product = 64 * (0.1 * 0.01) = 0.064
        // scale = 1/sqrt(64) = 0.125
        // expected score = 0.008
        for (int t = 0; t < seqLen; t++) {
            assertEquals(0.008f, scores.array(t), 1e-5f);
        }

        // 2. Softmax
        softmax.apply(scores, seqLen);
        // All scores equal -> all should be 1/seqLen = 0.25
        for (int t = 0; t < seqLen; t++) {
            assertEquals(0.25f, scores.array(t), 1e-5f);
        }

        // 3. Values
        attention.computeValues(scores, values, output, seqLen, headDim);
        // Output[i] = sum_t (0.25 * i) = 4 * 0.25 * i = i
        for (int i = 0; i < headDim; i++) {
            assertEquals((float) i, output.array(i), 1e-5f);
        }
    }
}
