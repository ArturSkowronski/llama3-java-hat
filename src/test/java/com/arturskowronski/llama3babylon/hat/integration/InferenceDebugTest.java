package com.arturskowronski.llama3babylon.hat.integration;

import com.arturskowronski.llama3babylon.hat.*;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

@Tag("integration")
@Tag("plain-java")
public class InferenceDebugTest {

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void debugForwardPass() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));
        LlamaInference inference = new LlamaInference(modelPath);
        Tokenizer tokenizer = inference.getTokenizer();

        // Test 1: BOS token → what does the model predict first?
        int bosToken = 128000;
        float[] logits = inference.forward(bosToken, 0);

        System.out.println("=== BOS token logits ===");
        printTopK(logits, tokenizer, 10);

        // Test 2: Generate 10 tokens from BOS and print each
        System.out.println("\n=== Greedy generation from BOS ===");
        int prevToken = bosToken;
        for (int pos = 0; pos < 10; pos++) {
            logits = inference.forward(prevToken, pos);
            int nextToken = LlamaInference.argmax(logits);
            String decoded = tokenizer.decodeToken(nextToken);
            System.out.printf("pos=%d token=%d decoded='%s' logit=%.4f%n",
                    pos, nextToken, decoded, logits[nextToken]);
            prevToken = nextToken;
        }

        // Test 3: Entropy of logits distribution
        float[] softmax = softmax(logits);
        float entropy = 0;
        for (float p : softmax) {
            if (p > 0) entropy -= p * Math.log(p);
        }
        System.out.printf("\nFinal logits entropy: %.4f (higher = more uniform)%n", entropy);
    }

    private void printTopK(float[] logits, Tokenizer tokenizer, int k) {
        Integer[] indices = new Integer[logits.length];
        for (int i = 0; i < logits.length; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Float.compare(logits[b], logits[a]));

        for (int i = 0; i < k; i++) {
            int idx = indices[i];
            String decoded = tokenizer.decodeToken(idx);
            System.out.printf("  #%d: token=%d logit=%.4f decoded='%s'%n",
                    i + 1, idx, logits[idx], decoded);
        }
    }

    private float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        float[] result = new float[logits.length];
        float sum = 0;
        for (int i = 0; i < logits.length; i++) {
            result[i] = (float) Math.exp(logits[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < logits.length; i++) result[i] /= sum;
        return result;
    }
}
