package com.arturskowronski.llama3babylon.hat;

import com.arturskowronski.llama3babylon.hat.utils.MinimalGGUFGenerator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.io.IOException;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

public class LlamaModelTest {

    @TempDir
    Path tempDir;

    @Test
    public void testArchitectureConstants() {
        // Verify hardcoded Llama 3.2 1B constants
        assertEquals(2048, LlamaModel.HIDDEN_SIZE);
        assertEquals(8192, LlamaModel.INTERMEDIATE_SIZE);
        assertEquals(16, LlamaModel.NUM_LAYERS);
        assertEquals(32, LlamaModel.NUM_HEADS);
        assertEquals(8, LlamaModel.NUM_KV_HEADS);
        assertEquals(64, LlamaModel.HEAD_DIM);
        assertEquals(128256, LlamaModel.VOCAB_SIZE);
    }

    @Test
    public void testRejectsNonLlamaArchitecture() throws IOException {
        // Create a GGUF with wrong architecture
        Path ggufPath = tempDir.resolve("non_llama.gguf");
        MinimalGGUFGenerator.generate(ggufPath); // generates "general.name" but no "general.architecture"

        assertThrows(IllegalArgumentException.class, () -> new LlamaModel(ggufPath));
    }

    @Test
    public void testRejectsInvalidFile() {
        Path invalidPath = tempDir.resolve("nonexistent.gguf");
        assertThrows(IOException.class, () -> new LlamaModel(invalidPath));
    }
}
