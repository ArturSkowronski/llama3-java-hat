package com.arturskowronski.llama3babylon.hat;

import com.arturskowronski.llama3babylon.hat.utils.MinimalGGUFGenerator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.io.IOException;
import java.nio.file.Path;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.types.F16;
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

    @Test
    public void testMapTensorF32() throws IOException {
        // Create a GGUF with llama architecture and F32 tensor
        Path ggufPath = tempDir.resolve("llama_with_tensor.gguf");
        float[] testData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        MinimalGGUFGenerator.generateLlamaWithTensor(ggufPath, "test.weight", testData);

        LlamaModel model = new LlamaModel(ggufPath, false); // false = skip strict validation for unit test
        
        assertTrue(model.hasTensor("test.weight"));
        assertNotNull(model.getTensorInfo("test.weight"));
        
        F32Array tensor = model.mapTensor("test.weight");
        assertEquals(5, tensor.length());
        assertEquals(1.0f, tensor.array(0), 0.0001f);
        assertEquals(2.0f, tensor.array(1), 0.0001f);
        assertEquals(3.0f, tensor.array(2), 0.0001f);
        assertEquals(4.0f, tensor.array(3), 0.0001f);
        assertEquals(5.0f, tensor.array(4), 0.0001f);
    }

    @Test
    public void testMapTensorCaching() throws IOException {
        Path ggufPath = tempDir.resolve("llama_cache_test.gguf");
        float[] testData = {1.0f, 2.0f, 3.0f};
        MinimalGGUFGenerator.generateLlamaWithTensor(ggufPath, "cached.weight", testData);

        LlamaModel model = new LlamaModel(ggufPath, false);
        
        F32Array tensor1 = model.mapTensor("cached.weight");
        F32Array tensor2 = model.mapTensor("cached.weight");
        
        // Should return the same cached instance
        assertSame(tensor1, tensor2);
    }

    @Test
    public void testMapTensorF16() throws IOException {
        Path ggufPath = tempDir.resolve("llama_f16_tensor.gguf");
        float[] testData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        MinimalGGUFGenerator.generateLlamaWithF16Tensor(ggufPath, "test.weight", testData);

        LlamaModel model = new LlamaModel(ggufPath, false);

        F16Array tensor = model.mapTensorF16("test.weight");
        assertEquals(5, tensor.length());
        // F16 has limited precision, so use tolerance
        assertEquals(1.0f, F16.f16ToFloat(tensor.array(0)), 0.01f);
        assertEquals(2.0f, F16.f16ToFloat(tensor.array(1)), 0.01f);
        assertEquals(3.0f, F16.f16ToFloat(tensor.array(2)), 0.01f);
        assertEquals(4.0f, F16.f16ToFloat(tensor.array(3)), 0.01f);
        assertEquals(5.0f, F16.f16ToFloat(tensor.array(4)), 0.01f);
    }

    @Test
    public void testMapTensorF16Caching() throws IOException {
        Path ggufPath = tempDir.resolve("llama_f16_cache.gguf");
        float[] testData = {1.0f, 2.0f, 3.0f};
        MinimalGGUFGenerator.generateLlamaWithF16Tensor(ggufPath, "cached.weight", testData);

        LlamaModel model = new LlamaModel(ggufPath, false);

        F16Array tensor1 = model.mapTensorF16("cached.weight");
        F16Array tensor2 = model.mapTensorF16("cached.weight");

        assertSame(tensor1, tensor2);
    }

    @Test
    public void testMapTensorF16RejectsF32() throws IOException {
        // F32 tensor should be rejected by mapTensorF16
        Path ggufPath = tempDir.resolve("llama_f32_reject.gguf");
        MinimalGGUFGenerator.generateLlamaWithTensor(ggufPath, "f32.weight", new float[]{1.0f});

        LlamaModel model = new LlamaModel(ggufPath, false);

        assertThrows(IOException.class, () -> model.mapTensorF16("f32.weight"));
    }

    @Test
    public void testMapTensorNotFound() throws IOException {
        Path ggufPath = tempDir.resolve("llama_no_tensor.gguf");
        MinimalGGUFGenerator.generateLlamaWithTensor(ggufPath, "existing.weight", new float[]{1.0f});

        LlamaModel model = new LlamaModel(ggufPath, false);
        
        assertFalse(model.hasTensor("nonexistent.weight"));
        assertThrows(IOException.class, () -> model.mapTensor("nonexistent.weight"));
    }
}
