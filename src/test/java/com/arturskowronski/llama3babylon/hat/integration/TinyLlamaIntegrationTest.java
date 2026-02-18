package com.arturskowronski.llama3babylon.hat.integration;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

@Tag("integration")
@Tag("plain-java")
public class TinyLlamaIntegrationTest {

    @Test
    @EnabledIfEnvironmentVariable(named = "TINY_LLAMA_PATH", matches = ".*")
    public void testTinyLlamaHeaderReading() throws IOException {
        String modelPathStr = System.getenv("TINY_LLAMA_PATH");
        java.nio.file.Path modelPath = java.nio.file.Paths.get(modelPathStr);
        
        assertTrue(java.nio.file.Files.exists(modelPath), "TinyLlama model file not found at: " + modelPathStr);
        
        var metadata = com.arturskowronski.llama3babylon.hat.GGUFReader.readMetadata(modelPath);
        
        assertNotNull(metadata, "Metadata should not be null");
        assertTrue(metadata.version() >= 2, "GGUF version should be at least 2");
        assertTrue(metadata.tensorCount() > 0, "Should have tensors");
        assertNotNull(metadata.metadata().get("general.architecture"), "Architecture should be present in metadata");
    }
}
