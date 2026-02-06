package com.arturskowronski.llama3babylon.hat;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.io.IOException;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

public class GGUFReaderTest {

    @TempDir
    Path tempDir;

    @Test
    public void testReadMinimalGGUF() throws IOException {
        Path ggufPath = tempDir.resolve("test.gguf");
        MinimalGGUFGenerator.generate(ggufPath);

        GGUFReader.GGUFMetadata metadata = GGUFReader.readMetadata(ggufPath);

        assertNotNull(metadata);
        assertEquals(3, metadata.version());
        assertEquals(0, metadata.tensorCount());
        assertEquals(1, metadata.metadataKVCount());
        assertEquals("MinimalTest", metadata.metadata().get("general.name"));
    }

    @Test
    public void testInvalidMagic() throws IOException {
        Path invalidPath = tempDir.resolve("invalid.gguf");
        java.nio.file.Files.write(invalidPath, new byte[]{1, 2, 3, 4});

        assertThrows(IOException.class, () -> {
            GGUFReader.readMetadata(invalidPath);
        });
    }
    @Test
    public void testReadMinimalGGUFWithArray() throws IOException {
        Path ggufPath = tempDir.resolve("test_array.gguf");
        try (java.nio.channels.FileChannel channel = java.nio.channels.FileChannel.open(ggufPath, java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.WRITE, java.nio.file.StandardOpenOption.READ, java.nio.file.StandardOpenOption.TRUNCATE_EXISTING);
             java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
            
            String key = "test.array";
            long totalSize = 4 + 4 + 8 + 8 + (8 + key.length()) + 4 + (4 + 8 + (2 * 4));
            java.lang.foreign.MemorySegment segment = channel.map(java.nio.channels.FileChannel.MapMode.READ_WRITE, 0, totalSize, arena);
            
            long offset = 0;
            segment.set(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, offset, 0x46554747); offset += 4;
            segment.set(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, offset, 3); offset += 4;
            segment.set(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED, offset, 0); offset += 8;
            segment.set(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED, offset, 1); offset += 8;
            
            // Key
            byte[] keyBytes = key.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            segment.set(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED, offset, keyBytes.length);
            java.lang.foreign.MemorySegment.copy(keyBytes, 0, segment, java.lang.foreign.ValueLayout.JAVA_BYTE, offset + 8, keyBytes.length);
            offset += 8 + keyBytes.length;
            
            // Type ARRAY
            segment.set(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, offset, 9); offset += 4;
            
            // Value: INT32 type (5), 2 items
            segment.set(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, offset, 5); offset += 4;
            segment.set(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED, offset, 2); offset += 8;
            segment.set(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, offset, 42); offset += 4;
            segment.set(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, offset, 43); offset += 4;
        }

        GGUFReader.GGUFMetadata metadata = GGUFReader.readMetadata(ggufPath);
        java.util.List<Integer> array = (java.util.List<Integer>) metadata.metadata().get("test.array");
        assertEquals(2, array.size());
        assertEquals(42, array.get(0));
        assertEquals(43, array.get(1));
    }
}
