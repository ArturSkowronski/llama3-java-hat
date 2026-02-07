package com.arturskowronski.llama3babylon.hat.utils;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public class MinimalGGUFGenerator {

    public static void generate(Path path) throws IOException {
        generateWithOptions(path, "MinimalTest", null, null, null);
    }

    /**
     * Generates a GGUF file with llama architecture and optional F32 tensor.
     */
    public static void generateLlamaWithTensor(Path path, String tensorName, float[] tensorData) throws IOException {
        generateWithOptions(path, "TestLlama", "llama", tensorName, tensorData);
    }

    private static void generateWithOptions(Path path, String name, String architecture, 
                                            String tensorName, float[] tensorData) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.READ, StandardOpenOption.TRUNCATE_EXISTING);
             Arena arena = Arena.ofConfined()) {
            
            String nameKey = "general.name";
            String archKey = "general.architecture";
            
            int kvCount = (architecture != null) ? 2 : 1;
            int tensorCount = (tensorName != null && tensorData != null) ? 1 : 0;
            
            // Calculate sizes
            long headerSize = 4 + 4 + 8 + 8; // magic + version + tensorCount + kvCount
            long kv1Size = 8 + nameKey.length() + 4 + 8 + name.length();
            long kv2Size = (architecture != null) ? 8 + archKey.length() + 4 + 8 + architecture.length() : 0;
            
            // Tensor info size: name (8 + len) + n_dims (4) + shape (8 per dim) + type (4) + offset (8)
            long tensorInfoSize = 0;
            if (tensorCount > 0) {
                tensorInfoSize = 8 + tensorName.length() + 4 + 8 + 4 + 8; // 1D tensor
            }
            
            long metadataEnd = headerSize + kv1Size + kv2Size + tensorInfoSize;
            long alignment = 32;
            long dataStart = (metadataEnd + alignment - 1) & ~(alignment - 1);
            long tensorDataSize = (tensorData != null) ? tensorData.length * 4L : 0;
            long totalSize = dataStart + tensorDataSize;
            
            MemorySegment segment = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalSize, arena);
            
            long offset = 0;
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 0x46554747); // "GGUF"
            offset += 4;
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 3); // Version 3
            offset += 4;
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, tensorCount);
            offset += 8;
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, kvCount);
            offset += 8;
            
            // KV 1: general.name
            offset = writeString(segment, offset, nameKey);
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 8); // STRING type
            offset += 4;
            offset = writeString(segment, offset, name);
            
            // KV 2: general.architecture (optional)
            if (architecture != null) {
                offset = writeString(segment, offset, archKey);
                segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 8); // STRING type
                offset += 4;
                offset = writeString(segment, offset, architecture);
            }
            
            // Tensor info (optional)
            if (tensorCount > 0) {
                offset = writeString(segment, offset, tensorName);
                segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 1); // n_dims = 1
                offset += 4;
                segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, tensorData.length); // shape[0]
                offset += 8;
                segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 0); // type = F32
                offset += 4;
                segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, 0); // offset from data start
                offset += 8;
            }
            
            // Write tensor data at aligned position
            if (tensorData != null) {
                for (int i = 0; i < tensorData.length; i++) {
                    segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, dataStart + (long) i * 4, tensorData[i]);
                }
            }
        }
    }

    private static long writeString(MemorySegment segment, long offset, String s) {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, bytes.length);
        MemorySegment.copy(bytes, 0, segment, ValueLayout.JAVA_BYTE, offset + 8, bytes.length);
        return offset + 8 + bytes.length;
    }
}
