package com.arturskowronski.llama3babylon.hat;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;

public class MinimalGGUFGenerator {

    public static void generate(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.READ, StandardOpenOption.TRUNCATE_EXISTING);
             Arena arena = Arena.ofConfined()) {
            
            // Magic (4), Version (4), Tensor Count (8), KV Count (8)
            // KV: 
            //   key "general.name" (string)
            //   type STRING (4)
            //   value "MinimalTest" (string)
            
            String nameKey = "general.name";
            String nameValue = "MinimalTest";
            
            long headerSize = 4 + 4 + 8 + 8;
            long kv1KeySize = 8 + nameKey.getBytes(StandardCharsets.UTF_8).length;
            long kv1ValueSize = 4 + 8 + nameValue.getBytes(StandardCharsets.UTF_8).length;
            
            long totalSize = headerSize + kv1KeySize + kv1ValueSize;
            
            MemorySegment segment = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalSize, arena);
            
            long offset = 0;
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 0x46554747); // "GGUF"
            offset += 4;
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 3); // Version 3
            offset += 4;
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, 0); // Tensor count
            offset += 8;
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, 1); // KV count
            offset += 8;
            
            // KV 1 Key
            offset = writeString(segment, offset, nameKey);
            // KV 1 Type
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 8); // STRING
            offset += 4;
            // KV 1 Value
            offset = writeString(segment, offset, nameValue);
        }
    }

    private static long writeString(MemorySegment segment, long offset, String s) {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, bytes.length);
        MemorySegment.copy(bytes, 0, segment, ValueLayout.JAVA_BYTE, offset + 8, bytes.length);
        return offset + 8 + bytes.length;
    }
}
