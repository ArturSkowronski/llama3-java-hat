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
        generateWithOptions(path, "TestLlama", "llama", new String[]{tensorName}, new float[][]{tensorData});
    }

    public static void generateLlamaWithTensors(Path path, String[] tensorNames, float[][] tensorData) throws IOException {
        generateWithOptions(path, "TestLlama", "llama", tensorNames, tensorData);
    }

    /**
     * Generates a GGUF file with mixed F32 and F16 tensors.
     * @param tensorTypes array of GGUF types: 0 = F32, 1 = F16
     */
    public static void generateLlamaWithMixedTensors(Path path, String[] tensorNames, float[][] tensorData, int[] tensorTypes) throws IOException {
        generateWithMixedOptions(path, "TestLlama", "llama", tensorNames, tensorData, tensorTypes);
    }

    /**
     * Generates a GGUF file with llama architecture and an F16 tensor.
     */
    public static void generateLlamaWithF16Tensor(Path path, String tensorName, float[] values) throws IOException {
        generateWithF16Options(path, "TestLlama", "llama", tensorName, values);
    }

    private static void generateWithF16Options(Path path, String name, String architecture,
                                                String tensorName, float[] values) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.READ, StandardOpenOption.TRUNCATE_EXISTING);
             Arena arena = Arena.ofConfined()) {

            String nameKey = "general.name";
            String archKey = "general.architecture";

            int kvCount = 2;
            int tensorCount = 1;

            long headerSize = 4 + 4 + 8 + 8;
            long kv1Size = 8 + nameKey.length() + 4 + 8 + name.length();
            long kv2Size = 8 + archKey.length() + 4 + 8 + architecture.length();
            long tensorInfoSize = 8 + tensorName.length() + 4 + 8 + 4 + 8;

            long metadataEnd = headerSize + kv1Size + kv2Size + tensorInfoSize;
            long alignment = 32;
            long dataStart = (metadataEnd + alignment - 1) & ~(alignment - 1);
            long totalTensorDataSize = (long) values.length * 2; // F16 = 2 bytes per element
            long totalSize = dataStart + totalTensorDataSize;

            MemorySegment segment = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalSize, arena);

            long offset = 0;
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 0x46554747);
            offset += 4;
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 3);
            offset += 4;
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, tensorCount);
            offset += 8;
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, kvCount);
            offset += 8;

            offset = writeString(segment, offset, nameKey);
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 8);
            offset += 4;
            offset = writeString(segment, offset, name);

            offset = writeString(segment, offset, archKey);
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 8);
            offset += 4;
            offset = writeString(segment, offset, architecture);

            // Tensor info: F16 type = 1
            offset = writeString(segment, offset, tensorName);
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 1); // n_dims = 1
            offset += 4;
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, values.length); // shape[0]
            offset += 8;
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 1); // type = F16
            offset += 4;
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, 0L); // offset from data start
            offset += 8;

            // Write F16 data
            for (int i = 0; i < values.length; i++) {
                short f16 = Float.floatToFloat16(values[i]);
                segment.set(ValueLayout.JAVA_SHORT_UNALIGNED, dataStart + (long) i * 2, f16);
            }
        }
    }

    private static void generateWithOptions(Path path, String name, String architecture, 
                                            String[] tensorNames, float[][] tensorData) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.READ, StandardOpenOption.TRUNCATE_EXISTING);
             Arena arena = Arena.ofConfined()) {
            
            String nameKey = "general.name";
            String archKey = "general.architecture";
            
            int kvCount = (architecture != null) ? 2 : 1;
            int tensorCount = (tensorNames != null && tensorData != null) ? tensorNames.length : 0;
            
            // Calculate sizes
            long headerSize = 4 + 4 + 8 + 8; // magic + version + tensorCount + kvCount
            long kv1Size = 8 + nameKey.length() + 4 + 8 + name.length();
            long kv2Size = (architecture != null) ? 8 + archKey.length() + 4 + 8 + architecture.length() : 0;
            
            // Tensor info size: name (8 + len) + n_dims (4) + shape (8 per dim) + type (4) + offset (8)
            long tensorInfoSize = 0;
            if (tensorCount > 0) {
                for (String tensorName : tensorNames) {
                    tensorInfoSize += 8 + tensorName.length() + 4 + 8 + 4 + 8; // 1D tensor
                }
            }
            
            long metadataEnd = headerSize + kv1Size + kv2Size + tensorInfoSize;
            long alignment = 32;
            long dataStart = (metadataEnd + alignment - 1) & ~(alignment - 1);
            
            long totalTensorDataSize = 0;
            long[] tensorOffsets = new long[tensorCount];
            if (tensorCount > 0) {
                for (int i = 0; i < tensorCount; i++) {
                    tensorOffsets[i] = totalTensorDataSize;
                    totalTensorDataSize += tensorData[i].length * 4L;
                    // Align each tensor data if needed, but for simplicity we'll just stack them
                    // GGUF requires data to be aligned to 'alignment' from start of file.
                    // Actually, tensor offsets are relative to dataStart, and dataStart is already aligned.
                    // We should probably align each tensor offset if we want to be strict.
                    totalTensorDataSize = (totalTensorDataSize + alignment - 1) & ~(alignment - 1);
                }
            }

            long totalSize = dataStart + totalTensorDataSize;
            
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
                for (int i = 0; i < tensorCount; i++) {
                    offset = writeString(segment, offset, tensorNames[i]);
                    segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 1); // n_dims = 1
                    offset += 4;
                    segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, tensorData[i].length); // shape[0]
                    offset += 8;
                    segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 0); // type = F32
                    offset += 4;
                    segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, tensorOffsets[i]); // offset from data start
                    offset += 8;
                }
            }
            
            // Write tensor data at aligned positions
            if (tensorCount > 0) {
                for (int i = 0; i < tensorCount; i++) {
                    long tensorDataStart = dataStart + tensorOffsets[i];
                    for (int j = 0; j < tensorData[i].length; j++) {
                        segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, tensorDataStart + (long) j * 4, tensorData[i][j]);
                    }
                }
            }
        }
    }

    private static void generateWithMixedOptions(Path path, String name, String architecture,
                                                     String[] tensorNames, float[][] tensorData, int[] tensorTypes) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.READ, StandardOpenOption.TRUNCATE_EXISTING);
             Arena arena = Arena.ofConfined()) {

            String nameKey = "general.name";
            String archKey = "general.architecture";

            int kvCount = 2;
            int tensorCount = tensorNames.length;

            long headerSize = 4 + 4 + 8 + 8;
            long kv1Size = 8 + nameKey.length() + 4 + 8 + name.length();
            long kv2Size = 8 + archKey.length() + 4 + 8 + architecture.length();

            long tensorInfoSize = 0;
            for (String tensorName : tensorNames) {
                tensorInfoSize += 8 + tensorName.length() + 4 + 8 + 4 + 8;
            }

            long metadataEnd = headerSize + kv1Size + kv2Size + tensorInfoSize;
            long alignment = 32;
            long dataStart = (metadataEnd + alignment - 1) & ~(alignment - 1);

            long totalTensorDataSize = 0;
            long[] tensorOffsets = new long[tensorCount];
            for (int i = 0; i < tensorCount; i++) {
                tensorOffsets[i] = totalTensorDataSize;
                int bytesPerElement = (tensorTypes[i] == 1) ? 2 : 4;
                totalTensorDataSize += (long) tensorData[i].length * bytesPerElement;
                totalTensorDataSize = (totalTensorDataSize + alignment - 1) & ~(alignment - 1);
            }

            long totalSize = dataStart + totalTensorDataSize;

            MemorySegment segment = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalSize, arena);

            long offset = 0;
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 0x46554747);
            offset += 4;
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 3);
            offset += 4;
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, tensorCount);
            offset += 8;
            segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, kvCount);
            offset += 8;

            offset = writeString(segment, offset, nameKey);
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 8);
            offset += 4;
            offset = writeString(segment, offset, name);

            offset = writeString(segment, offset, archKey);
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 8);
            offset += 4;
            offset = writeString(segment, offset, architecture);

            for (int i = 0; i < tensorCount; i++) {
                offset = writeString(segment, offset, tensorNames[i]);
                segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, 1); // n_dims = 1
                offset += 4;
                segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, tensorData[i].length);
                offset += 8;
                segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, tensorTypes[i]);
                offset += 4;
                segment.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, tensorOffsets[i]);
                offset += 8;
            }

            for (int i = 0; i < tensorCount; i++) {
                long tensorDataStart = dataStart + tensorOffsets[i];
                if (tensorTypes[i] == 1) {
                    // F16
                    for (int j = 0; j < tensorData[i].length; j++) {
                        short f16 = Float.floatToFloat16(tensorData[i][j]);
                        segment.set(ValueLayout.JAVA_SHORT_UNALIGNED, tensorDataStart + (long) j * 2, f16);
                    }
                } else {
                    // F32
                    for (int j = 0; j < tensorData[i].length; j++) {
                        segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, tensorDataStart + (long) j * 4, tensorData[i][j]);
                    }
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
