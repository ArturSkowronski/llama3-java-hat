package com.arturskowronski.llama3babylon.hat;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GGUFReader {

    public enum GGUFValueType {
        UINT8(0), INT8(1), UINT16(2), INT16(3), UINT32(4), INT32(5), FLOAT32(6), BOOL(7),
        STRING(8), ARRAY(9), UINT64(10), INT64(11), FLOAT64(12);

        private final int id;
        GGUFValueType(int id) { this.id = id; }
        public static GGUFValueType fromId(int id) {
            for (GGUFValueType v : values()) if (v.id == id) return v;
            throw new IllegalArgumentException("Unknown GGUF value type: " + id);
        }
    }

    public record GGUFMetadata(int version, long tensorCount, long metadataKVCount, Map<String, Object> metadata, List<GGUFTensorInfo> tensors, long dataStartOffset) {}

    public record GGUFTensorInfo(String name, int dimensions, long[] shape, int type, long offset) {
        public long size() {
            long count = 1;
            for (long d : shape) count *= d;

            return switch (type) {
                case 0 -> count * 4; // F32
                case 1 -> count * 2; // F16
                case 2 -> (count / 32) * 18; // Q4_0
                case 3 -> (count / 32) * 20; // Q4_1
                case 6 -> (count / 32) * 22; // Q5_0
                case 7 -> (count / 32) * 24; // Q5_1
                case 8 -> (count / 32) * 34; // Q8_0
                case 9 -> (count / 32) * 36; // Q8_1
                case 10 -> (count / 256) * 84; // Q2_K
                case 11 -> (count / 256) * 110; // Q3_K
                case 12 -> (count / 256) * 144; // Q4_K
                case 13 -> (count / 256) * 176; // Q5_K
                case 14 -> (count / 256) * 210; // Q6_K
                case 15 -> (count / 256) * 256; // Q8_K? No.
                default -> 0;
            };
        }
    }

    public static GGUFMetadata readMetadata(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
             Arena arena = Arena.ofConfined()) {
            MemorySegment segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), arena);
            
            long offset = 0;
            int magic = segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
            offset += 4;
            if (magic != 0x46554747) { // "GGUF" in little-endian
                throw new IOException("Not a GGUF file or wrong magic: " + Integer.toHexString(magic));
            }

            int version = segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
            offset += 4;
            
            long tensorCount = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
            offset += 8;
            
            long kvCount = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
            offset += 8;

            Map<String, Object> metadata = new HashMap<>();
            for (int i = 0; i < kvCount; i++) {
                String key = readString(segment, offset);
                offset += 8 + key.getBytes(StandardCharsets.UTF_8).length;
                
                int typeId = segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
                offset += 4;
                
                Object value = readValue(segment, offset, GGUFValueType.fromId(typeId));
                offset = updateOffsetAfterValue(segment, offset, GGUFValueType.fromId(typeId), value);
                metadata.put(key, value);
            }

            List<GGUFTensorInfo> tensors = new ArrayList<>();
            for (int i = 0; i < tensorCount; i++) {
                String name = readString(segment, offset);
                offset += 8 + name.getBytes(StandardCharsets.UTF_8).length;

                int n_dims = segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
                offset += 4;

                long[] shape = new long[n_dims];
                for (int d = 0; d < n_dims; d++) {
                    shape[d] = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
                    offset += 8;
                }

                int type = segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
                offset += 4;

                long tensorOffset = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
                offset += 8;

                tensors.add(new GGUFTensorInfo(name, n_dims, shape, type, tensorOffset));
            }

            long alignment = 32;
            Object alignmentVal = metadata.get("general.alignment");
            if (alignmentVal instanceof Number n) {
                alignment = n.longValue();
            }
            long dataStartOffset = (offset + alignment - 1) & ~(alignment - 1);

            return new GGUFMetadata(version, tensorCount, kvCount, metadata, tensors, dataStartOffset);
        }
    }

    private static String readString(MemorySegment segment, long offset) {
        long length = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
        byte[] bytes = new byte[(int) length];
        MemorySegment.copy(segment, ValueLayout.JAVA_BYTE, offset + 8, bytes, 0, (int) length);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static Object readValue(MemorySegment segment, long offset, GGUFValueType type) {
        return switch (type) {
            case UINT8 -> segment.get(ValueLayout.JAVA_BYTE, offset) & 0xFF;
            case INT8 -> segment.get(ValueLayout.JAVA_BYTE, offset);
            case UINT16 -> segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset) & 0xFFFF;
            case INT16 -> segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
            case UINT32 -> segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset) & 0xFFFFFFFFL;
            case INT32 -> segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
            case FLOAT32 -> segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
            case BOOL -> segment.get(ValueLayout.JAVA_BYTE, offset) != 0;
            case STRING -> readString(segment, offset);
            case UINT64, INT64 -> segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
            case FLOAT64 -> segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset);
            case ARRAY -> {
                int itemTypeId = segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
                long count = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset + 4);
                List<Object> items = new ArrayList<>();
                long itemOffset = offset + 12;
                GGUFValueType itemType = GGUFValueType.fromId(itemTypeId);
                for (int i = 0; i < count; i++) {
                    Object item = readValue(segment, itemOffset, itemType);
                    items.add(item);
                    itemOffset = updateOffsetAfterValue(segment, itemOffset, itemType, item);
                }
                yield items;
            }
        };
    }

    private static long updateOffsetAfterValue(MemorySegment segment, long offset, GGUFValueType type, Object value) {
        return switch (type) {
            case UINT8, INT8, BOOL -> offset + 1;
            case UINT16, INT16 -> offset + 2;
            case UINT32, INT32, FLOAT32 -> offset + 4;
            case UINT64, INT64, FLOAT64 -> offset + 8;
            case STRING -> offset + 8 + ((String) value).getBytes(StandardCharsets.UTF_8).length;
            case ARRAY -> {
                // The ARRAY readValue already calculated the end, but we need to return it here.
                // This is a bit redundant but for simplicity in this minimal reader:
                int itemTypeId = segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
                long count = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset + 4);
                long current = offset + 12;
                GGUFValueType itemType = GGUFValueType.fromId(itemTypeId);
                List<?> list = (List<?>) value;
                for (Object item : list) {
                    current = updateOffsetAfterValue(segment, current, itemType, item);
                }
                yield current;
            }
        };
    }

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.out.println("Usage: GGUFReader <path_to_gguf>");
            return;
        }
        Path path = Path.of(args[0]);
        GGUFMetadata metadata = readMetadata(path);
        System.out.println("GGUF Version: " + metadata.version());
        System.out.println("Tensors: " + metadata.tensorCount());
        System.out.println("KV Pairs: " + metadata.metadataKVCount());
        
        metadata.metadata().forEach((k, v) -> {
            if (v instanceof List<?> list) {
                System.out.println(k + ": Array[" + list.size() + "]");
            } else {
                System.out.println(k + ": " + v);
            }
        });

        System.out.println("\nTensors:");
        for (int i = 0; i < Math.min(metadata.tensors().size(), 10); i++) {
            GGUFTensorInfo t = metadata.tensors().get(i);
            System.out.printf("%s: type=%d, shape=%s, offset=%d\n", 
                t.name(), t.type(), java.util.Arrays.toString(t.shape()), t.offset());
        }
        if (metadata.tensors().size() > 10) {
            System.out.println("... and " + (metadata.tensors().size() - 10) + " more");
        }
        System.out.println("Data Start Offset: " + metadata.dataStartOffset());
    }
}
