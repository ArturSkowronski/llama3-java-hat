package com.arturskowronski.llama3babylon.hat;

import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * GPT-2 style byte-level BPE tokenizer for Llama 3.
 *
 * Reads vocabulary and merge rules from GGUF metadata.
 * Based on the approach from mukel/llama3.java and karpathy/minbpe.
 */
public class Tokenizer {

    private static final String LLAMA_3_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|" +
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private final Pattern compiledPattern;
    private final String[] vocabulary;
    private final Map<String, Integer> tokenToIndex;
    private final Map<Pair, Integer> merges;       // pair → merged token ID
    private final Map<Pair, Integer> mergeRanks;   // pair → rank (lower = higher priority)
    private final Map<String, Integer> specialTokens;

    private static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    private static final Map<Integer, Integer> BYTE_DECODER;
    static {
        BYTE_DECODER = new HashMap<>();
        BYTE_ENCODER.forEach((k, v) -> BYTE_DECODER.put(v, k));
    }

    record Pair(int first, int second) {}

    private Tokenizer(String[] vocabulary, Map<String, Integer> tokenToIndex,
                      Map<Pair, Integer> merges, Map<Pair, Integer> mergeRanks,
                      Map<String, Integer> specialTokens) {
        this.vocabulary = vocabulary;
        this.tokenToIndex = tokenToIndex;
        this.merges = merges;
        this.mergeRanks = mergeRanks;
        this.specialTokens = specialTokens;
        this.compiledPattern = Pattern.compile(LLAMA_3_PATTERN);
    }

    @SuppressWarnings("unchecked")
    public static Tokenizer fromGGUFMetadata(Map<String, Object> metadata) {
        String model = (String) metadata.get("tokenizer.ggml.model");
        if (!"gpt2".equals(model)) {
            throw new IllegalArgumentException("Expected gpt2 tokenizer model, got: " + model);
        }

        // Load vocabulary
        List<Object> tokenList = (List<Object>) metadata.get("tokenizer.ggml.tokens");
        String[] vocabulary = tokenList.stream()
                .map(Object::toString)
                .toArray(String[]::new);

        // Build token-to-index map
        Map<String, Integer> tokenToIndex = new HashMap<>(vocabulary.length * 2);
        for (int i = 0; i < vocabulary.length; i++) {
            tokenToIndex.put(vocabulary[i], i);
        }

        // Load and parse merge rules
        List<Object> mergeList = (List<Object>) metadata.get("tokenizer.ggml.merges");
        Map<Pair, Integer> mergeMap = new HashMap<>(mergeList.size() * 2);
        Map<Pair, Integer> rankMap = new HashMap<>(mergeList.size() * 2);
        for (int i = 0; i < mergeList.size(); i++) {
            String mergeRule = mergeList.get(i).toString();
            int spaceIdx = mergeRule.indexOf(' ');
            String first = mergeRule.substring(0, spaceIdx);
            String second = mergeRule.substring(spaceIdx + 1);

            Integer firstId = tokenToIndex.get(first);
            Integer secondId = tokenToIndex.get(second);
            String merged = first + second;
            Integer mergedId = tokenToIndex.get(merged);

            if (firstId != null && secondId != null && mergedId != null) {
                Pair pair = new Pair(firstId, secondId);
                mergeMap.put(pair, mergedId);
                rankMap.put(pair, i);
            }
        }

        // Build special tokens map (IDs 128000+)
        Map<String, Integer> specialTokens = new HashMap<>();
        for (int i = 128000; i < vocabulary.length; i++) {
            specialTokens.put(vocabulary[i], i);
        }

        return new Tokenizer(vocabulary, tokenToIndex, mergeMap, rankMap, specialTokens);
    }

    public int[] encode(String text) {
        List<Integer> tokens = encodeAsList(text);
        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }

    public List<Integer> encodeAsList(String text) {
        List<Integer> result = new ArrayList<>();
        Matcher matcher = compiledPattern.matcher(text);
        while (matcher.find()) {
            String chunk = matcher.group();
            result.addAll(encodeChunk(chunk));
        }
        return result;
    }

    private List<Integer> encodeChunk(String chunk) {
        // Convert each character to byte-encoded unicode representation
        byte[] bytes = chunk.getBytes(StandardCharsets.UTF_8);
        List<Integer> ids = new ArrayList<>(bytes.length);
        for (byte b : bytes) {
            int byteVal = b & 0xFF;
            int unicodeChar = BYTE_ENCODER.get(byteVal);
            String charStr = String.valueOf((char) unicodeChar);
            Integer tokenId = tokenToIndex.get(charStr);
            if (tokenId != null) {
                ids.add(tokenId);
            }
        }

        if (ids.size() < 2) {
            return ids;
        }

        // BPE merge loop
        while (ids.size() >= 2) {
            // Find the pair with the lowest merge rank
            Pair bestPair = null;
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i < ids.size() - 1; i++) {
                Pair pair = new Pair(ids.get(i), ids.get(i + 1));
                Integer rank = mergeRank(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestPair = pair;
                }
            }

            if (bestPair == null) {
                break; // No more merges possible
            }

            // Apply the merge
            int mergedId = merges.get(bestPair);
            List<Integer> newIds = new ArrayList<>(ids.size());
            int i = 0;
            while (i < ids.size()) {
                if (i < ids.size() - 1 && ids.get(i) == bestPair.first() && ids.get(i + 1) == bestPair.second()) {
                    newIds.add(mergedId);
                    i += 2;
                } else {
                    newIds.add(ids.get(i));
                    i++;
                }
            }
            ids = newIds;
        }

        return ids;
    }

    private Integer mergeRank(Pair pair) {
        return mergeRanks.get(pair);
    }

    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int tokenId : tokens) {
            if (tokenId >= 0 && tokenId < vocabulary.length) {
                sb.append(vocabulary[tokenId]);
            }
        }

        // Decode GPT-2 byte encoding back to UTF-8
        String encoded = sb.toString();
        byte[] bytes = new byte[encoded.length()];
        int byteCount = 0;
        for (int i = 0; i < encoded.length(); i++) {
            int ch = encoded.charAt(i);
            Integer byteVal = BYTE_DECODER.get(ch);
            if (byteVal != null) {
                bytes[byteCount++] = (byte) (int) byteVal;
            } else {
                // Non-mapped character (e.g., special token text) — encode as UTF-8 directly
                byte[] charBytes = String.valueOf((char) ch).getBytes(StandardCharsets.UTF_8);
                if (byteCount + charBytes.length > bytes.length) {
                    bytes = Arrays.copyOf(bytes, bytes.length * 2);
                }
                System.arraycopy(charBytes, 0, bytes, byteCount, charBytes.length);
                byteCount += charBytes.length;
            }
        }
        return new String(bytes, 0, byteCount, StandardCharsets.UTF_8);
    }

    public String decodeToken(int tokenId) {
        return decode(List.of(tokenId));
    }

    public boolean isSpecialToken(int tokenId) {
        return tokenId >= 128000 && tokenId < vocabulary.length;
    }

    public Map<String, Integer> getSpecialTokens() {
        return Collections.unmodifiableMap(specialTokens);
    }

    public int vocabularySize() {
        return vocabulary.length;
    }

    /**
     * GPT-2 byte-to-unicode mapping.
     * Maps all 256 byte values to printable unicode code points.
     */
    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        List<Integer> cs = new ArrayList<>();

        // Printable ASCII and Latin-1 supplement ranges
        for (int i = '!'; i <= '~'; i++) { bs.add(i); cs.add(i); }
        for (int i = 0xA1; i <= 0xAC; i++) { bs.add(i); cs.add(i); }
        for (int i = 0xAE; i <= 0xFF; i++) { bs.add(i); cs.add(i); }

        // Remaining byte values mapped to higher unicode
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }

        Map<Integer, Integer> result = new HashMap<>(512);
        for (int i = 0; i < bs.size(); i++) {
            result.put(bs.get(i), cs.get(i));
        }
        return result;
    }
}
