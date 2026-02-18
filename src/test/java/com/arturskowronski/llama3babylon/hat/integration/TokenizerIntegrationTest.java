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

import static org.junit.jupiter.api.Assertions.*;

@Tag("integration")
@Tag("plain-java")
public class TokenizerIntegrationTest {

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testTokenizerLoadsFromRealModel() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));
        var metadata = GGUFReader.readMetadata(modelPath);
        Tokenizer tokenizer = Tokenizer.fromGGUFMetadata(metadata.metadata());

        assertEquals(LlamaModel.VOCAB_SIZE, tokenizer.vocabularySize());

        // Verify known special tokens
        assertTrue(tokenizer.getSpecialTokens().containsKey("<|begin_of_text|>"));
        assertEquals(128000, tokenizer.getSpecialTokens().get("<|begin_of_text|>"));
        assertEquals(128001, tokenizer.getSpecialTokens().get("<|end_of_text|>"));
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testEncodeDecodeRoundTrip() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));
        var metadata = GGUFReader.readMetadata(modelPath);
        Tokenizer tokenizer = Tokenizer.fromGGUFMetadata(metadata.metadata());

        String text = "Hello, world!";
        int[] encoded = tokenizer.encode(text);
        assertTrue(encoded.length > 0, "Encoded tokens should not be empty");

        String decoded = tokenizer.decode(
                Arrays.stream(encoded).boxed().toList());
        assertEquals(text, decoded, "Round-trip encode/decode should preserve text");
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "LLAMA_FP16_PATH", matches = ".*")
    public void testChatFormatProducesExpectedTokens() throws IOException {
        Path modelPath = Paths.get(System.getenv("LLAMA_FP16_PATH"));
        var metadata = GGUFReader.readMetadata(modelPath);
        Tokenizer tokenizer = Tokenizer.fromGGUFMetadata(metadata.metadata());
        ChatFormat chatFormat = new ChatFormat(tokenizer);

        List<ChatFormat.Message> dialog = List.of(
                new ChatFormat.Message(ChatFormat.Role.SYSTEM, "You are a helpful assistant."),
                new ChatFormat.Message(ChatFormat.Role.USER, "Tell a joke about programming")
        );
        List<Integer> tokens = chatFormat.encodeDialogPrompt(dialog);

        // First token must be begin_of_text (128000)
        assertEquals(128000, tokens.get(0));
        // Must contain start_header_id tokens
        assertTrue(tokens.contains(128006));
        // Must contain end_header_id tokens
        assertTrue(tokens.contains(128007));

        System.out.println("Chat prompt tokens (" + tokens.size() + "): " + tokens);
    }
}
