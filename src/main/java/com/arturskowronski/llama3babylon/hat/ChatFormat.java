package com.arturskowronski.llama3babylon.hat;

import java.util.*;

/**
 * Llama 3 Instruct chat template formatter.
 * Produces token sequences like:
 * {@code <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>}
 * {@code <|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>}
 * {@code <|start_header_id|>assistant<|end_header_id|>\n}
 */
public class ChatFormat {

    private final Tokenizer tokenizer;
    private final int beginOfText;
    private final int startHeader;
    private final int endHeader;
    private final int endOfTurn;
    private final int endOfText;
    private final Set<Integer> stopTokens;

    public ChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        this.beginOfText = special.get("<|begin_of_text|>");
        this.startHeader = special.get("<|start_header_id|>");
        this.endHeader = special.get("<|end_header_id|>");
        this.endOfTurn = special.get("<|eot_id|>");
        this.endOfText = special.get("<|end_of_text|>");
        this.stopTokens = Set.of(endOfText, endOfTurn);
    }

    public Set<Integer> getStopTokens() {
        return stopTokens;
    }

    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(tokenizer.encodeAsList(message.role().name()));
        tokens.add(endHeader);
        tokens.addAll(tokenizer.encodeAsList("\n"));
        return tokens;
    }

    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(List<Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
        for (Message message : dialog) {
            tokens.addAll(encodeMessage(message));
        }
        // Append assistant turn header for model to complete
        tokens.addAll(encodeHeader(new Message(Role.ASSISTANT, "")));
        return tokens;
    }

    public record Message(Role role, String content) {}

    public record Role(String name) {
        public static final Role SYSTEM = new Role("system");
        public static final Role USER = new Role("user");
        public static final Role ASSISTANT = new Role("assistant");

        @Override
        public String toString() { return name; }
    }
}
