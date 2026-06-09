// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// Quick test: load dots_llm.gguf, tokenize text, print hidden state stats
#include "llama.h"
#include <cstdio>
#include <cmath>
#include <cstring>

int main() {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    
    const char * model_path = "/home/bym/dots.tts.cpp/models/dots_llm.gguf";
    printf("Loading %s...\n", model_path);
    
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { printf("FAILED to load\n"); return 1; }
    
    int n_embd = llama_model_n_embd(model);
    printf("Model loaded: n_embd=%d\n", n_embd);
    
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = 128;
    cp.embeddings = true;
    
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { printf("FAILED to create context\n"); return 1; }
    
    // Try tokenization (will fail without tokenizer data, but let's try)
    const char * text = "hello";
    
    // Try to get vocab
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) { printf("No vocab\n"); }
    
    int n_tokens = llama_tokenize(vocab, text, strlen(text), nullptr, 0, true, false);
    printf("Tokenize '%s': would produce %d tokens\n", text, n_tokens);
    
    llama_free(ctx);
    llama_model_free(model);
    
    printf("Done.\n");
    return 0;
}
