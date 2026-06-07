// Minimal LLM test — skip all dummy weights
#include "llama.h"
#include <cstdio>
#include <cstring>

int main() {
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level <= 2) fprintf(stderr, "llama: %s", text);
    }, nullptr);

    const char * path = "/home/bym/dots.tts.cpp/models/llm_qwen25_1.5b.gguf";
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    
    printf("Loading LLM from %s...\n", path);
    llama_model * llm = llama_model_load_from_file(path, mparams);
    if (!llm) { printf("FAILED\n"); return 1; }
    
    printf("Loaded. n_vocab=%d n_ctx=%d n_layers=%d n_embd=%d\n",
           llama_vocab_n_tokens(llama_model_get_vocab(llm)),
           llama_model_n_ctx_train(llm),
           llama_model_n_layer(llm),
           llama_model_n_embd(llm));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = 256;
    llama_context * lctx = llama_init_from_model(llm, cparams);

    const char * text = "Privet, how are you?";
    llama_token tokens[256];
    int n = llama_tokenize(llama_model_get_vocab(llm),
                           text, strlen(text), tokens, 256, true, false);
    printf("Tokenized '%s' -> %d tokens:", text, n);
    for (int i = 0; i < n; i++) printf(" %d", tokens[i]);
    printf("\n");

    llama_batch batch = llama_batch_get_one(tokens, n);
    int ret = llama_decode(lctx, batch);
    printf("Decode: %s\n", ret == 0 ? "OK" : ret < 0 ? "ERROR" : "PARTIAL");

    // Get logits
    float * logits = llama_get_logits(lctx);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llm));
    float max_logit = -1e9;
    int max_id = 0;
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] > max_logit) { max_logit = logits[i]; max_id = i; }
    }
    
    char buf[256];
    llama_token_to_piece(llama_model_get_vocab(llm), max_id, buf, sizeof(buf), 0, false);
    printf("Top token: %d ('%s') logit=%.4f\n", max_id, buf, max_logit);

    llama_free(lctx);
    llama_model_free(llm);
    printf("LLM test: PASS\n");
    return 0;
}
