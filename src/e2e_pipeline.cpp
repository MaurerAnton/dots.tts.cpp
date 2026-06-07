// dots.tts.cpp - End-to-end pipeline
// Text -> LLM -> hidden_proj -> DiT -> AudioVAE -> WAV

#include "dots_tts.h"
#include "dit.h"
#include "patchenc.h"
#include "audiovae.h"
#include "safetensors.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Forward declarations from other modules
bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);
bool load_patchenc_weights(SafeTensorsFile & sf, ggml_context * w_ctx, patch_encoder & enc);

static float * tensor_data(ggml_tensor * t) {
    return (float *)((char *)t->data + t->view_offs);
}

// ===========================================================================
// Main pipeline
// ===========================================================================

int main(int argc, char ** argv) {
    (void)argc; (void)argv;
    setbuf(stdout, NULL); // disable buffering for debug

    printf("dots.tts.cpp - End-to-end TTS pipeline\n");
    printf("======================================\n\n");

    // ---- 1. Load LLM ----
    printf("[1/5] Loading LLM (Qwen2.5-1.5B)...\n");
    const char * llm_path = "/home/bym/dots.tts.cpp/models/llm_qwen25_1.5b.gguf";
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * llm = llama_model_load_from_file(llm_path, mparams);
    if (!llm) { printf("FAILED to load LLM\n"); return 1; }
    printf("  LLM: %d layers, %d hidden, %d vocab\n",
           llama_model_n_layer(llm), llama_model_n_embd(llm),
           llama_vocab_n_tokens(llama_model_get_vocab(llm)));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 256; cparams.n_batch = 256;
    cparams.embeddings = true;
    llama_context * lctx = llama_init_from_model(llm, cparams);

    // Skip safetensors for debugging
    printf("[2/5] SKIPPED (debug)\n");
    const char * text = argc > 1 ? argv[1] : "Hello world";
    printf("[3/5] Tokenizing: '%s'\n", text);
    llama_token tokens[256];
    int n_tokens = llama_tokenize(llama_model_get_vocab(llm), text, strlen(text),
                                  tokens, 256, true, false);
    printf("  %d tokens\n", n_tokens);

    printf("[4/5] LLM decode...\n");
    llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    int r = llama_decode(lctx, batch);
    printf("  Decode: %s\n", r == 0 ? "OK" : "ERROR");

    if (r == 0) {
        int n_embd = llama_model_n_embd(llm);
        printf("  Hidden states:\n");
        for (int i = 0; i < n_tokens; i++) {
            float * emb = llama_get_embeddings_ith(lctx, i);
            if (emb) {
                float sum = 0;
                for (int j = 0; j < n_embd; j++) sum += emb[j];
                printf("    token %d: mean=%.6f\n", i, sum/n_embd);
            }
        }
    }

    llama_free(lctx);
    llama_model_free(llm);
    printf("\nDone.\n");
    return 0;
}
