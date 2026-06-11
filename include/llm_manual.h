// Manual LLM forward pass declarations (implementation in src/llm_manual.cpp)
#pragma once
#include "safetensors.h"

struct LLMWeights {
    float *embed_w;
    struct Layer {
        float *an, *fn;
        float *qw,*qb, *kw,*kb, *vw,*vb, *ow,*ob;
        float *gw, *uw, *dw;
    };
    Layer * layers;
    float * final_norm;
    int n_layers;
};

// KV cache for autoregressive decoding
struct LLMKVCache {
    int n_layers;       // 28
    int n_kv_heads;     // 2
    int head_dim;        // 128
    int seq_len;         // current cached length
    int max_seq;         // allocated capacity
    float ** k_cache;    // [n_layers] each: [n_kv_heads * max_seq * head_dim]
    float ** v_cache;    // [n_layers] each: [n_kv_heads * max_seq * head_dim]
};

bool load_llm_weights(SafeTensorsFile & sf, LLMWeights & w, int n_layers);
void free_llm_weights(LLMWeights & w);

// Full forward pass (used for initial prompt)
void llm_manual_forward(const LLMWeights & w, const int * token_ids, int n_tok, float * hiddens_out);

// Initialize KV cache by processing the initial prompt
// Returns the final hidden states in hiddens_out, fills the KV cache
void llm_kv_cache_init(const LLMWeights & w, const int * token_ids, int n_tok,
    float * hiddens_out, LLMKVCache & cache);

// Process a single embedding through the LLM using KV cache
// input_emb: [1536] single embedding vector
// output: [1536] new hidden state (after final norm)
void llm_kv_cache_step(const LLMWeights & w, const float * input_emb,
    LLMKVCache & cache, float * hidden_out);

void llm_kv_cache_free(LLMKVCache & cache);
