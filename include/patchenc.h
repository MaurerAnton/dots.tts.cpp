#pragma once

// dots.tts.cpp - PatchEncoder (VAE Semantic Encoder)
// 24-layer causal streaming transformer encoder
// Input:  audio VAE latent patch [patch_size=4, latent_dim=128]
// Output: LLM-compatible embedding [1536] (Qwen2.5-1.5B hidden size)
//
// Streaming: init_decode_state -> decode_step (incremental)

#include "dots_tts.h"
#include "ggml.h"
#include <vector>

struct patchenc_layer {
    // Self-attention
    ggml_tensor * attn_qkv_weight; // [hidden, 3*hidden]
    ggml_tensor * attn_qkv_bias;
    ggml_tensor * attn_o_weight;   // [hidden, hidden]
    ggml_tensor * attn_o_bias;

    // qk norms (per-head)
    ggml_tensor * q_norm_w;  // [head_dim]
    ggml_tensor * k_norm_w;

    // FFN
    ggml_tensor * ffn_w1;  // [hidden, ffn]
    ggml_tensor * ffn_w2;  // [ffn, hidden]

    // RMS norms (no adaLN — just standard norms)
    ggml_tensor * attn_norm_w;  // [hidden]
    ggml_tensor * ffn_norm_w;
};

struct patch_encoder {
    int n_layers    = PATCHENC_NUM_LAYERS;
    int hidden_size = PATCHENC_HIDDEN;
    int num_heads   = PATCHENC_NUM_HEADS;
    int head_dim    = PATCHENC_HEAD_SIZE;
    int ffn_size    = PATCHENC_FFN_SIZE;
    int latent_dim  = PATCHENC_LATENT_DIM;
    int patch_size  = PATCHENC_PATCH_SIZE;

    std::vector<patchenc_layer> layers;

    // Input conv: causal Conv1d(128 -> 128, kernel=2, stride=2)
    ggml_tensor * conv_weight;  // [out_ch=128, in_ch=128, kernel=2]
    ggml_tensor * conv_bias;

    // Input projection: Linear(128 -> 1024)
    ggml_tensor * in_proj_w;  // [latent_dim, hidden]
    ggml_tensor * in_proj_b;

    // Output projection: Linear(2*hidden -> 1536)
    ggml_tensor * out_proj_w;  // [2*hidden, 1536]
    ggml_tensor * out_proj_b;

    // Final norm
    ggml_tensor * final_norm_w;  // [hidden]
};

// Full forward pass (non-streaming): encodes an entire patch sequence
// x: [n_patches * patch_size, latent_dim] (flattened patches)
// returns: [n_patches, llm_dim=1536]
ggml_tensor * patchenc_forward(
    patch_encoder & enc,
    ggml_context * ctx,
    ggml_tensor * x,
    int n_patches);

// Streaming step: processes one new patch incrementally
// state: persistent streaming state (maintained across calls)
struct patchenc_stream_state;

patchenc_stream_state * patchenc_stream_init(patch_encoder & enc);
void patchenc_stream_free(patchenc_stream_state * st);

// Feed one new patch [patch_size, latent_dim]
// Returns LLM embedding [llm_dim=1536]
void patchenc_decode_step(
    patch_encoder & enc,
    patchenc_stream_state * st,
    ggml_context * ctx,
    float * patch,             // [patch_size * latent_dim] input
    float * llm_emb_out);      // [1536] output
