#pragma once

// dots.tts.cpp - DiT (Diffusion Transformer) for flow matching
// 18 modulated DiTBlocks with adaLN: timestep + speaker x-vector -> shift/scale/gate
// Predicts velocity field v_t for ODE-based flow matching

#include "dots_tts.h"
#include "ggml.h"
#include <vector>
#include <string>

struct dit_layer {
    // attention weights
    ggml_tensor * attn_q_weight;  // [hidden, hidden]
    ggml_tensor * attn_q_bias;
    ggml_tensor * attn_k_weight;
    ggml_tensor * attn_k_bias;
    ggml_tensor * attn_v_weight;
    ggml_tensor * attn_v_bias;
    ggml_tensor * attn_o_weight;
    ggml_tensor * attn_o_bias;

    // qk layer norms
    ggml_tensor * attn_q_norm_w;
    ggml_tensor * attn_k_norm_w;

    // ffn
    ggml_tensor * ffn_w1;  // [hidden, ffn]
    ggml_tensor * ffn_w2;  // [ffn, hidden]
    ggml_tensor * ffn_b1;
    ggml_tensor * ffn_b2;

    // adaLN modulation projection (input: timestep_emb + speaker_xvec)
    ggml_tensor * adaln_w1;  // [ada_dim, 6*hidden]
    ggml_tensor * adaln_w2;  // [ada_dim, 6*hidden]
    ggml_tensor * adaln_b1;
    ggml_tensor * adaln_b2;

    // two norm layers (pre-attn, pre-ffn)
    ggml_tensor * norm1_w;   // [hidden] rms/rms norm
    ggml_tensor * norm2_w;
};

struct dit_block {
    // attention weights
    ggml_tensor * attn_qkv_weight; // [hidden, 3*hidden]
    ggml_tensor * attn_qkv_bias;
    ggml_tensor * attn_o_weight;   // [hidden, hidden]
    ggml_tensor * attn_o_bias;

    // qk norms (per-head)
    ggml_tensor * q_norm_w;  // [head_dim]
    ggml_tensor * k_norm_w;  // [head_dim]

    // ffn
    ggml_tensor * ffn_w1;  // [hidden, ffn]  first linear
    ggml_tensor * ffn_w2;  // [ffn, hidden]  second linear

    // adaLN modulation: Linear(ada_dim -> 6*hidden)
    ggml_tensor * adaln_linear_w;  // [ada_dim, 6*hidden]
    ggml_tensor * adaln_linear_b;  // [6*hidden]

    // RMS norms
    ggml_tensor * attn_norm_w;  // [hidden]
    ggml_tensor * ffn_norm_w;   // [hidden]
};

struct dit_model {
    int n_layers     = DIT_NUM_LAYERS;
    int hidden_size  = DIT_HIDDEN_SIZE;
    int num_heads    = DIT_NUM_HEADS;
    int head_dim     = DIT_HEAD_SIZE;
    int ffn_size     = DIT_FFN_SIZE;
    int ada_dim      = DIT_ADA_DIM;
    int speaker_dim  = DIT_SPEAKER_DIM;

    std::vector<dit_block> layers;

    // input projections
    ggml_tensor * cond_in_proj;   // [cond_hidden -> hidden]
    ggml_tensor * noise_in_proj;  // [latent_dim -> hidden]

    // timestep embedding
    ggml_tensor * t_embed_w1;  // [freq_dim, ada_dim]
    ggml_tensor * t_embed_w2;  // [ada_dim, ada_dim]

    // speaker projection
    ggml_tensor * spk_proj_w;  // [speaker_dim, ada_dim]

    // final output projection
    ggml_tensor * out_proj_w;  // [hidden, latent_dim]
    ggml_tensor * out_proj_b;  // [latent_dim]

    // compute buffer (KV cache for self-attn, etc.)
    ggml_context * ctx = nullptr;
    ggml_backend_buffer * buf = nullptr;
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

// load model weights from a ggml context (already populated with tensors)
bool dit_model_load(dit_model & model, ggml_context * ctx);

// build the forward pass graph
// x: input sequence [seq_len, n_batch, hidden] — conditioning + noise targets
// t: timestep [1, n_batch] — scalar in [0, 1]
// speaker_emb: speaker x-vector [speaker_dim, n_batch] or nullptr for unconditional
// returns: velocity field [seq_len, latent_dim]
ggml_tensor * dit_forward(
    dit_model & model,
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * t,
    ggml_tensor * speaker_emb);

// timestep embedding: scalar t -> [ada_dim] sinusoidal
ggml_tensor * dit_timestep_embedding(ggml_context * ctx, ggml_tensor * t, int dim);
