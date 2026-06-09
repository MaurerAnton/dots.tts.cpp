// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// dots.tts.cpp - DiT forward pass (simplified for ggml)
// All operations use either:
//   1. ggml_mul_mat for linear projections (2D: [out_features, batch*seq])
//   2. Element-wise ops on correctly shaped tensors
//   3. ggml_rms_norm, ggml_silu, ggml_rope, ggml_soft_max

#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <cstdio>

// Multi-head attention (defined in dit_attention.cpp)
ggml_tensor * dit_attention_multihead(
    ggml_context * ctx, ggml_tensor * x,
    ggml_tensor * q_weight, ggml_tensor * k_weight, ggml_tensor * v_weight,
    ggml_tensor * o_weight,
    ggml_tensor * q_bias, ggml_tensor * k_bias, ggml_tensor * v_bias,
    ggml_tensor * o_bias,
    int seq_len, int n_batch,
    int n_heads, int head_dim, ggml_tensor * q_norm_w, ggml_tensor * k_norm_w);

// ===========================================================================
// Timestep embedding: sinusoidal features (diffusers convention: sin, then cos)
// ===========================================================================

ggml_tensor * dit_timestep_embedding(ggml_context * ctx, ggml_tensor * t, int dim) {
    int half_dim = dim / 2;
    int n_batch = t->ne[0];

    ggml_tensor * emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n_batch);
    float * ed = tensor_data(emb);

    float * td = tensor_data(t);
    for (int b = 0; b < n_batch; b++) {
        float t_val = td[b];
        for (int i = 0; i < half_dim; i++) {
            float freq = expf(-logf(10000.0f) * (float)i / (float)(half_dim > 1 ? half_dim - 1 : 1));
            float arg = t_val * freq;
            ed[b * dim + i] = sinf(arg);
            ed[b * dim + half_dim + i] = cosf(arg);
        }
    }

    return emb;
}

// ===========================================================================
// FFN: two linear layers with SiLU in between, with biases
// ===========================================================================

static ggml_tensor * dit_ffn(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w1,
    ggml_tensor * w2,
    ggml_tensor * b1,
    ggml_tensor * b2
) {
    ggml_tensor * h = ggml_mul_mat(ctx, w1, x);
    if (b1) h = ggml_add(ctx, h, b1);
    h = ggml_silu(ctx, h);
    h = ggml_mul_mat(ctx, w2, h);
    if (b2) h = ggml_add(ctx, h, b2);
    return h;
}

// ===========================================================================
// DiT Block forward
// ===========================================================================

static ggml_tensor * dit_block_forward_simple(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * cond,
    dit_block & block,
    int seq_len,
    int n_batch
) {
    int hidden = DIT_HIDDEN_SIZE;
    int n_tokens = seq_len * n_batch;

    // adaLN: cond @ adaln_linear_w + bias -> [6*hidden, n_batch]
    ggml_tensor * adaln = ggml_mul_mat(ctx, block.adaln_linear_w, cond);
    if (block.adaln_linear_b) adaln = ggml_add(ctx, adaln, block.adaln_linear_b);

    // Split into 6 modulations, each [hidden, n_batch]
    int stride = hidden;
    ggml_tensor * shift_msa = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 0);
    ggml_tensor * scale_msa = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], stride * sizeof(float));
    ggml_tensor * gate_msa  = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 2 * stride * sizeof(float));
    ggml_tensor * shift_mlp = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 3 * stride * sizeof(float));
    ggml_tensor * scale_mlp = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 4 * stride * sizeof(float));
    ggml_tensor * gate_mlp  = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 5 * stride * sizeof(float));

    // Broadcast modulations from [hidden, n_batch] -> [hidden, n_tokens]
    auto broadcast_batch_to_tokens = [&](ggml_tensor * mod) -> ggml_tensor * {
        ggml_tensor * out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        float * src = tensor_data(mod);
        float * dst = tensor_data(out);
        for (int s = 0; s < seq_len; s++)
            for (int b = 0; b < n_batch; b++)
                memcpy(dst + (b * seq_len + s) * hidden, src + b * hidden, hidden * sizeof(float));
        return out;
    };

    ggml_tensor * shift_msa_tok = broadcast_batch_to_tokens(shift_msa);
    ggml_tensor * scale_msa_tok = broadcast_batch_to_tokens(scale_msa);
    ggml_tensor * gate_msa_tok  = broadcast_batch_to_tokens(gate_msa);
    ggml_tensor * shift_mlp_tok = broadcast_batch_to_tokens(shift_mlp);
    ggml_tensor * scale_mlp_tok = broadcast_batch_to_tokens(scale_mlp);
    ggml_tensor * gate_mlp_tok  = broadcast_batch_to_tokens(gate_mlp);

    // --- Self-attention sub-layer ---
    ggml_tensor * h = ggml_rms_norm(ctx, x, 1e-6f);
    if (block.attn_norm_w) h = ggml_mul(ctx, h, block.attn_norm_w);
    ggml_tensor * ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    for (int i = 0; i < hidden * n_tokens; i++) ((float*)tensor_data(ones))[i] = 1.0f;
    ggml_tensor * scale_p1 = ggml_add(ctx, ones, scale_msa_tok);
    h = ggml_mul(ctx, scale_p1, h);
    h = ggml_add(ctx, h, shift_msa_tok);

    // Multi-head attention with biases
    ggml_tensor * attn = dit_attention_multihead(ctx, h,
        block.attn_q_weight, block.attn_k_weight, block.attn_v_weight,
        block.attn_o_weight,
        nullptr, nullptr, nullptr,  // Q/K/V biases not in dit_block struct
        block.attn_o_bias,
        seq_len, n_batch, DIT_NUM_HEADS, DIT_HEAD_SIZE,
        block.q_norm_w, block.k_norm_w);

    x = ggml_add(ctx, x, ggml_mul(ctx, gate_msa_tok, attn));

    // --- FFN sub-layer ---
    h = ggml_rms_norm(ctx, x, 1e-6f);
    if (block.ffn_norm_w) h = ggml_mul(ctx, h, block.ffn_norm_w);
    scale_p1 = ggml_add(ctx, ones, scale_mlp_tok);
    h = ggml_mul(ctx, scale_p1, h);
    h = ggml_add(ctx, h, shift_mlp_tok);

    ggml_tensor * ffn = dit_ffn(ctx, h, block.ffn_w1, block.ffn_w2, block.ffn_b1, block.ffn_b2);
    x = ggml_add(ctx, x, ggml_mul(ctx, gate_mlp_tok, ffn));

    return x;
}

// ===========================================================================
// Full DiT forward
// ===========================================================================

ggml_tensor * dit_forward(
    dit_model & model,
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * t,
    ggml_tensor * speaker_emb)
{
    int seq_len = x->ne[0];
    int n_batch = x->ne[1];
    int hidden  = DIT_HIDDEN_SIZE;
    int n_tokens = seq_len * n_batch;

    // Flatten to [hidden, n_tokens]
    x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
    x = ggml_reshape_2d(ctx, x, hidden, n_tokens);

    // Timestep embedding: sinusoidal -> MLP
    ggml_tensor * t_sin = dit_timestep_embedding(ctx, t, 256);

    // time_embedder MLP: 256 -> 1024 -> 1024
    ggml_tensor * t_mlp = ggml_mul_mat(ctx, model.t_embed_w1, t_sin);
    if (model.t_embed_b1) t_mlp = ggml_add(ctx, t_mlp, model.t_embed_b1);
    t_mlp = ggml_silu(ctx, t_mlp);
    t_mlp = ggml_mul_mat(ctx, model.t_embed_w2, t_mlp);
    if (model.t_embed_b2) t_mlp = ggml_add(ctx, t_mlp, model.t_embed_b2);

    // Speaker embedding via xvec_proj
    ggml_tensor * cond = t_mlp;
    if (speaker_emb && model.spk_proj_w1) {
        ggml_tensor * spk = ggml_mul_mat(ctx, model.spk_proj_w1, speaker_emb);
        if (model.spk_proj_b1) spk = ggml_add(ctx, spk, model.spk_proj_b1);
        spk = ggml_silu(ctx, spk);
        cond = ggml_add(ctx, cond, spk);
    }

    // SiLU on combined conditioning
    cond = ggml_silu(ctx, cond);

    // Input layer
    if (model.input_layer_w) {
        x = ggml_mul_mat(ctx, model.input_layer_w, x);
        if (model.input_layer_b) x = ggml_add(ctx, x, model.input_layer_b);
    }

    for (int i = 0; i < model.n_layers; i++) {
        x = dit_block_forward_simple(ctx, x, cond, model.layers[i], seq_len, n_batch);
    }

    // Output: adaLN modulation + linear projection
    ggml_tensor * out = x;
    if (model.out_adaln_w) {
        ggml_tensor * mod = ggml_mul_mat(ctx, model.out_adaln_w, cond);
        if (model.out_adaln_b) mod = ggml_add(ctx, mod, model.out_adaln_b);

        int stride = DIT_HIDDEN_SIZE;
        ggml_tensor * shift = ggml_view_2d(ctx, mod, DIT_HIDDEN_SIZE, n_batch, mod->nb[1], 0);
        ggml_tensor * scale = ggml_view_2d(ctx, mod, DIT_HIDDEN_SIZE, n_batch, mod->nb[1], stride * sizeof(float));

        ggml_tensor * shift_tok = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, n_tokens);
        ggml_tensor * scale_tok = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, n_tokens);
        {
            float * ss = tensor_data(shift), * sd = tensor_data(shift_tok);
            float * cs = tensor_data(scale), * cd = tensor_data(scale_tok);
            for (int s = 0; s < seq_len; s++)
                for (int b = 0; b < n_batch; b++) {
                    memcpy(sd + (b * seq_len + s) * DIT_HIDDEN_SIZE,
                           ss + b * DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE * sizeof(float));
                    memcpy(cd + (b * seq_len + s) * DIT_HIDDEN_SIZE,
                           cs + b * DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE * sizeof(float));
                }
        }

        out = ggml_rms_norm(ctx, out, 1e-6f);
        ggml_tensor * ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, n_tokens);
        for (int i = 0; i < DIT_HIDDEN_SIZE * n_tokens; i++) ((float*)tensor_data(ones))[i] = 1.0f;
        ggml_tensor * scale_1 = ggml_add(ctx, ones, scale_tok);
        out = ggml_mul(ctx, out, scale_1);
        out = ggml_add(ctx, out, shift_tok);
    }
    out = ggml_mul_mat(ctx, model.out_proj_w, out);
    if (model.out_proj_b) out = ggml_add(ctx, out, model.out_proj_b);

    return out;
}
