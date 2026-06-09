// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// dots.tts.cpp - DiT forward pass for ggml
// All biases applied. Time embedder computed manually.

#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <cstdio>

ggml_tensor * dit_attention_multihead(
    ggml_context * ctx, ggml_tensor * x,
    ggml_tensor * q_weight, ggml_tensor * k_weight, ggml_tensor * v_weight,
    ggml_tensor * o_weight,
    ggml_tensor * q_bias, ggml_tensor * k_bias, ggml_tensor * v_bias,
    ggml_tensor * o_bias,
    int seq_len, int n_batch,
    int n_heads, int head_dim, ggml_tensor * q_norm_w, ggml_tensor * k_norm_w);

// ===========================================================================
// Manual time_embedder: sinusoidal(t,256) -> Linear(256->1024)+bias -> SiLU -> Linear(1024->1024)+bias
// Computed in pure C++ because ggml graph was not computing the time_embedder branch.
// ===========================================================================

static ggml_tensor * compute_time_embedding_manual(
    ggml_context * ctx, ggml_tensor * t_tensor, dit_model & model)
{
    int n_batch = t_tensor->ne[0];
    float t_val = ((float*)t_tensor->data)[0];
    int emb_dim = 256, half = emb_dim / 2;

    // Sinusoidal embedding
    float sin_emb[256];
    for (int i = 0; i < half; i++) {
        float freq = expf(-logf(10000.0f) * (float)i / (float)(half - 1));
        float arg = t_val * freq;
        sin_emb[i] = sinf(arg);
        sin_emb[half + i] = cosf(arg);
    }

    // MLP layer 1: Linear(256 -> 1024) + bias
    float h1[1024];
    {
        float * w1 = tensor_data(model.t_embed_w1);
        float * b1 = model.t_embed_b1 ? tensor_data(model.t_embed_b1) : nullptr;
        // w1: ggml ne=[256,1024]; PyTorch W[o,i] = ggml_W[i + o*256]
        for (int o = 0; o < 1024; o++) {
            float s = b1 ? b1[o] : 0.0f;
            for (int i = 0; i < emb_dim; i++) s += w1[i + o * 256] * sin_emb[i];
            h1[o] = s;
        }
    }

    // SiLU
    for (int i = 0; i < 1024; i++) { float x = h1[i]; h1[i] = x / (1.0f + expf(-x)); }

    // MLP layer 2: Linear(1024 -> 1024) + bias
    float h2[1024];
    {
        float * w2 = tensor_data(model.t_embed_w2);
        float * b2 = model.t_embed_b2 ? tensor_data(model.t_embed_b2) : nullptr;
        // w2: ggml ne=[1024,1024]; PyTorch W[o,i] = ggml_W[o*1024+i]
        for (int o = 0; o < 1024; o++) {
            float s = b2 ? b2[o] : 0.0f;
            for (int i = 0; i < 1024; i++) s += w2[o * 1024 + i] * h1[i];
            h2[o] = s;
        }
    }

    ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, n_batch);
    memcpy(tensor_data(result), h2, DIT_HIDDEN_SIZE * n_batch * sizeof(float));
    return result;
}

// ===========================================================================
// Timestep embedding (kept for reference)
// ===========================================================================

ggml_tensor * dit_timestep_embedding(ggml_context * ctx, ggml_tensor * t, int dim) {
    int half_dim = dim / 2, n_batch = t->ne[0];
    ggml_tensor * emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n_batch);
    float * ed = tensor_data(emb), * td = tensor_data(t);
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
// FFN
// ===========================================================================

static ggml_tensor * dit_ffn(ggml_context * ctx, ggml_tensor * x,
    ggml_tensor * w1, ggml_tensor * w2, ggml_tensor * b1, ggml_tensor * b2)
{
    ggml_tensor * h = ggml_mul_mat(ctx, w1, x);
    if (b1) h = ggml_add(ctx, h, b1);
    h = ggml_silu(ctx, h);
    h = ggml_mul_mat(ctx, w2, h);
    if (b2) h = ggml_add(ctx, h, b2);
    return h;
}

// ===========================================================================
// DiT Block
// ===========================================================================

static ggml_tensor * dit_block_forward_simple(ggml_context * ctx,
    ggml_tensor * x, ggml_tensor * cond, dit_block & block, int seq_len, int n_batch)
{
    int hidden = DIT_HIDDEN_SIZE, n_tokens = seq_len * n_batch;

    ggml_tensor * adaln = ggml_mul_mat(ctx, block.adaln_linear_w, cond);
    if (block.adaln_linear_b) adaln = ggml_add(ctx, adaln, block.adaln_linear_b);

    int stride = hidden;
    ggml_tensor * sm = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 0);
    ggml_tensor * scm = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], stride * sizeof(float));
    ggml_tensor * gm  = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 2 * stride * sizeof(float));
    ggml_tensor * sml = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 3 * stride * sizeof(float));
    ggml_tensor * scl = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 4 * stride * sizeof(float));
    ggml_tensor * gml = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 5 * stride * sizeof(float));

    auto broadcast = [&](ggml_tensor * mod) {
        ggml_tensor * out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        float * src = tensor_data(mod), * dst = tensor_data(out);
        for (int s = 0; s < seq_len; s++)
            for (int b = 0; b < n_batch; b++)
                memcpy(dst + (b * seq_len + s) * hidden, src + b * hidden, hidden * sizeof(float));
        return out;
    };

    ggml_tensor * sm_t = broadcast(sm), * scm_t = broadcast(scm), * gm_t = broadcast(gm);
    ggml_tensor * sml_t = broadcast(sml), * scl_t = broadcast(scl), * gml_t = broadcast(gml);

    ggml_tensor * ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    for (int i = 0; i < hidden * n_tokens; i++) ((float*)tensor_data(ones))[i] = 1.0f;

    // Self-attention
    ggml_tensor * h = ggml_rms_norm(ctx, x, 1e-6f);
    if (block.attn_norm_w) h = ggml_mul(ctx, h, block.attn_norm_w);
    h = ggml_mul(ctx, ggml_add(ctx, ones, scm_t), h);
    h = ggml_add(ctx, h, sm_t);
    ggml_tensor * attn = dit_attention_multihead(ctx, h,
        block.attn_q_weight, block.attn_k_weight, block.attn_v_weight,
        block.attn_o_weight, nullptr, nullptr, nullptr, block.attn_o_bias,
        seq_len, n_batch, DIT_NUM_HEADS, DIT_HEAD_SIZE,
        block.q_norm_w, block.k_norm_w);
    x = ggml_add(ctx, x, ggml_mul(ctx, gm_t, attn));

    // FFN
    h = ggml_rms_norm(ctx, x, 1e-6f);
    if (block.ffn_norm_w) h = ggml_mul(ctx, h, block.ffn_norm_w);
    h = ggml_mul(ctx, ggml_add(ctx, ones, scl_t), h);
    h = ggml_add(ctx, h, sml_t);
    ggml_tensor * ffn = dit_ffn(ctx, h, block.ffn_w1, block.ffn_w2, block.ffn_b1, block.ffn_b2);
    x = ggml_add(ctx, x, ggml_mul(ctx, gml_t, ffn));
    return x;
}

// ===========================================================================
// Full DiT forward
// ===========================================================================

ggml_tensor * dit_forward(dit_model & model, ggml_context * ctx,
    ggml_tensor * x, ggml_tensor * t, ggml_tensor * speaker_emb)
{
    int seq_len = x->ne[0], n_batch = x->ne[1], hidden = DIT_HIDDEN_SIZE;
    int n_tokens = seq_len * n_batch;

    x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
    x = ggml_reshape_2d(ctx, x, hidden, n_tokens);

    // Time embedding (manual) + speaker
    ggml_tensor * t_emb = compute_time_embedding_manual(ctx, t, model);
    ggml_tensor * cond = t_emb;
    if (speaker_emb && model.spk_proj_w1) {
        ggml_tensor * spk = ggml_mul_mat(ctx, model.spk_proj_w1, speaker_emb);
        if (model.spk_proj_b1) spk = ggml_add(ctx, spk, model.spk_proj_b1);
        spk = ggml_silu(ctx, spk);
        cond = ggml_add(ctx, cond, spk);
    }
    cond = ggml_silu(ctx, cond);

    // Input layer
    if (model.input_layer_w) {
        x = ggml_mul_mat(ctx, model.input_layer_w, x);
        if (model.input_layer_b) x = ggml_add(ctx, x, model.input_layer_b);
    }

    for (int i = 0; i < model.n_layers; i++)
        x = dit_block_forward_simple(ctx, x, cond, model.layers[i], seq_len, n_batch);

    // Output
    ggml_tensor * out = x;
    if (model.out_adaln_w) {
        ggml_tensor * mod = ggml_mul_mat(ctx, model.out_adaln_w, cond);
        if (model.out_adaln_b) mod = ggml_add(ctx, mod, model.out_adaln_b);
        int stride = DIT_HIDDEN_SIZE;
        ggml_tensor * shift = ggml_view_2d(ctx, mod, DIT_HIDDEN_SIZE, n_batch, mod->nb[1], 0);
        ggml_tensor * scale = ggml_view_2d(ctx, mod, DIT_HIDDEN_SIZE, n_batch, mod->nb[1], stride * sizeof(float));
        ggml_tensor * st = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, n_tokens);
        ggml_tensor * sct = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, n_tokens);
        { float *ss = tensor_data(shift), *sd = tensor_data(st);
          float *cs = tensor_data(scale), *cd = tensor_data(sct);
          for (int s = 0; s < seq_len; s++) for (int b = 0; b < n_batch; b++) {
              memcpy(sd+(b*seq_len+s)*DIT_HIDDEN_SIZE, ss+b*DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE*sizeof(float));
              memcpy(cd+(b*seq_len+s)*DIT_HIDDEN_SIZE, cs+b*DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE*sizeof(float)); } }
        out = ggml_rms_norm(ctx, out, 1e-6f);
        ggml_tensor * ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, n_tokens);
        for (int i = 0; i < DIT_HIDDEN_SIZE * n_tokens; i++) ((float*)tensor_data(ones))[i] = 1.0f;
        out = ggml_mul(ctx, out, ggml_add(ctx, ones, sct));
        out = ggml_add(ctx, out, st);
    }
    out = ggml_mul_mat(ctx, model.out_proj_w, out);
    if (model.out_proj_b) out = ggml_add(ctx, out, model.out_proj_b);
    return out;
}
