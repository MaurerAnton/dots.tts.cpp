// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// dots.tts.cpp - DiT forward pass for ggml
// All biases applied. Time embedder + speaker computed manually.

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
// Weight access: PyTorch W[o,i] = ggml_W at offset i + o * ne[0] (row-major, ne[0]=stride)
// ===========================================================================

static ggml_tensor * compute_time_embedding_manual(
    ggml_context * ctx, ggml_tensor * t_tensor, dit_model & model)
{
    int n_batch = t_tensor->ne[0];
    float t_val = ((float*)t_tensor->data)[0];
    int emb_dim = 256, half = emb_dim / 2;

    float sin_emb[256];
    for (int i = 0; i < half; i++) {
        float freq = expf(-logf(10000.0f) * (float)i / (float)half);  // Python: / half, not / (half-1)
        float arg = t_val * freq;
        sin_emb[i] = cosf(arg);           // Python: [cos, sin] order
        sin_emb[half + i] = sinf(arg);
    }

    // MLP layer 1: Linear(256 -> 1024) + bias + SiLU
    float h1[1024];
    {
        float * w1 = tensor_data(model.t_embed_w1);
        float * b1 = model.t_embed_b1 ? tensor_data(model.t_embed_b1) : nullptr;
        for (int o = 0; o < 1024; o++) {
            float s = b1 ? b1[o] : 0.0f;
            for (int i = 0; i < emb_dim; i++) s += w1[i + o * 256] * sin_emb[i];
            h1[o] = s / (1.0f + expf(-s));  // SiLU
        }
    }

    // MLP layer 2: Linear(1024 -> 1024) + bias
    float h2[1024];
    {
        float * w2 = tensor_data(model.t_embed_w2);
        float * b2 = model.t_embed_b2 ? tensor_data(model.t_embed_b2) : nullptr;
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
// Manual speaker projection: Linear(512->1024) + LayerNorm(1024)
// Python: xvec_proj = Sequential(Linear(512,1024), LayerNorm(1024))
// ===========================================================================

static void compute_speaker_manual(dit_model & model, ggml_tensor * speaker_emb, float * out) {
    float * sw1 = tensor_data(model.spk_proj_w1);
    float * sb1 = model.spk_proj_b1 ? tensor_data(model.spk_proj_b1) : nullptr;
    float * se = tensor_data(speaker_emb);
    float temp[1024];

    // Step 1: Linear(512 -> 1024)
    for (int o = 0; o < 1024; o++) {
        float s = sb1 ? sb1[o] : 0.0f;
        for (int i = 0; i < 512; i++) s += sw1[i + o * 512] * se[i];
        temp[o] = s;
    }

    // Step 2: LayerNorm
    float mean = 0, var = 0;
    for (int i = 0; i < 1024; i++) mean += temp[i];
    mean /= 1024;
    for (int i = 0; i < 1024; i++) { float d = temp[i] - mean; var += d * d; }
    var = var / 1024 + 1e-5f;
    float inv_std = 1.0f / sqrtf(var);
    float * ln_w = model.spk_ln_w ? tensor_data(model.spk_ln_w) : nullptr;
    float * ln_b = model.spk_ln_b ? tensor_data(model.spk_ln_b) : nullptr;
    for (int i = 0; i < 1024; i++) {
        float x = (temp[i] - mean) * inv_std;
        if (ln_w) x *= ln_w[i];
        if (ln_b) x += ln_b[i];
        out[i] = x;
    }
}

// ===========================================================================
// Timestep embedding (kept for reference/testing)
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

    // adaLN: compute MANUALLY (ggml graph fails with leaf cond)
    // Python: adaLN_modulation = Sequential(SiLU(), Linear(1024, 6144))
    // So: apply SiLU to cond first, then matmul
    float cond_silu[1024];
    {
        float * cd = tensor_data(cond);
        for (int i = 0; i < 1024; i++) cond_silu[i] = cd[i] / (1.0f + expf(-cd[i]));
        float r = 0; for (int i = 0; i < 1024; i++) r += cond_silu[i]*cond_silu[i];
        static int cnt = 0;
        if (cnt == 0) fprintf(stderr, "  block0 cond rms=%.4f silu_rms=%.4f first3=[%.4f,%.4f,%.4f]\n",
            sqrtf(r/1024), cnt, cond_silu[0], cond_silu[1], cond_silu[2]);
        cnt++;
    }
    float adaln_raw[6 * DIT_HIDDEN_SIZE];
    {
        float * aw = tensor_data(block.adaln_linear_w);  // ne=[1024, 6144]
        float * ab = block.adaln_linear_b ? tensor_data(block.adaln_linear_b) : nullptr;
        for (int o = 0; o < 6 * hidden; o++) {
            float s = ab ? ab[o] : 0.0f;
            for (int i = 0; i < DIT_HIDDEN_SIZE; i++) s += aw[i + o * DIT_HIDDEN_SIZE] * cond_silu[i];
            adaln_raw[o] = s;
        }
    }
    ggml_tensor * adaln = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6 * hidden, 1);
    memcpy(tensor_data(adaln), adaln_raw, 6 * hidden * sizeof(float));
    // DEBUG
    { float r = 0; for (int i = 0; i < 6*hidden; i++) r += adaln_raw[i]*adaln_raw[i];
      fprintf(stderr, "  adaLN manual: rms=%.4f first5=[%.4f,%.4f,%.4f,%.4f,%.4f] has_nan=%d\n",
          sqrtf(r/(6*hidden)), adaln_raw[0], adaln_raw[1], adaln_raw[2], adaln_raw[3], adaln_raw[4],
          (int)std::isnan(adaln_raw[0])); }

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

    // Time embedding (manual, leaf tensor)
    ggml_tensor * t_emb = compute_time_embedding_manual(ctx, t, model);

    // Speaker conditioning (manual, merged into new leaf tensor)
    // NOTE: xvec_proj = Sequential(Linear(512,1024), LayerNorm(1024))
    // Currently loading only first layer; TODO: add LayerNorm
    ggml_tensor * cond;
    if (speaker_emb && model.spk_proj_w1) {
        float spk_vals[1024];
        compute_speaker_manual(model, speaker_emb, spk_vals);
        // Manual add: t_emb + spk
        float * te = tensor_data(t_emb);
        cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, n_batch);
        float * cd = tensor_data(cond);
        for (int i = 0; i < 1024; i++) cd[i] = te[i] + spk_vals[i];
        { float r = 0; for (int i = 0; i < 1024; i++) r += spk_vals[i]*spk_vals[i];
          fprintf(stderr, "  speaker g_cond rms=%.4f first3=[%.4f,%.4f,%.4f] ln_w=%s ln_b=%s\n",
              sqrtf(r/1024), spk_vals[0], spk_vals[1], spk_vals[2],
              model.spk_ln_w ? "LOADED" : "NULL", model.spk_ln_b ? "LOADED" : "NULL"); }
    } else {
        cond = t_emb;
    }
    // NOTE: no SiLU on cond — Python vfp.forward does NOT apply SiLU to c.

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
        // Compute output adaLN manually (same pattern as block adaLN)
        float cond_silu[1024];
        { float * cd = tensor_data(cond); for (int i = 0; i < 1024; i++) cond_silu[i] = cd[i] / (1.0f + expf(-cd[i])); }
        float mod_raw[2 * DIT_HIDDEN_SIZE];
        {
            float * aw = tensor_data(model.out_adaln_w);  // ne=[1024, 2048]
            float * ab = model.out_adaln_b ? tensor_data(model.out_adaln_b) : nullptr;
            for (int o = 0; o < 2 * hidden; o++) {
                float s = ab ? ab[o] : 0.0f;
                for (int i = 0; i < DIT_HIDDEN_SIZE; i++) s += aw[i + o * DIT_HIDDEN_SIZE] * cond_silu[i];
                mod_raw[o] = s;
            }
        }
        ggml_tensor * mod = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * DIT_HIDDEN_SIZE, 1);
        memcpy(tensor_data(mod), mod_raw, 2 * DIT_HIDDEN_SIZE * sizeof(float));
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
