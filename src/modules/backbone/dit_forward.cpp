// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// dots.tts.cpp - DiT forward pass using manual C++ blocks
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "dit_manual.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <cstdio>

static void compute_time_embedding_manual(dit_model & model, float t_val, float * out) {
    int half = 128; float sin_emb[256];
    for (int i = 0; i < half; i++) { float freq = expf(-logf(10000.0f)*(float)i/(float)half);
        sin_emb[i] = cosf(t_val * freq); sin_emb[half + i] = sinf(t_val * freq); }
    float h1[1024]; { float * w1 = tensor_data(model.t_embed_w1); float * b1 = model.t_embed_b1 ? tensor_data(model.t_embed_b1) : nullptr;
      for (int o = 0; o < 1024; o++) { float s = b1 ? b1[o] : 0.0f;
          for (int i = 0; i < 256; i++) s += w1[o * 256 + i] * sin_emb[i]; h1[o] = s / (1.0f + expf(-s)); } }
    { float * w2 = tensor_data(model.t_embed_w2); float * b2 = model.t_embed_b2 ? tensor_data(model.t_embed_b2) : nullptr;
      for (int o = 0; o < 1024; o++) { float s = b2 ? b2[o] : 0.0f;
          for (int i = 0; i < 1024; i++) s += w2[o * 1024 + i] * h1[i]; out[o] = s; } }
}

static void compute_speaker_manual(dit_model & model, const float * speaker_emb, float * out) {
    float * sw1 = tensor_data(model.spk_proj_w1); float * sb1 = model.spk_proj_b1 ? tensor_data(model.spk_proj_b1) : nullptr;
    float temp[1024];
    for (int o = 0; o < 1024; o++) { float s = sb1 ? sb1[o] : 0.0f; for (int i = 0; i < 512; i++) s += sw1[o * 512 + i] * speaker_emb[i]; temp[o] = s; }
    float mean = 0; for (int i = 0; i < 1024; i++) mean += temp[i]; mean /= 1024;
    float var = 0; for (int i = 0; i < 1024; i++) { float d = temp[i] - mean; var += d * d; } var = var / 1024 + 1e-5f;
    float inv_std = 1.0f / sqrtf(var);
    float * ln_w = model.spk_ln_w ? tensor_data(model.spk_ln_w) : nullptr;
    float * ln_b = model.spk_ln_b ? tensor_data(model.spk_ln_b) : nullptr;
    for (int i = 0; i < 1024; i++) { float x = (temp[i] - mean) * inv_std; if (ln_w) x *= ln_w[i]; if (ln_b) x += ln_b[i]; out[i] = x; }
}

void dit_forward_raw(dit_model & model, const float * x, int seq_len, float t_val,
    const float * speaker_emb, float * out) {
    int hidden = DIT_HIDDEN_SIZE, n_tokens = seq_len;
    float cond[1024];
    compute_time_embedding_manual(model, t_val, cond);
    if (speaker_emb) { float spk_vals[1024]; compute_speaker_manual(model, speaker_emb, spk_vals);
        for (int i = 0; i < 1024; i++) cond[i] += spk_vals[i]; }
    float * h = new float[n_tokens * hidden];
    float * iw = tensor_data(model.input_layer_w); float * ib = model.input_layer_b ? tensor_data(model.input_layer_b) : nullptr;
    for (int t = 0; t < n_tokens; t++) manual_linear(h + t*hidden, x + t*hidden, iw, ib, hidden, hidden);
    float * block_out = new float[n_tokens * hidden];
    for (int i = 0; i < model.n_layers; i++) {
        manual_dit_block(h, cond, model.layers[i], block_out, n_tokens);
        float * tmp = h; h = block_out; block_out = tmp;
    }
    float cs[1024]; for(int i=0;i<1024;i++) cs[i]=cond[i]/(1.0f+expf(-cond[i]));
    float mod_raw[2*DIT_HIDDEN_SIZE];
    { float * aw = tensor_data(model.out_adaln_w); float * ab = model.out_adaln_b ? tensor_data(model.out_adaln_b) : nullptr;
      for(int o=0;o<2*hidden;o++){ float s=ab?ab[o]:0.0f; for(int i=0;i<DIT_HIDDEN_SIZE;i++) s+=aw[o*DIT_HIDDEN_SIZE+i]*cs[i]; mod_raw[o]=s; } }
    float * shift = mod_raw, * scale = mod_raw + hidden;
    float * ln_out = new float[n_tokens * hidden];
    for (int t = 0; t < n_tokens; t++) { manual_layernorm(ln_out + t*hidden, h + t*hidden, hidden);
        for (int i = 0; i < hidden; i++) ln_out[t*hidden+i] = ln_out[t*hidden+i] * (1.0f + scale[i]) + shift[i]; }
    float * ow = tensor_data(model.out_proj_w); float * ob = model.out_proj_b ? tensor_data(model.out_proj_b) : nullptr;
    for (int t = 0; t < n_tokens; t++) manual_linear(out + t*VAE_LATENT_DIM, ln_out + t*hidden, ow, ob, hidden, VAE_LATENT_DIM);
    delete[] h; delete[] block_out; delete[] ln_out;
}

ggml_tensor * dit_forward(dit_model & model, ggml_context * ctx,
    ggml_tensor * x, ggml_tensor * t, ggml_tensor * speaker_emb) {
    int seq_len = x->ne[0], n_batch = x->ne[1], n_tokens = seq_len * n_batch;
    float * x_data = tensor_data(x); float t_val = tensor_data(t)[0];
    float * spk_data = speaker_emb ? tensor_data(speaker_emb) : nullptr;
    float * result_data = new float[n_tokens * VAE_LATENT_DIM];
    dit_forward_raw(model, x_data, n_tokens, t_val, spk_data, result_data);
    ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, VAE_LATENT_DIM, n_tokens);
    memcpy(tensor_data(result), result_data, n_tokens * VAE_LATENT_DIM * sizeof(float));
    delete[] result_data;
    return result;
}
