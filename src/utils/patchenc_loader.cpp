// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// dots.tts.cpp - Load PatchEncoder weights from safetensors

#include "patchenc.h"
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "safetensors.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>

bool load_patchenc_weights(SafeTensorsFile & sf, ggml_context * w_ctx, patch_encoder & enc) {
    enc.n_layers = PATCHENC_NUM_LAYERS;
    enc.hidden_size = PATCHENC_HIDDEN;
    enc.num_heads = PATCHENC_NUM_HEADS;
    enc.head_dim = PATCHENC_HEAD_SIZE;
    enc.ffn_size = PATCHENC_FFN_SIZE;
    enc.latent_dim = PATCHENC_LATENT_DIM;
    enc.patch_size = PATCHENC_PATCH_SIZE;
    enc.layers.resize(enc.n_layers);

    auto load = [&](const char * sf_name, ggml_tensor * & field) {
        const st_tensor_info * info = sf.find(sf_name);
        if (!info) return;
        field = sf.load_tensor(w_ctx, *info);
    };

    // Encoder layers
    for (int i = 0; i < enc.n_layers; i++) {
        patchenc_layer & l = enc.layers[i];
        char name[256];

        snprintf(name, sizeof(name), "patch_encoder.encoder.layers.%d.attn.q_proj.weight", i);
        load(name, l.attn_q_weight);
        snprintf(name, sizeof(name), "patch_encoder.encoder.layers.%d.attn.k_proj.weight", i);
        load(name, l.attn_k_weight);
        snprintf(name, sizeof(name), "patch_encoder.encoder.layers.%d.attn.v_proj.weight", i);
        load(name, l.attn_v_weight);
        snprintf(name, sizeof(name), "patch_encoder.encoder.layers.%d.attn.o_proj.weight", i);
        load(name, l.attn_o_weight);
        snprintf(name, sizeof(name), "patch_encoder.encoder.layers.%d.attn.o_proj.bias", i);
        load(name, l.attn_o_bias);

        snprintf(name, sizeof(name), "patch_encoder.encoder.layers.%d.attn_norm.weight", i);
        load(name, l.attn_norm_w);

        snprintf(name, sizeof(name), "patch_encoder.encoder.layers.%d.ffn.fc1.weight", i);
        load(name, l.ffn_w1);
        snprintf(name, sizeof(name), "patch_encoder.encoder.layers.%d.ffn.fc1.bias", i);
        load(name, l.ffn_b1);
        snprintf(name, sizeof(name), "patch_encoder.encoder.layers.%d.ffn.fc2.weight", i);
        load(name, l.ffn_w2);
        snprintf(name, sizeof(name), "patch_encoder.encoder.layers.%d.ffn.fc2.bias", i);
        load(name, l.ffn_b2);

        snprintf(name, sizeof(name), "patch_encoder.encoder.layers.%d.ffn_norm.weight", i);
        load(name, l.ffn_norm_w);
    }

    // Input/output projections
    load("patch_encoder.ds_proj.weight", enc.conv_weight);
    load("patch_encoder.ds_proj.bias",   enc.conv_bias);
    load("patch_encoder.in_proj.weight", enc.in_proj_w);
    load("patch_encoder.in_proj.bias",   enc.in_proj_b);
    load("patch_encoder.out_proj.weight", enc.out_proj_w);
    load("patch_encoder.out_proj.bias",   enc.out_proj_b);

    int loaded = 0;
    auto count = [&](ggml_tensor * t) { if (t) loaded++; };
    for (int i = 0; i < enc.n_layers; i++) {
        patchenc_layer & l = enc.layers[i];
        count(l.attn_q_weight); count(l.attn_k_weight); count(l.attn_v_weight);
        count(l.attn_o_weight); count(l.attn_norm_w);
        count(l.ffn_w1); count(l.ffn_w2); count(l.ffn_norm_w);
    }
    count(enc.conv_weight); count(enc.in_proj_w); count(enc.out_proj_w);

    printf("Loaded %d PatchEncoder tensors (%.1f MB)\n", loaded,
           ggml_used_mem(w_ctx) / (1024.0 * 1024.0));
    return loaded > 50;
}
