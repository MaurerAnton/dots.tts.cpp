// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// dots.tts.cpp - GGUF model loader

#include "dots_tts.h"
#include "dit.h"
#include "patchenc.h"
#include "gguf.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>

// Load all model weights from a GGUF file.
// Weight context must be pre-allocated large enough.
// Returns true on success.
bool dots_tts_load_gguf(
    const char * fname,
    ggml_context * w_ctx,
    dit_model & dit,
    patch_encoder & enc)
{
    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &w_ctx,
    };
    struct gguf_context * gctx = gguf_init_from_file(fname, params);
    if (!gctx) {
        fprintf(stderr, "Failed to load GGUF: %s\n", fname);
        return false;
    }

    int n_tensors = gguf_get_n_tensors(gctx);
    printf("GGUF: %d tensors\n", n_tensors);

    // Allocate layers
    dit.layers.resize(DIT_NUM_LAYERS);
    enc.layers.resize(PATCHENC_NUM_LAYERS);

    // Zero-init all pointers in layers
    for (auto & l : dit.layers)  memset(&l, 0, sizeof(l));
    for (auto & l : enc.layers) memset(&l, 0, sizeof(l));

    // Map tensors by name
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gctx, i);
        ggml_tensor * t = ggml_get_tensor(w_ctx, name);
        if (!t) {
            fprintf(stderr, "Tensor not found in context: %s\n", name);
            continue;
        }

        // --- PatchEncoder ---
        if (strncmp(name, "pe.layers.", 10) == 0) {
            int layer_idx;
            char sub[128];
            if (sscanf(name, "pe.layers.%d.%127s", &layer_idx, sub) == 2) {
                if (layer_idx < 0 || layer_idx >= PATCHENC_NUM_LAYERS) continue;
                patchenc_layer & l = enc.layers[layer_idx];

                if      (strcmp(sub, "attn.q.weight") == 0) l.attn_qkv_weight = t;
                else if (strcmp(sub, "attn.k.weight") == 0) { /* stored in patchenc but currently merged QKV */ }
                else if (strcmp(sub, "attn.v.weight") == 0) { /* stored in patchenc but currently merged QKV */ }
                else if (strcmp(sub, "attn.o.weight") == 0) l.attn_o_weight = t;
                else if (strcmp(sub, "attn.o.bias")   == 0) l.attn_o_bias = t;
                else if (strcmp(sub, "attn_norm.weight") == 0) l.attn_norm_w = t;
                else if (strcmp(sub, "ffn_norm.weight") == 0) l.ffn_norm_w = t;
                else if (strcmp(sub, "ffn.w1.weight") == 0) l.ffn_w1 = t;
                else if (strcmp(sub, "ffn.w1.bias")   == 0) { /* skip */ }
                else if (strcmp(sub, "ffn.w2.weight") == 0) l.ffn_w2 = t;
                else if (strcmp(sub, "ffn.w2.bias")   == 0) { /* skip */ }
                else if (strcmp(sub, "q_norm.weight") == 0) l.q_norm_w = t;
                else if (strcmp(sub, "k_norm.weight") == 0) l.k_norm_w = t;
            }
            continue;
        }

        // PatchEncoder projections
        if (strcmp(name, "pe.conv.weight") == 0) enc.conv_weight = t;
        else if (strcmp(name, "pe.conv.bias") == 0) enc.conv_bias = t;
        else if (strcmp(name, "pe.in_proj.weight") == 0) enc.in_proj_w = t;
        else if (strcmp(name, "pe.in_proj.bias") == 0) enc.in_proj_b = t;
        else if (strcmp(name, "pe.out_proj.weight") == 0) enc.out_proj_w = t;
        else if (strcmp(name, "pe.out_proj.bias") == 0) enc.out_proj_b = t;

        // --- DiT ---
        else if (strncmp(name, "dit.layers.", 11) == 0) {
            int layer_idx;
            char sub[128];
            if (sscanf(name, "dit.layers.%d.%127s", &layer_idx, sub) == 2) {
                if (layer_idx < 0 || layer_idx >= DIT_NUM_LAYERS) continue;
                dit_block & b = dit.layers[layer_idx];

                if      (strcmp(sub, "attn.q.weight") == 0) b.attn_q_weight = t;
                else if (strcmp(sub, "attn.k.weight") == 0) b.attn_k_weight = t;
                else if (strcmp(sub, "attn.v.weight") == 0) b.attn_v_weight = t;
                else if (strcmp(sub, "attn.o.weight") == 0) b.attn_o_weight = t;
                else if (strcmp(sub, "attn.o.bias")   == 0) b.attn_o_bias = t;
                else if (strcmp(sub, "q_norm.weight") == 0) b.q_norm_w = t;
                else if (strcmp(sub, "k_norm.weight") == 0) b.k_norm_w = t;
                else if (strcmp(sub, "ffn.w1.weight") == 0) b.ffn_w1 = t;
                else if (strcmp(sub, "ffn.w1.bias")   == 0) b.ffn_b1 = t;
                else if (strcmp(sub, "ffn.w2.weight") == 0) b.ffn_w2 = t;
                else if (strcmp(sub, "ffn.w2.bias")   == 0) b.ffn_b2 = t;
                else if (strcmp(sub, "adaln.linear.weight") == 0) b.adaln_linear_w = t;
                else if (strcmp(sub, "adaln.linear.bias")   == 0) b.adaln_linear_b = t;
                else if (strcmp(sub, "attn_norm.weight") == 0) b.attn_norm_w = t;
                else if (strcmp(sub, "ffn_norm.weight") == 0) b.ffn_norm_w = t;
            }
            continue;
        }

        // DiT projections
        else if (strcmp(name, "dit.spk_proj.w1.weight") == 0) dit.spk_proj_w1 = t;
        else if (strcmp(name, "dit.spk_proj.w1.bias")   == 0) dit.spk_proj_b1 = t;
        else if (strcmp(name, "dit.t_embed.w1.weight")  == 0) dit.t_embed_w1 = t;
        else if (strcmp(name, "dit.t_embed.w1.bias")    == 0) dit.t_embed_b1 = t;
        else if (strcmp(name, "dit.t_embed.w2.weight")  == 0) dit.t_embed_w2 = t;
        else if (strcmp(name, "dit.t_embed.w2.bias")    == 0) dit.t_embed_b2 = t;
        else if (strcmp(name, "dit.out_proj.weight")    == 0) dit.out_proj_w = t;
        else if (strcmp(name, "dit.out_proj.bias")      == 0) dit.out_proj_b = t;
    }

    gguf_free(gctx);
    printf("GGUF loaded successfully\n");
    return true;
}
