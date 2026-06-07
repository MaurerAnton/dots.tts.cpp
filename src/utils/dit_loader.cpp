// dots.tts.cpp - Load DiT weights from safetensors
// Tensor name mapping: safetensors -> dit_model/dit_block fields

#include "dit.h"
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "safetensors.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m) {
    m.n_layers = DIT_NUM_LAYERS;
    m.layers.resize(m.n_layers);

    auto load = [&](const char * sf_name, ggml_tensor * & field) -> bool {
        const st_tensor_info * info = sf.find(sf_name);
        if (!info) {
            // Some tensors are optional (LLM, etc.) — skip silently
            return false;
        }
        field = sf.load_tensor(w_ctx, *info);
        if (!field) {
            fprintf(stderr, "FAILED to load %s\n", sf_name);
            return false;
        }
        return true;
    };

    // DiT blocks
    for (int i = 0; i < m.n_layers; i++) {
        dit_block & b = m.layers[i];
        char name[256];

        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.attn.q_proj.weight", i);
        load(name, b.attn_q_weight);

        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.attn.k_proj.weight", i);
        load(name, b.attn_k_weight);

        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.attn.v_proj.weight", i);
        load(name, b.attn_v_weight);

        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.attn.o_proj.weight", i);
        load(name, b.attn_o_weight);

        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.attn.o_proj.bias", i);
        load(name, b.attn_o_bias);

        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.attn.q_norm.weight", i);
        load(name, b.q_norm_w);

        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.attn.k_norm.weight", i);
        load(name, b.k_norm_w);

        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.ffn.fc1.weight", i);
        load(name, b.ffn_w1);
        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.ffn.fc1.bias", i);
        load(name, b.ffn_b1);

        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.ffn.fc2.weight", i);
        load(name, b.ffn_w2);
        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.ffn.fc2.bias", i);
        load(name, b.ffn_b2);

        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.adaLN_modulation.1.weight", i);
        load(name, b.adaln_linear_w);
        snprintf(name, sizeof(name), "velocity_field_predictor.blocks.%d.adaLN_modulation.1.bias", i);
        load(name, b.adaln_linear_b);
    }

    // Projections
    load("velocity_field_predictor.time_embedder.mlp.0.weight", m.t_embed_w1);
    load("velocity_field_predictor.time_embedder.mlp.0.bias",   m.t_embed_b1);
    load("velocity_field_predictor.time_embedder.mlp.2.weight", m.t_embed_w2);
    load("velocity_field_predictor.time_embedder.mlp.2.bias",   m.t_embed_b2);

    load("xvec_proj.0.weight", m.spk_proj_w1);
    load("xvec_proj.0.bias",   m.spk_proj_b1);

    load("velocity_field_predictor.output_layer.linear.weight", m.out_proj_w);
    load("velocity_field_predictor.output_layer.linear.bias",   m.out_proj_b);

    load("hidden_proj.weight", m.hidden_proj_w);
    load("hidden_proj.bias",   m.hidden_proj_b);
    load("latent_proj.weight", m.latent_proj_w);
    load("latent_proj.bias",   m.latent_proj_b);
    load("coordinate_proj.weight", m.coord_proj_w);
    load("coordinate_proj.bias",   m.coord_proj_b);

    load("velocity_field_predictor.input_layer.weight", m.input_layer_w);
    load("velocity_field_predictor.input_layer.bias",   m.input_layer_b);

    // Count loaded tensors
    int loaded = 0;
    auto count_field = [&](ggml_tensor * t) { if (t) loaded++; };
    for (int i = 0; i < m.n_layers; i++) {
        dit_block & b = m.layers[i];
        count_field(b.attn_q_weight);
        count_field(b.attn_k_weight);
        count_field(b.attn_v_weight);
        count_field(b.attn_o_weight);
        count_field(b.q_norm_w);
        count_field(b.k_norm_w);
        count_field(b.ffn_w1);
        count_field(b.ffn_w2);
        count_field(b.adaln_linear_w);
    }
    count_field(m.t_embed_w1);
    count_field(m.t_embed_w2);
    count_field(m.spk_proj_w1);
    count_field(m.out_proj_w);

    printf("Loaded %d DiT tensors (%.1f MB)\n", loaded, ggml_used_mem(w_ctx) / (1024.0 * 1024.0));
    return loaded > 100; // expect ~180 tensors
}
