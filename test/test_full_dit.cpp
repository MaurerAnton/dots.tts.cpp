// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// Full 18-block DiT calibration: dump per-block outputs + final velocity
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "dit_manual.h"
#include "safetensors.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cmath>

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);

static void write_bin(const char * name, const float * data, int n) {
    char path[256]; snprintf(path, sizeof(path), "debug/cpp_%s.bin", name);
    FILE * f = fopen(path, "wb"); fwrite(data, sizeof(float), n, f); fclose(f);
}

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";

    SafeTensorsFile sf; sf.open(model_path);
    ggml_init_params wp = { .mem_size = 2ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model m; load_dit_weights(sf, w_ctx, m); sf.close();

    int n_tokens = 8, hidden = DIT_HIDDEN_SIZE;

    // Load C++ DiT input
    float * x_in = new float[n_tokens * hidden];
    FILE * f_in = fopen("debug/cpp_dit_input.bin", "rb");
    if (!f_in) { fprintf(stderr, "Run e2e_pipeline --dump first\n"); return 1; }
    fread(x_in, sizeof(float), n_tokens * hidden, f_in); fclose(f_in);

    // Run dit_forward_raw (exact same code as e2e_pipeline)
    float * h = new float[n_tokens * hidden];
    {
        float * iw = tensor_data(m.input_layer_w);
        float * ib = m.input_layer_b ? tensor_data(m.input_layer_b) : nullptr;
        for (int t = 0; t < n_tokens; t++)
            manual_linear(h + t*hidden, x_in + t*hidden, iw, ib, hidden, hidden);
    }
    write_bin("dit_input_layer", h, n_tokens * hidden);

    // Compute cond
    float cond[1024];
    {
        float t_val = 0.0f;
        int half = 128; float se[256], h1[1024];
        for (int i = 0; i < half; i++) {
            float f = expf(-logf(10000.0f) * (float)i / (float)half);
            se[i] = cosf(t_val * f); se[half + i] = sinf(t_val * f);
        }
        float * w1 = tensor_data(m.t_embed_w1);
        float * b1 = m.t_embed_b1 ? tensor_data(m.t_embed_b1) : nullptr;
        for (int o = 0; o < 1024; o++) {
            float s = b1 ? b1[o] : 0.0f;
            for (int i = 0; i < 256; i++) s += w1[o*256 + i] * se[i];
            h1[o] = s / (1.0f + expf(-s));
        }
        float * w2 = tensor_data(m.t_embed_w2);
        float * b2 = m.t_embed_b2 ? tensor_data(m.t_embed_b2) : nullptr;
        for (int o = 0; o < 1024; o++) {
            float s = b2 ? b2[o] : 0.0f;
            for (int i = 0; i < 1024; i++) s += w2[o*1024 + i] * h1[i];
            cond[o] = s;
        }
        // Speaker zeros
        float temp[1024];
        float * sw1 = tensor_data(m.spk_proj_w1);
        float * sb1 = m.spk_proj_b1 ? tensor_data(m.spk_proj_b1) : nullptr;
        for (int o = 0; o < 1024; o++) {
            float s = sb1 ? sb1[o] : 0.0f;
            for (int i = 0; i < 512; i++) s += sw1[o*512 + i] * 0.0f;
            temp[o] = s;
        }
        float mean = 0;
        for (int i = 0; i < 1024; i++) mean += temp[i]; mean /= 1024;
        float var = 0;
        for (int i = 0; i < 1024; i++) { float d = temp[i] - mean; var += d * d; }
        var = var / 1024 + 1e-5f;
        float istd = 1.0f / sqrtf(var);
        float * lw = m.spk_ln_w ? tensor_data(m.spk_ln_w) : nullptr;
        float * lb = m.spk_ln_b ? tensor_data(m.spk_ln_b) : nullptr;
        for (int i = 0; i < 1024; i++) {
            float x = (temp[i] - mean) * istd;
            if (lw) x *= lw[i]; if (lb) x += lb[i];
            cond[i] += x;
        }
    }

    // Run all blocks
    float * bo = new float[n_tokens * hidden];
    for (int i = 0; i < m.n_layers; i++) {
        manual_dit_block(h, cond, m.layers[i], bo, n_tokens);
        float * tmp = h; h = bo; bo = tmp;
        char name[64]; snprintf(name, sizeof(name), "dit_block_%d", i);
        write_bin(name, h, n_tokens * hidden);
    }

    // Output layer
    float cs[1024];
    for (int i = 0; i < 1024; i++) cs[i] = cond[i] / (1.0f + expf(-cond[i]));
    float mod_raw[2*DIT_HIDDEN_SIZE];
    {
        float * aw = tensor_data(m.out_adaln_w);
        float * ab = m.out_adaln_b ? tensor_data(m.out_adaln_b) : nullptr;
        for (int o = 0; o < 2*hidden; o++) {
            float s = ab ? ab[o] : 0.0f;
            for (int i = 0; i < DIT_HIDDEN_SIZE; i++) s += aw[o*DIT_HIDDEN_SIZE + i] * cs[i];
            mod_raw[o] = s;
        }
    }
    float * shift = mod_raw, * scale = mod_raw + hidden;
    float * ln = new float[n_tokens * hidden];
    for (int t = 0; t < n_tokens; t++) {
        manual_layernorm(ln + t*hidden, h + t*hidden, hidden);
        for (int i = 0; i < hidden; i++)
            ln[t*hidden + i] = ln[t*hidden + i] * (1.0f + scale[i]) + shift[i];
    }
    float * ow = tensor_data(m.out_proj_w);
    float * ob = m.out_proj_b ? tensor_data(m.out_proj_b) : nullptr;
    float * out_v = new float[n_tokens * VAE_LATENT_DIM];
    for (int t = 0; t < n_tokens; t++)
        manual_linear(out_v + t*VAE_LATENT_DIM, ln + t*hidden, ow, ob, hidden, VAE_LATENT_DIM);
    write_bin("dit_velocity", out_v, n_tokens * VAE_LATENT_DIM);

    float rms = 0;
    for (int i = 0; i < n_tokens * VAE_LATENT_DIM; i++) rms += out_v[i] * out_v[i];
    fprintf(stderr, "Final velocity RMS: %.4f (n=%d elems)\n",
        sqrtf(rms / (n_tokens * VAE_LATENT_DIM)), n_tokens * VAE_LATENT_DIM);

    delete[] x_in;
    delete[] h; delete[] bo; delete[] ln; delete[] out_v;
    ggml_free(w_ctx);
    return 0;
}
