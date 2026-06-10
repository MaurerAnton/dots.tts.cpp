// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// Minimal test: time_embedder MLP 256->1024->1024
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "safetensors.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <cmath>

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);

int main() {
    const char * model_path = "models/model.safetensors";
    const char * env = getenv("DOTS_TTS_MODEL");
    if (env) model_path = env;

    // Load model
    SafeTensorsFile sf;
    if (!sf.open(model_path)) { fprintf(stderr, "Failed to open %s\n", model_path); return 1; }
    ggml_init_params wp = { .mem_size = 2ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model m;
    load_dit_weights(sf, w_ctx, m);
    sf.close();

    printf("t_embed_w1: %s ne=[%ld,%ld,%ld,%ld]\n",
        m.t_embed_w1 ? "LOADED" : "NULL",
        m.t_embed_w1->ne[0], m.t_embed_w1->ne[1], m.t_embed_w1->ne[2], m.t_embed_w1->ne[3]);
    printf("t_embed_b1: %s\n", m.t_embed_b1 ? "LOADED" : "NULL");
    printf("t_embed_w2: %s ne=[%ld,%ld,%ld,%ld]\n",
        m.t_embed_w2 ? "LOADED" : "NULL",
        m.t_embed_w2->ne[0], m.t_embed_w2->ne[1], m.t_embed_w2->ne[2], m.t_embed_w2->ne[3]);
    printf("t_embed_b2: %s\n", m.t_embed_b2 ? "LOADED" : "NULL");

    if (!m.t_embed_w1 || !m.t_embed_w2) {
        fprintf(stderr, "Weights not loaded\n");
        return 1;
    }

    // Print first 5 values of w2 to verify it's loaded
    {
        float * wd = tensor_data(m.t_embed_w2);
        printf("w2 first 5: %.6f %.6f %.6f %.6f %.6f\n", wd[0], wd[1], wd[2], wd[3], wd[4]);
        // Check if w2 is all zeros
        float rms = 0; int nz = 0;
        for (int i = 0; i < 1024*1024; i++) { rms += wd[i]*wd[i]; if (wd[i] != 0) nz++; }
        printf("w2 RMS=%.4f nonzeros=%d/%d\n", sqrtf(rms/(1024*1024)), nz, 1024*1024);
    }

    // Create compute context and build time_embedder
    ggml_init_params gp = { .mem_size = 64ULL*1024*1024 };
    ggml_context * ctx = ggml_init(gp);

    // Sinusoidal embedding for t=0
    int t_dim = 256;
    ggml_tensor * t_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ((float*)t_tensor->data)[0] = 0.0f;

    // Step 1: sinusoidal features
    ggml_tensor * t_sin = dit_timestep_embedding(ctx, t_tensor, t_dim);
    printf("t_sin: ne=[%ld,%ld,%ld,%ld]\n", t_sin->ne[0], t_sin->ne[1], t_sin->ne[2], t_sin->ne[3]);
    {
        float * sd = tensor_data(t_sin);
        float r = 0;
        for (int i = 0; i < t_dim; i++) r += sd[i]*sd[i];
        printf("  RMS=%.4f (expected 0.7071 for t=0)\n", sqrtf(r/t_dim));
    }

    // Step 2: MLP 256->1024 (w1 + bias)
    ggml_tensor * h = ggml_mul_mat(ctx, m.t_embed_w1, t_sin);
    printf("After w1: ne=[%ld,%ld,%ld,%ld]\n", h->ne[0], h->ne[1], h->ne[2], h->ne[3]);
    if (m.t_embed_b1) h = ggml_add(ctx, h, m.t_embed_b1);

    // Step 3: SiLU
    h = ggml_silu(ctx, h);

    // Step 4: MLP 1024->1024 (w2 + bias)
    ggml_tensor * out = ggml_mul_mat(ctx, m.t_embed_w2, h);
    printf("After w2: ne=[%ld,%ld,%ld,%ld]\n", out->ne[0], out->ne[1], out->ne[2], out->ne[3]);
    if (m.t_embed_b2) out = ggml_add(ctx, out, m.t_embed_b2);

    // Compute
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    int n_threads = 1;
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    // Print results
    {
        float * od = tensor_data(out);
        float rms = 0;
        for (int i = 0; i < 1024; i++) rms += od[i]*od[i];
        rms = sqrtf(rms/1024);
        printf("\nFinal output: RMS=%.6f\n", rms);
        printf("  First 10: ");
        for (int i = 0; i < 10; i++) printf("%.6f ", od[i]);
        printf("\n");
        int nz = 0;
        for (int i = 0; i < 1024; i++) if (od[i] != 0.0f) nz++;
        printf("  Nonzeros: %d/1024\n", nz);
    }

    ggml_free(ctx);
    ggml_free(w_ctx);
    return 0;
}
