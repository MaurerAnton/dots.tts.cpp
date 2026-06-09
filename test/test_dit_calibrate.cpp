// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// DiT test: check for NaN/Inf in output
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
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";

    SafeTensorsFile sf;
    if (!sf.open(model_path)) return 1;
    ggml_init_params wp = { .mem_size = 2ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model m;
    load_dit_weights(sf, w_ctx, m);
    sf.close();

    ggml_init_params gp = { .mem_size = 512ULL*1024*1024 };
    ggml_context * ctx = ggml_init(gp);

    // Use small random input (not all zeros — ggml graph needs non-leaf input)
    int seq_len = 8, n_batch = 1, hidden = DIT_HIDDEN_SIZE;
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, n_batch, seq_len);
    { float * xd = tensor_data(x);
      for (int i = 0; i < seq_len * hidden; i++) xd[i] = ((float)(rand() % 1000) / 1000.0f - 0.5f) * 0.01f; }
    x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 1, 0, 3));

    ggml_tensor * t_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ((float*)t_tensor->data)[0] = 0.0f;
    ggml_tensor * spk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1);
    memset(tensor_data(spk), 0, 512 * sizeof(float));

    printf("Running DiT...\n");
    ggml_tensor * dout = dit_forward(m, ctx, x, t_tensor, spk);
    {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dout);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
    }

    float * od = tensor_data(dout);
    int nnan = 0, ninf = 0, nfin = 0;
    float rms = 0;
    for (int i = 0; i < 1024; i++) {
        if (std::isnan(od[i])) nnan++;
        else if (std::isinf(od[i])) ninf++;
        else { rms += od[i]*od[i]; nfin++; }
    }
    printf("dit_output: nan=%d inf=%d finite=%d rms=%.4f\n",
        nnan, ninf, nfin, nfin > 0 ? sqrtf(rms/nfin) : 0.0f);
    printf("  first10: ");
    for (int i = 0; i < 10 && i < 1024; i++) printf("%.4f ", od[i]);
    printf("\n");

    // Compare with Python
    float py_rms = 0.4756;
    float cpp_rms = nfin > 0 ? sqrtf(rms/nfin) : 0.0f;
    printf("  Python RMS=%.4f  C++ RMS=%.4f  %s\n",
        py_rms, cpp_rms,
        fabsf(cpp_rms - py_rms) < 0.02 ? "MATCH!" : "DIFFERS");

    ggml_free(ctx); ggml_free(w_ctx);
    return 0;
}
