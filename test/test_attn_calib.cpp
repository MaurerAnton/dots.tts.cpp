// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// Block 0 attention calibration: dump all intermediates for Python comparison
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
    float rms = 0; for (int i = 0; i < n; i++) rms += data[i]*data[i];
    fprintf(stderr, "  cpp_%s: %d elems, rms=%.4f\n", name, n, sqrtf(rms/n));
}

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";

    SafeTensorsFile sf; sf.open(model_path);
    ggml_init_params wp = { .mem_size = 2ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model m; load_dit_weights(sf, w_ctx, m); sf.close();

    int n_tokens = 8, hidden = DIT_HIDDEN_SIZE, n_heads = DIT_NUM_HEADS, head_dim = DIT_HEAD_SIZE;

    // Load C++ DiT input (same as what e2e_pipeline produces with --dump)
    float * x_in = new float[n_tokens * hidden];
    FILE * f_in = fopen("debug/cpp_dit_input.bin", "rb");
    if (!f_in) { fprintf(stderr, "Run e2e_pipeline --dump first\n"); return 1; }
    fread(x_in, sizeof(float), n_tokens * hidden, f_in); fclose(f_in);

    // Compute input_layer (same as dit_forward_raw)
    float * h = new float[n_tokens * hidden];
    {
        float * iw = tensor_data(m.input_layer_w);
        float * ib = m.input_layer_b ? tensor_data(m.input_layer_b) : nullptr;
        for (int t = 0; t < n_tokens; t++)
            manual_linear(h + t*hidden, x_in + t*hidden, iw, ib, hidden, hidden);
    }
    write_bin("b0_x_in", h, n_tokens * hidden);

    // Compute cond (t=0, speaker=zeros)
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
        // Add speaker (zeros)
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
    write_bin("b0_cond", cond, 1024);

    // adaLN modulation
    float * cs = new float[1024];
    for (int i = 0; i < 1024; i++) cs[i] = cond[i] / (1.0f + expf(-cond[i]));
    write_bin("b0_cond_silu", cs, 1024);

    float * adaln_raw = new float[6*DIT_HIDDEN_SIZE];
    {
        float * aw = tensor_data(m.layers[0].adaln_linear_w);
        float * ab = m.layers[0].adaln_linear_b ? tensor_data(m.layers[0].adaln_linear_b) : nullptr;
        for (int o = 0; o < 6*hidden; o++) {
            float s = ab ? ab[o] : 0.0f;
            for (int i = 0; i < DIT_HIDDEN_SIZE; i++) s += aw[o*DIT_HIDDEN_SIZE + i] * cs[i];
            adaln_raw[o] = s;
        }
    }
    float * shift_msa = adaln_raw, * scale_msa = adaln_raw + hidden, * gate_msa = adaln_raw + 2*hidden;
    float * shift_mlp = adaln_raw + 3*hidden, * scale_mlp = adaln_raw + 4*hidden, * gate_mlp = adaln_raw + 5*hidden;
    write_bin("b0_shift_msa", shift_msa, hidden);
    write_bin("b0_scale_msa", scale_msa, hidden);
    write_bin("b0_gate_msa", gate_msa, hidden);

    // LayerNorm + modulation
    float * normed = new float[n_tokens * hidden];
    float * mod = new float[n_tokens * hidden];
    for (int t = 0; t < n_tokens; t++) {
        manual_layernorm(normed + t*hidden, h + t*hidden, hidden);
        for (int i = 0; i < hidden; i++)
            mod[t*hidden + i] = normed[t*hidden + i] * (1.0f + scale_msa[i]) + shift_msa[i];
    }
    write_bin("b0_normed", normed, n_tokens * hidden);
    write_bin("b0_adaln_modulated", mod, n_tokens * hidden);

    // Q/K/V projections
    float * q = new float[n_tokens * hidden];
    float * k = new float[n_tokens * hidden];
    float * v = new float[n_tokens * hidden];
    float * qw = tensor_data(m.layers[0].attn_q_weight);
    float * kw = tensor_data(m.layers[0].attn_k_weight);
    float * vw = tensor_data(m.layers[0].attn_v_weight);
    for (int t = 0; t < n_tokens; t++) {
        manual_linear(q + t*hidden, mod + t*hidden, qw, nullptr, hidden, hidden);
        manual_linear(k + t*hidden, mod + t*hidden, kw, nullptr, hidden, hidden);
        manual_linear(v + t*hidden, mod + t*hidden, vw, nullptr, hidden, hidden);
    }
    write_bin("b0_q_raw", q, n_tokens * hidden);
    write_bin("b0_k_raw", k, n_tokens * hidden);
    write_bin("b0_v_raw", v, n_tokens * hidden);

    // Per-head extraction + qk_norm + RoPE
    float * qnw = m.layers[0].q_norm_w ? tensor_data(m.layers[0].q_norm_w) : nullptr;
    float * knw = m.layers[0].k_norm_w ? tensor_data(m.layers[0].k_norm_w) : nullptr;

    // Do per-head QKV extraction for ALL heads
    float * q_heads = new float[n_heads * n_tokens * head_dim];
    float * k_heads = new float[n_heads * n_tokens * head_dim];
    float * v_heads = new float[n_heads * n_tokens * head_dim];
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < n_tokens; t++) {
            for (int d = 0; d < head_dim; d++) {
                q_heads[(h*n_tokens + t)*head_dim + d] = q[t*hidden + h*head_dim + d];
                k_heads[(h*n_tokens + t)*head_dim + d] = k[t*hidden + h*head_dim + d];
                v_heads[(h*n_tokens + t)*head_dim + d] = v[t*hidden + h*head_dim + d];
            }
        }
    }

    // qk_norm per-head per-token
    float * q_normed = new float[n_heads * n_tokens * head_dim];
    float * k_normed = new float[n_heads * n_tokens * head_dim];
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < n_tokens; t++) {
            manual_rms_norm(q_normed + (h*n_tokens + t)*head_dim, 
                           q_heads + (h*n_tokens + t)*head_dim, qnw, head_dim);
            manual_rms_norm(k_normed + (h*n_tokens + t)*head_dim,
                           k_heads + (h*n_tokens + t)*head_dim, knw, head_dim);
        }
    }

    // RoPE (half-swap)
    float * q_rope = new float[n_heads * n_tokens * head_dim];
    float * k_rope = new float[n_heads * n_tokens * head_dim];
    memcpy(q_rope, q_normed, n_heads * n_tokens * head_dim * sizeof(float));
    memcpy(k_rope, k_normed, n_heads * n_tokens * head_dim * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        manual_rope(q_rope + h*n_tokens*head_dim, k_rope + h*n_tokens*head_dim, n_tokens, head_dim);
    }

    write_bin("b0_q_normed", q_normed, n_heads * n_tokens * head_dim);
    write_bin("b0_k_normed", k_normed, n_heads * n_tokens * head_dim);
    write_bin("b0_q_rope", q_rope, n_heads * n_tokens * head_dim);
    write_bin("b0_k_rope", k_rope, n_heads * n_tokens * head_dim);

    // Attention scores per head
    float * all_scores = new float[n_heads * n_tokens * n_tokens];
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int h = 0; h < n_heads; h++) {
        float * qh = q_rope + h*n_tokens*head_dim;
        float * kh = k_rope + h*n_tokens*head_dim;
        for (int i = 0; i < n_tokens; i++) {
            for (int j = 0; j < n_tokens; j++) {
                float s = 0;
                for (int d = 0; d < head_dim; d++)
                    s += qh[i*head_dim + d] * kh[j*head_dim + d];
                all_scores[(h*n_tokens + i)*n_tokens + j] = s * scale;
            }
        }
    }
    write_bin("b0_scores_raw", all_scores, n_heads * n_tokens * n_tokens);

    // Causal mask + softmax per head
    float * attn_weights = new float[n_heads * n_tokens * n_tokens];
    for (int h = 0; h < n_heads; h++) {
        float * scores = all_scores + h*n_tokens*n_tokens;
        for (int i = 0; i < n_tokens; i++) {
            for (int j = i+1; j < n_tokens; j++) scores[i*n_tokens + j] = -INFINITY;
            manual_softmax(scores + i*n_tokens, n_tokens);
        }
    }
    memcpy(attn_weights, all_scores, n_heads * n_tokens * n_tokens * sizeof(float));
    write_bin("b0_attn_weights", attn_weights, n_heads * n_tokens * n_tokens);

    // Attention output (weighted sum of V)
    float * ao_flat = new float[n_tokens * hidden];
    memset(ao_flat, 0, n_tokens * hidden * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        float * vh = v_heads + h*n_tokens*head_dim;
        for (int i = 0; i < n_tokens; i++) {
            for (int d = 0; d < head_dim; d++) {
                float s = 0;
                for (int j = 0; j < n_tokens; j++)
                    s += attn_weights[(h*n_tokens + i)*n_tokens + j] * vh[j*head_dim + d];
                ao_flat[i*hidden + h*head_dim + d] = s;
            }
        }
    }
    write_bin("b0_attn_pre_o", ao_flat, n_tokens * hidden);

    // O-projection
    float * ao = new float[n_tokens * hidden];
    float * ow = tensor_data(m.layers[0].attn_o_weight);
    float * ob = m.layers[0].attn_o_bias ? tensor_data(m.layers[0].attn_o_bias) : nullptr;
    for (int t = 0; t < n_tokens; t++)
        manual_linear(ao + t*hidden, ao_flat + t*hidden, ow, ob, hidden, hidden);
    write_bin("b0_attn_out", ao, n_tokens * hidden);

    // Residual: h + gate * attn_out
    float * after_attn = new float[n_tokens * hidden];
    for (int i = 0; i < n_tokens * hidden; i++)
        after_attn[i] = h[i] + gate_msa[i % hidden] * ao[i];
    write_bin("b0_after_attn", after_attn, n_tokens * hidden);

    fprintf(stderr, "\nAll cpp_* dumps written to debug/\n");

    // Cleanup
    delete[] x_in; delete[] h; delete[] cs; delete[] adaln_raw;
    delete[] normed; delete[] mod;
    delete[] q; delete[] k; delete[] v;
    delete[] q_heads; delete[] k_heads; delete[] v_heads;
    delete[] q_normed; delete[] k_normed;
    delete[] q_rope; delete[] k_rope;
    delete[] all_scores; delete[] attn_weights;
    delete[] ao_flat; delete[] ao; delete[] after_attn;
    ggml_free(w_ctx);
    return 0;
}
