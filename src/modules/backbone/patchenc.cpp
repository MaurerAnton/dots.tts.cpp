// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// PatchEncoder: ggml graph for Q/K/V, then CPU per-head qk_norm, then ggml for attention+FFN
// This avoids the in-graph reshape/permute issue.

#include "patchenc.h"
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <cstdio>

// ===========================================================================
// Causal Conv1d (separate input buffer)
// ===========================================================================
static void causal_conv1d_stride2(ggml_context * ctx, ggml_tensor * x,
    ggml_tensor * weight, ggml_tensor * bias, int seq_len, int channels) {
    float * xd = (float*)malloc(seq_len * channels * sizeof(float));
    memcpy(xd, tensor_data(x), seq_len * channels * sizeof(float));
    float * wd = tensor_data(weight);
    float * bd = bias ? tensor_data(bias) : nullptr;
    float * od = tensor_data(x);
    int out_seq = seq_len / 2;
    for (int o = 0; o < out_seq; o++)
        for (int oc = 0; oc < channels; oc++) {
            float s = bd ? bd[oc] : 0;
            for (int ic = 0; ic < channels; ic++) {
                int wb = (oc * channels + ic) * 2;
                int i0 = 2*o - 1; if (i0 >= 0 && i0 < seq_len) s += xd[i0*channels + ic] * wd[wb+0];
                int i1 = 2*o;     if (i1 >= 0 && i1 < seq_len) s += xd[i1*channels + ic] * wd[wb+1];
            }
            od[o*channels + oc] = s;
        }
    memset(od + out_seq * channels, 0, (seq_len - out_seq) * channels * sizeof(float));
    free(xd);
}

// ===========================================================================
// CPU per-head qk_norm
// ===========================================================================
static void per_head_qk_norm_cpu(float * qd, float * kd, int hidden, int n_tokens,
                                  int n_heads, int head_dim) {
    for (int t = 0; t < n_tokens; t++)
        for (int h = 0; h < n_heads; h++) {
            float rq=0, rk=0; int base = h*head_dim + t*hidden;
            for (int d=0; d<head_dim; d++) { rq += qd[base+d]*qd[base+d]; rk += kd[base+d]*kd[base+d]; }
            rq = sqrtf(rq/head_dim + 1e-6f); rk = sqrtf(rk/head_dim + 1e-6f);
            for (int d=0; d<head_dim; d++) { qd[base+d] /= rq; kd[base+d] /= rk; }
        }
}

// ===========================================================================
// Per-head attention on CPU (Python: separate scores + softmax per head)
// Input: q/k/v [hidden, n_tokens] in ggml layout (i + t*hidden)
// ===========================================================================
static void per_head_attention_cpu(float * out_ctx,  // [hidden, n_tokens] output context
    const float * q, const float * k, const float * v,
    int n_tokens, int n_heads, int head_dim) {
    int hidden = n_heads * head_dim;
    memset(out_ctx, 0, hidden * n_tokens * sizeof(float));
    float scale = 1.0f / sqrtf((float)head_dim);
    float * scores = (float*)malloc(n_tokens * n_tokens * sizeof(float));
    
    for (int h = 0; h < n_heads; h++) {
        // Scores for this head
        for (int ti = 0; ti < n_tokens; ti++) {
            for (int tj = 0; tj < n_tokens; tj++) {
                float s = 0;
                for (int d = 0; d < head_dim; d++)
                    s += q[(h*head_dim+d) + ti*hidden] * k[(h*head_dim+d) + tj*hidden];
                scores[ti*n_tokens + tj] = s * scale;
            }
            // Causal mask
            for (int tj = ti + 1; tj < n_tokens; tj++)
                scores[ti*n_tokens + tj] = -INFINITY;
            // Softmax
            float max_s = scores[ti*n_tokens];
            for (int tj = 1; tj < n_tokens; tj++)
                if (scores[ti*n_tokens+tj] > max_s) max_s = scores[ti*n_tokens+tj];
            float sum_exp = 0;
            for (int tj = 0; tj < n_tokens; tj++) {
                scores[ti*n_tokens+tj] = expf(scores[ti*n_tokens+tj] - max_s);
                sum_exp += scores[ti*n_tokens+tj];
            }
            for (int tj = 0; tj < n_tokens; tj++)
                scores[ti*n_tokens+tj] /= sum_exp;
        }
        // Weighted sum of V for this head
        for (int ti = 0; ti < n_tokens; ti++)
            for (int d = 0; d < head_dim; d++) {
                float s = 0;
                for (int tj = 0; tj < n_tokens; tj++)
                    s += scores[ti*n_tokens+tj] * v[(h*head_dim+d) + tj*hidden];
                out_ctx[(h*head_dim+d) + ti*hidden] += s;
            }
    }
    free(scores);
}

// ===========================================================================
// Attention graph (RoPE only, then CPU per-head attention)
// ===========================================================================
static ggml_tensor * build_attn_graph(ggml_context * ctx,
    ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
    ggml_tensor * ow, ggml_tensor * ob, int n_tokens, int n_heads, int head_dim) {
    int hidden = n_heads * head_dim;
    // RoPE via ggml
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, n_tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    { int *pd = (int*)tensor_data(pos); for(int s=0;s<n_tokens;s++) pd[s]=s; }
    q = ggml_rope(ctx, q, pos, head_dim, 0);
    k = ggml_rope(ctx, k, pos, head_dim, 0);
    // Compute RoPE graph
    { ggml_cgraph * cg = ggml_new_graph(ctx); ggml_build_forward_expand(cg, q);
      ggml_build_forward_expand(cg, k); ggml_graph_compute_with_ctx(ctx, cg, 8); }
    
    // Extract data for CPU per-head attention
    float * qd = (float*)malloc(hidden * n_tokens * sizeof(float));
    float * kd = (float*)malloc(hidden * n_tokens * sizeof(float));
    float * vd = (float*)malloc(hidden * n_tokens * sizeof(float));
    memcpy(qd, tensor_data(q), hidden * n_tokens * sizeof(float));
    memcpy(kd, tensor_data(k), hidden * n_tokens * sizeof(float));
    memcpy(vd, tensor_data(v), hidden * n_tokens * sizeof(float));
    
    float * ctx_d = (float*)malloc(hidden * n_tokens * sizeof(float));
    per_head_attention_cpu(ctx_d, qd, kd, vd, n_tokens, n_heads, head_dim);
    free(qd); free(kd); free(vd);
    
    // O-proj via ggml
    ggml_tensor * ao = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    memcpy(tensor_data(ao), ctx_d, hidden * n_tokens * sizeof(float));
    free(ctx_d);
    
    ao = ggml_mul_mat(ctx, ow, ao);
    if (ob) { ggml_tensor * ob2 = ggml_reshape_2d(ctx, ob, hidden, 1); ao = ggml_add(ctx, ao, ob2); }
    return ao;
}

// ===========================================================================
// Full forward — Q/K/V via ggml, CPU per-head norm, ggml attention+FFN per layer
// ===========================================================================
ggml_tensor * patchenc_forward(patch_encoder & enc, ggml_context * ctx,
    ggml_tensor * x, int n_patches) {
    int seq = n_patches * PATCHENC_PATCH_SIZE, ch = PATCHENC_LATENT_DIM;
    causal_conv1d_stride2(ctx, x, enc.conv_weight, enc.conv_bias, seq, ch);
    int conv_seq = seq / 2, n_tokens = conv_seq, n_batch = 1;
    int hidden = PATCHENC_HIDDEN;

    // in_proj + compute
    ggml_tensor * xc = ggml_view_2d(ctx, x, ch, conv_seq, x->nb[1], 0); xc = ggml_cont(ctx, xc);
    ggml_tensor * h = ggml_mul_mat(ctx, enc.in_proj_w, xc);
    if (enc.in_proj_b) { ggml_tensor * ib = ggml_reshape_2d(ctx, enc.in_proj_b, hidden, 1); h = ggml_add(ctx, h, ib); }
    { ggml_cgraph * cg = ggml_new_graph(ctx); ggml_build_forward_expand(cg, h); ggml_graph_compute_with_ctx(ctx, cg, 8); }

    // Allocate buffer for hidden state between layers
    float * hd = (float*)malloc(hidden * n_tokens * sizeof(float));
    memcpy(hd, tensor_data(h), hidden * n_tokens * sizeof(float));

    for (int i = 0; i < enc.n_layers; i++) {
        patchenc_layer & l = enc.layers[i];
        ggml_reset(ctx);

        // Input tensor
        ggml_tensor * xi = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        memcpy(tensor_data(xi), hd, hidden * n_tokens * sizeof(float));

        // attn_norm
        ggml_tensor * xn = ggml_rms_norm(ctx, xi, 1e-6f);
        if (l.attn_norm_w) xn = ggml_mul(ctx, xn, l.attn_norm_w);

        // Q/K/V projections via ggml
        ggml_tensor * q = ggml_mul_mat(ctx, l.attn_q_weight, xn);
        ggml_tensor * k = ggml_mul_mat(ctx, l.attn_k_weight, xn);
        ggml_tensor * v = ggml_mul_mat(ctx, l.attn_v_weight, xn);
        q = ggml_cont(ctx, q); k = ggml_cont(ctx, k); v = ggml_cont(ctx, v);

        // Compute Q/K/V graph and extract data for CPU per-head norm
        { ggml_cgraph * cg = ggml_new_graph(ctx); ggml_build_forward_expand(cg, q);
          ggml_build_forward_expand(cg, k); ggml_build_forward_expand(cg, v);
          ggml_graph_compute_with_ctx(ctx, cg, 8); }

        float * qd = (float*)malloc(hidden * n_tokens * sizeof(float));
        float * kd = (float*)malloc(hidden * n_tokens * sizeof(float));
        float * vd = (float*)malloc(hidden * n_tokens * sizeof(float));
        memcpy(qd, tensor_data(q), hidden * n_tokens * sizeof(float));
        memcpy(kd, tensor_data(k), hidden * n_tokens * sizeof(float));
        memcpy(vd, tensor_data(v), hidden * n_tokens * sizeof(float));

        // qk_norm: Python uses Identity (despite config qk_norm=True — model saved without weights)
        // Do NOT apply per-head norm — Python doesn't

        // --- Build ggml graph for attention + FFN ---
        ggml_tensor * qn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        ggml_tensor * kn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        ggml_tensor * vn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        memcpy(tensor_data(qn), qd, hidden * n_tokens * sizeof(float));
        memcpy(tensor_data(kn), kd, hidden * n_tokens * sizeof(float));
        memcpy(tensor_data(vn), vd, hidden * n_tokens * sizeof(float));
        free(qd); free(kd); free(vd);

        ggml_tensor * attn = build_attn_graph(ctx, qn, kn, vn, l.attn_o_weight, l.attn_o_bias,
            n_tokens, PATCHENC_NUM_HEADS, PATCHENC_HEAD_SIZE);
        ggml_tensor * h1 = ggml_add(ctx, xi, attn);

        // FFN
        ggml_tensor * hn = ggml_rms_norm(ctx, h1, 1e-6f);
        if (l.ffn_norm_w) hn = ggml_mul(ctx, hn, l.ffn_norm_w);
        ggml_tensor * ff = ggml_mul_mat(ctx, l.ffn_w1, hn);
        if (l.ffn_b1) { ggml_tensor * fb = ggml_reshape_2d(ctx, l.ffn_b1, PATCHENC_FFN_SIZE, 1); ff = ggml_add(ctx, ff, fb); }
        ff = ggml_silu(ctx, ff);
        ff = ggml_mul_mat(ctx, l.ffn_w2, ff);
        if (l.ffn_b2) { ggml_tensor * fb = ggml_reshape_2d(ctx, l.ffn_b2, hidden, 1); ff = ggml_add(ctx, ff, fb); }
        ggml_tensor * ho = ggml_add(ctx, h1, ff);

        // Compute and extract
        { ggml_cgraph * cg = ggml_new_graph(ctx); ggml_build_forward_expand(cg, ho);
          ggml_graph_compute_with_ctx(ctx, cg, 8); }
        memcpy(hd, tensor_data(ho), hidden * n_tokens * sizeof(float));
    }

    // Output projection
    ggml_reset(ctx);
    ggml_tensor * ht = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    memcpy(tensor_data(ht), hd, hidden * n_tokens * sizeof(float));
    free(hd);
    ggml_tensor * h3 = ggml_reshape_3d(ctx, ht, hidden, 2, n_patches);
    h3 = ggml_cont(ctx, ggml_permute(ctx, h3, 0, 2, 1, 3));
    ggml_tensor * hc = ggml_reshape_2d(ctx, h3, hidden*2, n_patches); hc = ggml_cont(ctx, hc);
    ggml_tensor * out = ggml_mul_mat(ctx, enc.out_proj_w, hc);
    if (enc.out_proj_b) { ggml_tensor * ob = ggml_reshape_2d(ctx, enc.out_proj_b, 1536, 1); out = ggml_add(ctx, out, ob); }
    return out;
}
