// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// Manual DiT operations: LayerNorm, RMSNorm, Linear, RoPE, Softmax, Attention, Block
#pragma once
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cblas.h>

inline void manual_layernorm(float * out, const float * x, int n) {
    float mean = 0; for (int i = 0; i < n; i++) mean += x[i]; mean /= n;
    float var = 0; for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    float inv_std = 1.0f / sqrtf(var / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv_std;
}

inline void manual_rms_norm(float * out, const float * x, const float * weight, int n) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) { float v = x[i] * inv; if (weight) v *= weight[i]; out[i] = v; }
}

// BLAS-based linear: uses OpenBLAS sgemm for large matrices, manual loops for small ones
inline void manual_linear(float * out, const float * x, const float * w, const float * bias, int in_feat, int out_feat) {
    // Use BLAS only for large matmuls (DiT FFN, projections).
    // Small attention matmuls (128-dim heads) are faster with manual loops.
    // Use BLAS only for DiT matmuls (hidden=1024, output=128).
    // LLM matmuls (1536, 256, 8960) use manual loops — already byte-perfect.
    bool use_blas = (in_feat == 1024 || out_feat == 128);
    if (use_blas) {
        if (bias) {
            for (int o = 0; o < out_feat; o++) out[o] = bias[o];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                1, out_feat, in_feat, 1.0f, x, in_feat, w, in_feat, 1.0f, out, out_feat);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                1, out_feat, in_feat, 1.0f, x, in_feat, w, in_feat, 0.0f, out, out_feat);
        }
    } else {
        // Manual loop for small matrices (lower overhead)
        if (bias) {
            for (int o = 0; o < out_feat; o++) {
                float s = bias[o];
                for (int i = 0; i < in_feat; i++) s += w[o * in_feat + i] * x[i];
                out[o] = s;
            }
        } else {
            for (int o = 0; o < out_feat; o++) {
                float s = 0.0f;
                for (int i = 0; i < in_feat; i++) s += w[o * in_feat + i] * x[i];
                out[o] = s;
            }
        }
    }
}

inline void manual_rope_theta(float * q, float * k, int seq_len, int head_dim, float rope_theta) {
    int half = head_dim / 2;
    for (int s = 0; s < seq_len; s++) {
        float theta = 1.0f;
        float ts = powf(rope_theta, -2.0f/(float)head_dim);
        for (int d = 0; d < half; d++) {
            float cs = cosf((float)s*theta), sn = sinf((float)s*theta);
            int i1 = s*head_dim + d;
            int i2 = s*head_dim + d + half;
            float q0=q[i1], q1=q[i2], k0=k[i1], k1=k[i2];
            q[i1]=q0*cs-q1*sn; q[i2]=q1*cs+q0*sn;
            k[i1]=k0*cs-k1*sn; k[i2]=k1*cs+k0*sn;
            theta *= ts;
        }
    }
}
inline void manual_rope(float * q, float * k, int seq_len, int head_dim) {
    manual_rope_theta(q, k, seq_len, head_dim, 10000.0f);
}

inline void manual_softmax(float * x, int n) {
    float mx = x[0]; for (int i=1;i<n;i++) if(x[i]>mx) mx=x[i]; float sum=0;
    for (int i=0;i<n;i++) { x[i]=expf(x[i]-mx); sum+=x[i]; } for (int i=0;i<n;i++) x[i]/=sum;
}

inline void manual_attention(float * out, const float * x,
    const float * qw, const float * kw, const float * vw, const float * ow,
    const float * qb, const float * kb, const float * vb, const float * ob,
    const float * qnw, const float * knw,
    int n_tokens, int hidden, int n_heads, int head_dim,
    const bool * attn_mask = nullptr, const float * pos_ids = nullptr) {
    int n = n_tokens * hidden;
    float * q = new float[n], * k = new float[n], * v = new float[n];
    for (int t = 0; t < n_tokens; t++) {
        manual_linear(q + t*hidden, x + t*hidden, qw, qb, hidden, hidden);
        manual_linear(k + t*hidden, x + t*hidden, kw, kb, hidden, hidden);
        manual_linear(v + t*hidden, x + t*hidden, vw, vb, hidden, hidden);
    }
    float * scores = new float[n_tokens * n_tokens];
    float * ao_flat = new float[n];
    for (int h = 0; h < n_heads; h++) {
        float * qh = new float[n_tokens*head_dim], * kh = new float[n_tokens*head_dim], * vh = new float[n_tokens*head_dim];
        for (int t=0;t<n_tokens;t++) for (int d=0;d<head_dim;d++) {
            qh[t*head_dim+d]=q[t*hidden+h*head_dim+d]; kh[t*head_dim+d]=k[t*hidden+h*head_dim+d];
            vh[t*head_dim+d]=v[t*hidden+h*head_dim+d]; }
        for (int t=0;t<n_tokens;t++) { manual_rms_norm(qh+t*head_dim, qh+t*head_dim, qnw, head_dim);
            manual_rms_norm(kh+t*head_dim, kh+t*head_dim, knw, head_dim); }
        // Use pos_ids if provided, otherwise sequential 0..n_tokens-1
        if (pos_ids) {
            float theta_init = 1.0f;
            for (int t = 0; t < n_tokens; t++) {
                float theta = theta_init; float ts = powf(10000.0f, -2.0f/head_dim);
                float pos = pos_ids[t];
                int half = head_dim/2;
                for (int d = 0; d < half; d++) {
                    float cs = cosf(pos * theta), sn = sinf(pos * theta);
                    int i1 = t*head_dim + d, i2 = t*head_dim + d + half;
                    float q0=qh[i1], q1=qh[i2], k0=kh[i1], k1=kh[i2];
                    qh[i1]=q0*cs-q1*sn; qh[i2]=q1*cs+q0*sn;
                    kh[i1]=k0*cs-k1*sn; kh[i2]=k1*cs+k0*sn;
                    theta *= ts;
                }
            }
        } else {
            manual_rope(qh, kh, n_tokens, head_dim);
        }
        float scale=1.0f/sqrtf((float)head_dim);
        for(int i=0;i<n_tokens;i++){
            for(int j=0;j<n_tokens;j++){
                float s=0;
                for(int d=0;d<head_dim;d++) s+=qh[i*head_dim+d]*kh[j*head_dim+d];
                scores[i*n_tokens+j]=s*scale;
                if (attn_mask && !attn_mask[i*n_tokens+j]) scores[i*n_tokens+j] = -INFINITY;
            }
            manual_softmax(scores+i*n_tokens,n_tokens);
        }
        for(int i=0;i<n_tokens;i++) for(int d=0;d<head_dim;d++){ float s=0;
            for(int j=0;j<n_tokens;j++) s+=scores[i*n_tokens+j]*vh[j*head_dim+d]; ao_flat[i*hidden+h*head_dim+d]=s; }
        delete[] qh; delete[] kh; delete[] vh;
    }
    delete[] scores;
    for (int t = 0; t < n_tokens; t++) manual_linear(out + t*hidden, ao_flat + t*hidden, ow, ob, hidden, hidden);
    delete[] q; delete[] k; delete[] v; delete[] ao_flat;
}

inline void manual_dit_block(const float * x_in, const float * cond, const dit_block & block,
    float * out, int n_tokens, const bool * attn_mask = nullptr, const float * pos_ids = nullptr) {
    int hidden = DIT_HIDDEN_SIZE;
    float * cs = new float[1024]; for(int i=0;i<1024;i++) cs[i]=cond[i]/(1.0f+expf(-cond[i]));
    float * adaln_raw = new float[6*DIT_HIDDEN_SIZE];
    { float * aw = tensor_data(block.adaln_linear_w); float * ab = block.adaln_linear_b?tensor_data(block.adaln_linear_b):nullptr;
      for(int o=0;o<6*hidden;o++){ float s=ab?ab[o]:0.0f; for(int i=0;i<DIT_HIDDEN_SIZE;i++) s+=aw[o*DIT_HIDDEN_SIZE+i]*cs[i]; adaln_raw[o]=s; } }
    float * sm=adaln_raw, * scm=adaln_raw+hidden, * gm=adaln_raw+2*hidden;
    float * sml=adaln_raw+3*hidden, * scl=adaln_raw+4*hidden, * gml=adaln_raw+5*hidden;
    float * normed = new float[n_tokens*hidden], * mod = new float[n_tokens*hidden], * ao = new float[n_tokens*hidden];
    for(int t=0;t<n_tokens;t++){ manual_layernorm(normed+t*hidden, x_in+t*hidden, hidden);
        for(int i=0;i<hidden;i++) mod[t*hidden+i]=normed[t*hidden+i]*(1.0f+scm[i])+sm[i]; }
    manual_attention(ao, mod, tensor_data(block.attn_q_weight), tensor_data(block.attn_k_weight),
        tensor_data(block.attn_v_weight), tensor_data(block.attn_o_weight),
        nullptr,nullptr,nullptr, block.attn_o_bias?tensor_data(block.attn_o_bias):nullptr,
        block.q_norm_w?tensor_data(block.q_norm_w):nullptr, block.k_norm_w?tensor_data(block.k_norm_w):nullptr,
        n_tokens,hidden,DIT_NUM_HEADS,DIT_HEAD_SIZE, attn_mask, pos_ids);
    float * h = new float[n_tokens*hidden];
    for(int i=0;i<n_tokens*hidden;i++) h[i]=x_in[i]+gm[i%hidden]*ao[i];
    for(int t=0;t<n_tokens;t++){ manual_layernorm(normed+t*hidden, h+t*hidden, hidden);
        for(int i=0;i<hidden;i++) mod[t*hidden+i]=normed[t*hidden+i]*(1.0f+scl[i])+sml[i]; }
    
    // FFN: use ggml (SIMD-optimized)
    {
        static thread_local ggml_context * ffn_ctx = nullptr;
        static thread_local ggml_init_params ffn_gp = { .mem_size = 64ULL*1024*1024 };
        if (!ffn_ctx) ffn_ctx = ggml_init(ffn_gp);
        else ggml_reset(ffn_ctx);
        ggml_tensor * xt = ggml_new_tensor_2d(ffn_ctx, GGML_TYPE_F32, hidden, n_tokens);
        memcpy(xt->data, mod, n_tokens * hidden * sizeof(float));
        ggml_tensor * f1 = ggml_mul_mat(ffn_ctx, block.ffn_w1, xt);
        if (block.ffn_b1) f1 = ggml_add(ffn_ctx, f1, block.ffn_b1);
        f1 = ggml_gelu(ffn_ctx, f1);
        ggml_tensor * f2 = ggml_mul_mat(ffn_ctx, block.ffn_w2, f1);
        if (block.ffn_b2) f2 = ggml_add(ffn_ctx, f2, block.ffn_b2);
        ggml_cgraph * cg = ggml_new_graph(ffn_ctx);
        ggml_build_forward_expand(cg, f2);
        ggml_graph_compute_with_ctx(ffn_ctx, cg, 1);
        const float * f2d = (const float*)f2->data;
        for (int t = 0; t < n_tokens; t++)
            for (int i = 0; i < hidden; i++)
                out[t*hidden+i] = h[t*hidden+i] + gml[i] * f2d[t*hidden+i];
    }
    delete[] cs; delete[] adaln_raw; delete[] normed; delete[] mod; delete[] ao; delete[] h;
}
