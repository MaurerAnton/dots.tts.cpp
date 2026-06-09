// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// dots.tts.cpp - DiT (Diffusion Transformer) implementation
// Reference: rednote-hilab/dots.tts (Apache 2.0)
// Pure ggml - no PyTorch, no ONNX, no Python runtime

#include "dit.h"
#include "dots_tts.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <cstdio>

// ===========================================================================
// Utilities
// ===========================================================================

static float * tensor_data(ggml_tensor * t) {
    return (float *)((char *)t->data + t->view_offs);
}

static void debug_tensor(const char * name, ggml_tensor * t) {
    if (!t) { printf("%s: NULL\n", name); return; }
    printf("%s: [%lld %lld %lld %lld] type=%d\n",
           name, (long long)t->ne[0], (long long)t->ne[1],
           (long long)t->ne[2], (long long)t->ne[3], t->type);
}

// ===========================================================================
// Timestep embedding - sinusoidal, same as diffusion models
// ===========================================================================

ggml_tensor * dit_timestep_embedding(ggml_context * ctx, ggml_tensor * t, int dim) {
    // t: scalar [1] or batch [n_batch, 1]
    // compute sinusoidal embedding:
    //   half = dim / 2
    //   freqs = exp(-log(10000) * arange(0, half) / (half - 1))
    //   emb = cat([sin(t * freqs), cos(t * freqs)])
    int n_batch = t->ne[0];  // number of scalar timesteps
    int half_dim = dim / 2;

    // Create frequencies: pre-computed float array
    ggml_tensor * freqs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, half_dim);
    {
        float * fd = tensor_data(freqs);
        for (int i = 0; i < half_dim; i++) {
            fd[i] = expf(-logf(10000.0f) * (float)i / (float)(half_dim > 1 ? half_dim - 1 : 1));
        }
    }

    // t * freqs: broadcast t [n_batch, 1] * freqs [half_dim]
    // We'll do it as: repeat t to [n_batch, half_dim], then mul with freqs
    ggml_tensor * t_expanded = ggml_repeat(ctx, t, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, half_dim));
    ggml_reshape_2d(ctx, t_expanded, half_dim, n_batch);

    ggml_tensor * freq_expanded = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, half_dim);
    memcpy(tensor_data(freq_expanded), tensor_data(freqs), half_dim * sizeof(float));
    freq_expanded = ggml_repeat(ctx, freq_expanded, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_batch));
    ggml_reshape_2d(ctx, freq_expanded, half_dim, n_batch);

    ggml_tensor * t_freq = ggml_mul(ctx, t_expanded, freq_expanded);

    // sin and cos
    ggml_tensor * emb_sin = ggml_sin(ctx, t_freq);
    ggml_tensor * emb_cos = ggml_cos(ctx, t_freq);

    // concat: [sin, cos] -> [dim, n_batch]
    ggml_tensor * emb = ggml_concat(ctx, emb_sin, emb_cos, 0);

    return emb;  // [dim, n_batch]
}

// ===========================================================================
// adaLN modulation: given conditioning vector c [ada_dim, n_batch]
// compute shift/scale/gate for each DiT sub-layer
// ===========================================================================

static void adaln_modulate(
    ggml_context * ctx,
    ggml_tensor * x,           // [seq_len, n_batch, hidden]
    ggml_tensor * c,           // [ada_dim, n_batch]
    ggml_tensor * shift_weight,// [ada_dim, hidden]  - adaLN linear output slice
    ggml_tensor * scale_weight,// [ada_dim, hidden]
    ggml_tensor * gate_weight, // [ada_dim, hidden]
    ggml_tensor * shift_bias,  // [hidden]
    ggml_tensor * scale_bias,  // [hidden]
    ggml_tensor * gate_bias,   // [hidden]
    ggml_tensor ** out_shift,
    ggml_tensor ** out_scale,
    ggml_tensor ** out_gate
) {
    int seq_len  = x->ne[0];
    int n_batch  = x->ne[1];
    int hidden   = x->ne[2];

    // compute: c [ada_dim, batch] @ weight [ada_dim, hidden] + bias [hidden]
    // -> [batch, hidden]
    ggml_tensor * shift_mod = ggml_mul_mat(ctx, shift_weight, c); // [hidden, batch]
    if (shift_bias) {
        shift_mod = ggml_add(ctx, shift_mod,
            ggml_repeat(ctx, shift_bias,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_batch)));
    }
    shift_mod = ggml_cont(ctx, ggml_permute(ctx, shift_mod, 1, 0, 2, 3)); // [batch, hidden]

    ggml_tensor * scale_mod = ggml_mul_mat(ctx, scale_weight, c);
    if (scale_bias) {
        scale_mod = ggml_add(ctx, scale_mod,
            ggml_repeat(ctx, scale_bias,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_batch)));
    }
    scale_mod = ggml_cont(ctx, ggml_permute(ctx, scale_mod, 1, 0, 2, 3));

    ggml_tensor * gate_mod = ggml_mul_mat(ctx, gate_weight, c);
    if (gate_bias) {
        gate_mod = ggml_add(ctx, gate_mod,
            ggml_repeat(ctx, gate_bias,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_batch)));
    }
    gate_mod = ggml_cont(ctx, ggml_permute(ctx, gate_mod, 1, 0, 2, 3));

    // broadcast to [seq_len, n_batch, hidden]
    *out_shift = ggml_repeat(ctx, shift_mod, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len));
    ggml_reshape_3d(ctx, *out_shift, hidden, n_batch, seq_len);
    *out_shift = ggml_cont(ctx, ggml_permute(ctx, *out_shift, 2, 1, 0, 3));

    *out_scale = ggml_repeat(ctx, scale_mod, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len));
    ggml_reshape_3d(ctx, *out_scale, hidden, n_batch, seq_len);
    *out_scale = ggml_cont(ctx, ggml_permute(ctx, *out_scale, 2, 1, 0, 3));

    *out_gate = ggml_repeat(ctx, gate_mod, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len));
    ggml_reshape_3d(ctx, *out_gate, hidden, n_batch, seq_len);
    *out_gate = ggml_cont(ctx, ggml_permute(ctx, *out_gate, 2, 1, 0, 3));
}

// ===========================================================================
// RMS Norm with adaLN: norm(x) * (1 + scale) + shift
// ===========================================================================

static ggml_tensor * adaln_norm(
    ggml_context * ctx,
    ggml_tensor * x,         // [seq_len, n_batch, hidden]
    ggml_tensor * norm_w,    // [hidden]
    ggml_tensor * shift,     // [seq_len, n_batch, hidden]  adaLN shift modulation
    ggml_tensor * scale,     // [seq_len, n_batch, hidden]  adaLN scale modulation
    float eps = 1e-6f
) {
    int hidden = x->ne[2];

    // RMS norm with weight
    ggml_tensor * xn = ggml_rms_norm(ctx, x, eps);
    if (norm_w) {
        xn = ggml_mul(ctx, xn,
            ggml_repeat(ctx, norm_w,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, x->ne[0] * x->ne[1])));
        xn = ggml_reshape_3d(ctx, xn, hidden, x->ne[1], x->ne[0]);
        xn = ggml_cont(ctx, ggml_permute(ctx, xn, 2, 1, 0, 3));
    }

    // (1 + scale) * norm(x) + shift
    ggml_tensor * one = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, 1, 1);
    float val = 1.0f;
    memcpy(tensor_data(one), &val, sizeof(float));
    // set all elements to 1.0f
    for (int i = 1; i < hidden; i++) {
        ((float*)tensor_data(one))[i] = 1.0f;
    }
    one = ggml_repeat(ctx, one, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, x->ne[1]));
    one = ggml_reshape_3d(ctx, one, hidden, x->ne[1], 1);
    one = ggml_repeat(ctx, one, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, x->ne[0]));
    one = ggml_reshape_3d(ctx, one, hidden, x->ne[1], x->ne[0]);
    one = ggml_cont(ctx, ggml_permute(ctx, one, 2, 1, 0, 3));

    ggml_tensor * scale_1 = ggml_add(ctx, one, scale);
    ggml_tensor * normed = ggml_mul(ctx, scale_1, xn);

    return ggml_add(ctx, normed, shift);
}

// ===========================================================================
// Self-attention block for DiT (with RoPE and qk_norm)
// ===========================================================================

static ggml_tensor * dit_attention(
    ggml_context * ctx,
    ggml_tensor * x,            // [seq_len, n_batch, hidden]
    ggml_tensor * qkv_weight,   // [hidden, 3*hidden]
    ggml_tensor * qkv_bias,     // [3*hidden]
    ggml_tensor * o_weight,     // [hidden, hidden]
    ggml_tensor * o_bias,       // [hidden]
    ggml_tensor * q_norm_w,     // [head_dim]
    ggml_tensor * k_norm_w,     // [head_dim]
    int n_heads,
    int head_dim,
    ggml_tensor * k_cache       // [seq_len, n_batch, n_kv_heads, head_dim] for KV cache or nullptr
) {
    int seq_len  = x->ne[0];
    int n_batch  = x->ne[1];
    int hidden   = n_heads * head_dim;

    // QKV projection: [seq_len, n_batch, hidden] @ [hidden, 3*hidden] -> [seq_len, n_batch, 3*hidden]
    // reshape to [seq_len*n_batch, hidden] for matmul
    ggml_tensor * x_flat = ggml_reshape_2d(ctx, x, hidden, seq_len * n_batch);
    ggml_tensor * qkv = ggml_mul_mat(ctx, qkv_weight, x_flat);  // [3*hidden, seq_len*n_batch]
    if (qkv_bias) {
        ggml_tensor * bias = ggml_repeat(ctx, qkv_bias,
            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len * n_batch));
        qkv = ggml_add(ctx, qkv, bias);
    }
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 1, 0, 2, 3));
    qkv = ggml_reshape_3d(ctx, qkv, hidden, n_batch, seq_len);
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 2, 1, 0, 3)); // [seq_len, n_batch, 3*hidden]

    // Split Q, K, V
    int q_size = n_heads * head_dim;
    ggml_tensor * q = ggml_view_3d(ctx, qkv, q_size, n_batch, seq_len,
        qkv->nb[1], qkv->nb[2], 0);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 2, 1, 0, 3)); // [seq_len, n_batch, n_heads*head_dim]

    ggml_tensor * k = ggml_view_3d(ctx, qkv, q_size, n_batch, seq_len,
        qkv->nb[1], qkv->nb[2], q_size * sizeof(float));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 2, 1, 0, 3));

    ggml_tensor * v = ggml_view_3d(ctx, qkv, q_size, n_batch, seq_len,
        qkv->nb[1], qkv->nb[2], 2 * q_size * sizeof(float));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 2, 1, 0, 3));

    // Reshape to multi-head: [seq_len, n_batch, n_heads, head_dim]
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, seq_len * n_batch);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [head_dim, seq_len*n_batch, n_heads]
    q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 0, 2, 3)); // [seq_len*n_batch, head_dim, n_heads]

    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, seq_len * n_batch);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 0, 2, 3));

    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, seq_len * n_batch);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));

    // qk_norm: norm each head separately
    if (q_norm_w) {
        q = ggml_rms_norm(ctx, q, 1e-6f);
        q = ggml_mul(ctx, q,
            ggml_repeat(ctx, q_norm_w,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len * n_batch * n_heads)));
    }
    if (k_norm_w) {
        k = ggml_rms_norm(ctx, k, 1e-6f);
        k = ggml_mul(ctx, k,
            ggml_repeat(ctx, k_norm_w,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len * n_batch * n_heads)));
    }

    // RoPE: apply rotary position embedding
    // q,k: [seq_len*n_batch, head_dim, n_heads]
    // positions: [seq_len] — 0, 1, ..., seq_len-1
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    {
        int * pd = (int *)tensor_data(pos);
        for (int i = 0; i < seq_len; i++) pd[i] = i;
    }
    pos = ggml_repeat(ctx, pos, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_batch * n_heads));
    // pos: [seq_len * n_batch * n_heads]

    // We need q and k in [seq_len*n_batch, n_heads, head_dim] for ggml_rope
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [seq_len*n_batch, n_heads, head_dim]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

    q = ggml_rope(ctx, q, pos, head_dim, 0);
    k = ggml_rope(ctx, k, pos, head_dim, 0);

    // Attention: Q @ K^T -> softmax -> * V
    // Q: [seq_len*n_batch, n_heads, head_dim]    -> [seq_len, n_batch, n_heads, head_dim]
    q = ggml_reshape_4d(ctx, q, head_dim, n_heads, seq_len, n_batch);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 2, 0, 1, 3)); // [seq_len, head_dim, n_heads, n_batch]
    // reshape to [head_dim, seq_len * n_heads, n_batch] for batch matmul
    q = ggml_reshape_3d(ctx, q, head_dim, seq_len * n_heads, n_batch);
    // transpose to [head_dim, n_batch, seq_len * n_heads]
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    // reshape to [head_dim, n_batch * seq_len * n_heads]
    q = ggml_reshape_2d(ctx, q, head_dim, n_batch * seq_len * n_heads);

    k = ggml_reshape_4d(ctx, k, head_dim, n_heads, seq_len, n_batch);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 2, 0, 1, 3));
    k = ggml_reshape_3d(ctx, k, head_dim, seq_len * n_heads, n_batch);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    k = ggml_reshape_2d(ctx, k, head_dim, n_batch * seq_len * n_heads);

    // scores = q^T @ k: q is [head_dim, N], k is [head_dim, N]
    // We need [seq_len, seq_len] attention pattern per head per batch
    // This is a complex reshape; let's use flash attention approach
    // For MVP, implement simple matmul-based attention

    // Reshape for batched matmul: K as [seq_len, head_dim * n_heads * n_batch]
    // Q as [head_dim, ...]  — actually let's do simpler approach

    // Q @ K^T: Q [seq_len, head_dim, n_heads*batch], K [seq_len, head_dim, n_heads*batch]
    // For manual attention: reshape to [seq_len, head_dim] per (head, batch), do matmul

    // Let's use the standard approach: 
    // Q: [seq_len, n_heads, head_dim] per batch
    // K^T: [head_dim, seq_len] -> scores [seq_len, seq_len]

    // For now, reshape back to 3D and do per-batch matmul
    // Q: [head_dim, seq_len, n_heads * n_batch]
    // K: [head_dim, seq_len, n_heads * n_batch]

    q = ggml_cont(ctx, ggml_permute(ctx,
        ggml_reshape_3d(ctx, q, head_dim, seq_len, n_heads * n_batch),
        0, 2, 1, 3)); // [head_dim, n_heads*n_batch, seq_len]
    q = ggml_reshape_2d(ctx, q, head_dim, n_heads * n_batch * seq_len);

    k = ggml_cont(ctx, ggml_permute(ctx,
        ggml_reshape_3d(ctx, k, head_dim, seq_len, n_heads * n_batch),
        0, 2, 1, 3));
    k = ggml_reshape_2d(ctx, k, head_dim, n_heads * n_batch * seq_len);

    // scores = K^T @ Q: K [head_dim, N], Q [head_dim, N] -> [N, N] per (head, batch)
    // Since we flattened heads*batch*seq_len, we need a different approach
    // Use ggml_mul_mat: Q [head_dim, M] * K [head_dim, M]? No, need Q [seq_len, head_dim], K [head_dim, seq_len]

    // Better approach: use the Q [seq_len, head_dim] shape
    // Reshape K to [head_dim, seq_len * n_heads * n_batch]
    // But Q needs to be [seq_len, head_dim * n_heads * n_batch] for correct matmul
    // Actually let's simplify: compute per batch item for MVP

    // K^T: [n_heads*n_batch*seq_len, head_dim] -> transpose to [head_dim, n_heads*n_batch*seq_len]
    ggml_tensor * kt = ggml_cont(ctx, ggml_permute(ctx,
        ggml_reshape_2d(ctx,
            ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, k, head_dim, seq_len, n_heads * n_batch),
                1, 0, 2, 3)),
            seq_len, head_dim * n_heads * n_batch),
        1, 0, 2, 3)); // [head_dim * n_heads * n_batch, seq_len]

    // Q: [seq_len, head_dim * n_heads * n_batch]
    ggml_tensor * q2 = ggml_reshape_2d(ctx,
        ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, q, head_dim, seq_len, n_heads * n_batch),
            1, 0, 2, 3)),
        seq_len, head_dim * n_heads * n_batch);

    // scores = Q @ K^T: [seq_len, seq_len]  (but NOT per head — we need per-head)
    // This approach doesn't separate heads. For MVP let's simplify by using just the
    // reshape and standard ggml_mul_mat
    // Actually, the correct way: 
    // Q [seq_len, n_heads, head_dim] -> [seq_len*n_heads, head_dim]
    // K [seq_len, n_heads, head_dim] -> [seq_len*n_heads, head_dim] 
    // K^T -> [head_dim, seq_len*n_heads]
    // scores = Q @ K^T -> [seq_len*n_heads, seq_len*n_heads]  — NO, wrong
    // 
    // The simplest correct approach for MVP: loop over heads
    // But that's slow. Instead, let's just do a reduced version.
    //
    // For now: implement with explicit head loop (MVP quality, optimize later)

    // Flatten heads into batch for matrix multiply:
    // Q: [seq_len, n_heads*n_batch, head_dim]
    // K: [seq_len, n_heads*n_batch, head_dim] -> transpose -> [head_dim, seq_len, n_heads*n_batch]
    // scores = matmul(Q, K^T) in batch mode -> [seq_len, seq_len, n_heads*n_batch]

    ggml_tensor * q_bmm = ggml_reshape_3d(ctx,
        ggml_cont(ctx, ggml_permute(ctx, q, 0, 1, 2, 3)),
        head_dim, n_heads * n_batch, seq_len);
    q_bmm = ggml_cont(ctx, ggml_permute(ctx, q_bmm, 2, 1, 0, 3)); // [seq_len, n_heads*n_batch, head_dim]

    ggml_tensor * k_bmm = ggml_reshape_3d(ctx,
        ggml_cont(ctx, ggml_permute(ctx, k, 0, 1, 2, 3)),
        head_dim, n_heads * n_batch, seq_len);
    k_bmm = ggml_cont(ctx, ggml_permute(ctx, k_bmm, 2, 1, 0, 3)); // [seq_len, n_heads*n_batch, head_dim]
    k_bmm = ggml_cont(ctx, ggml_permute(ctx, k_bmm, 1, 0, 2, 3)); // [n_heads*n_batch, seq_len, head_dim]

    // scores [seq_len, n_heads*n_batch, seq_len]
    ggml_tensor * scores = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, k_bmm, head_dim, n_heads * n_batch * seq_len),
        ggml_reshape_2d(ctx, q_bmm, head_dim, seq_len * n_heads * n_batch));
    // scores is [seq_len, seq_len*n_heads*n_batch]
    scores = ggml_reshape_3d(ctx, scores, seq_len, n_heads * n_batch, seq_len);
    scores = ggml_cont(ctx, ggml_permute(ctx, scores, 0, 2, 1, 3)); // [seq_len, seq_len, n_heads*n_batch]

    // Scale
    float scale = 1.0f / sqrtf((float)head_dim);
    scores = ggml_scale(ctx, scores, scale);

    // Causal mask (for autoregressive - optional for DiT)
    // For DiT, the full sequence is visible — no causal mask needed for FM conditioning
    // But if used in AR mode, add causal mask

    // Softmax over seq_len dimension (dim 1)
    scores = ggml_soft_max(ctx, scores);

    // V: [seq_len, n_heads*n_batch, head_dim]
    ggml_tensor * v_bmm = ggml_reshape_3d(ctx,
        ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, v, head_dim, n_heads * n_batch, seq_len),
            2, 1, 0, 3)),
        head_dim, n_heads * n_batch, seq_len);
    v_bmm = ggml_cont(ctx, ggml_permute(ctx, v_bmm, 2, 1, 0, 3)); // [seq_len, n_heads*n_batch, head_dim]
    // reshape for matmul: [head_dim, seq_len*n_heads*n_batch]
    v_bmm = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx,
        v_bmm, 2, 0, 1, 3)), seq_len, n_heads * n_batch * head_dim);
    v_bmm = ggml_cont(ctx, ggml_permute(ctx, v_bmm, 1, 0, 2, 3)); // [n_heads*n_batch*head_dim, seq_len]

    // out = scores @ V: scores [seq_len, seq_len, n_heads*n_batch]
    // Reshape scores: [seq_len*n_heads*n_batch, seq_len]
    scores = ggml_reshape_2d(ctx, scores, seq_len, seq_len * n_heads * n_batch);
    ggml_tensor * attn_out = ggml_mul_mat(ctx,
        v_bmm,
        scores); // [n_heads*n_batch*head_dim, seq_len*n_heads*n_batch]
    attn_out = ggml_reshape_3d(ctx, attn_out, head_dim, n_heads * n_batch, seq_len);
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 2, 0, 1, 3)); // [seq_len, head_dim, n_heads*n_batch]
    attn_out = ggml_reshape_3d(ctx, attn_out, hidden, n_batch, seq_len); // merge heads
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 2, 1, 0, 3)); // [seq_len, n_batch, hidden]

    // Output projection
    attn_out = ggml_reshape_2d(ctx, attn_out, hidden, seq_len * n_batch);
    attn_out = ggml_mul_mat(ctx, o_weight, attn_out);
    if (o_bias) {
        attn_out = ggml_add(ctx, attn_out,
            ggml_repeat(ctx, o_bias, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len * n_batch)));
    }
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 1, 0, 2, 3));
    attn_out = ggml_reshape_3d(ctx, attn_out, hidden, n_batch, seq_len);
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 2, 1, 0, 3));

    return attn_out;
}

// ===========================================================================
// DiT Block forward
// ===========================================================================

static ggml_tensor * dit_block_forward(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * cond,        // [ada_dim, n_batch]
    dit_block & block,
    ggml_tensor * k_cache
) {
    // adaLN modulation: Linear(cond) -> [6*hidden]
    // then split into shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    int hidden = DIT_HIDDEN_SIZE;
    int n_batch = x->ne[1];
    int seq_len = x->ne[0];

    // Compute full adaLN output: cond @ adaln_linear_w + bias -> [6*hidden, n_batch]
    ggml_tensor * adaln_out = ggml_mul_mat(ctx, block.adaln_linear_w, cond);
    if (block.adaln_linear_b) {
        adaln_out = ggml_add(ctx, adaln_out,
            ggml_repeat(ctx, block.adaln_linear_b,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_batch)));
    }
    adaln_out = ggml_cont(ctx, ggml_permute(ctx, adaln_out, 1, 0, 2, 3)); // [n_batch, 6*hidden]

    // View slices for each modulation
    // Layout: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] each [hidden]
    // In the contiguous tensor: bytes offset for each slice
    size_t stride = hidden * sizeof(float);

    ggml_tensor * shift_msa = ggml_view_2d(ctx, adaln_out, hidden, n_batch,
        adaln_out->nb[1], 0); // bytes offset 0
    shift_msa = ggml_repeat(ctx, shift_msa, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len));
    ggml_reshape_3d(ctx, shift_msa, hidden, n_batch, seq_len);
    shift_msa = ggml_cont(ctx, ggml_permute(ctx, shift_msa, 2, 1, 0, 3));

    ggml_tensor * scale_msa = ggml_view_2d(ctx, adaln_out, hidden, n_batch,
        adaln_out->nb[1], stride);
    scale_msa = ggml_repeat(ctx, scale_msa, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len));
    ggml_reshape_3d(ctx, scale_msa, hidden, n_batch, seq_len);
    scale_msa = ggml_cont(ctx, ggml_permute(ctx, scale_msa, 2, 1, 0, 3));

    ggml_tensor * gate_msa = ggml_view_2d(ctx, adaln_out, hidden, n_batch,
        adaln_out->nb[1], 2 * stride);
    gate_msa = ggml_repeat(ctx, gate_msa, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len));
    ggml_reshape_3d(ctx, gate_msa, hidden, n_batch, seq_len);
    gate_msa = ggml_cont(ctx, ggml_permute(ctx, gate_msa, 2, 1, 0, 3));

    ggml_tensor * shift_mlp = ggml_view_2d(ctx, adaln_out, hidden, n_batch,
        adaln_out->nb[1], 3 * stride);
    shift_mlp = ggml_repeat(ctx, shift_mlp, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len));
    ggml_reshape_3d(ctx, shift_mlp, hidden, n_batch, seq_len);
    shift_mlp = ggml_cont(ctx, ggml_permute(ctx, shift_mlp, 2, 1, 0, 3));

    ggml_tensor * scale_mlp = ggml_view_2d(ctx, adaln_out, hidden, n_batch,
        adaln_out->nb[1], 4 * stride);
    scale_mlp = ggml_repeat(ctx, scale_mlp, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len));
    ggml_reshape_3d(ctx, scale_mlp, hidden, n_batch, seq_len);
    scale_mlp = ggml_cont(ctx, ggml_permute(ctx, scale_mlp, 2, 1, 0, 3));

    ggml_tensor * gate_mlp = ggml_view_2d(ctx, adaln_out, hidden, n_batch,
        adaln_out->nb[1], 5 * stride);
    gate_mlp = ggml_repeat(ctx, gate_mlp, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len));
    ggml_reshape_3d(ctx, gate_mlp, hidden, n_batch, seq_len);
    gate_mlp = ggml_cont(ctx, ggml_permute(ctx, gate_mlp, 2, 1, 0, 3));

    // --- Self-attention ---
    ggml_tensor * h = adaln_norm(ctx, x, block.attn_norm_w, shift_msa, scale_msa);
    ggml_tensor * attn_out = dit_attention(ctx, h,
        block.attn_qkv_weight, block.attn_qkv_bias,
        block.attn_o_weight, block.attn_o_bias,
        block.q_norm_w, block.k_norm_w,
        DIT_NUM_HEADS, DIT_HEAD_SIZE,
        k_cache);
    x = ggml_add(ctx, x, ggml_mul(ctx, gate_msa, attn_out));

    // --- FFN ---
    h = adaln_norm(ctx, x, block.ffn_norm_w, shift_mlp, scale_mlp);

    // FFN: Linear -> SiLU -> Linear
    ggml_tensor * h_flat = ggml_reshape_2d(ctx, h, hidden, seq_len * n_batch);
    ggml_tensor * ffn_h = ggml_mul_mat(ctx, block.ffn_w1, h_flat); // [ffn, seq_len*n_batch]
    ffn_h = ggml_silu(ctx, ffn_h);
    ffn_h = ggml_mul_mat(ctx, block.ffn_w2, ffn_h); // [hidden, seq_len*n_batch]
    ffn_h = ggml_cont(ctx, ggml_permute(ctx, ffn_h, 1, 0, 2, 3));
    ffn_h = ggml_reshape_3d(ctx, ffn_h, hidden, n_batch, seq_len);
    ffn_h = ggml_cont(ctx, ggml_permute(ctx, ffn_h, 2, 1, 0, 3));

    x = ggml_add(ctx, x, ggml_mul(ctx, gate_mlp, ffn_h));

    return x;
}

// ===========================================================================
// Full DiT forward pass
// ===========================================================================

ggml_tensor * dit_forward(
    dit_model & model,
    ggml_context * ctx,
    ggml_tensor * x,           // [seq_len, n_batch, hidden]
    ggml_tensor * t,           // [n_batch] timestep in [0, 1]
    ggml_tensor * speaker_emb) // [speaker_dim, n_batch] or nullptr
{
    int seq_len = x->ne[0];
    int n_batch = x->ne[1];

    // Compute timestep embedding
    ggml_tensor * t_emb = dit_timestep_embedding(ctx, t, DIT_ADA_DIM);

    // Combine timestep + speaker embedding
    if (speaker_emb) {
        // Project speaker: spk_proj_w [speaker_dim, ada_dim]
        ggml_tensor * spk_proj = ggml_mul_mat(ctx, model.spk_proj_w, speaker_emb);
        // spk_proj [ada_dim, n_batch]
        t_emb = ggml_add(ctx, t_emb, spk_proj);
    }

    // SiLU on combined embedding
    ggml_tensor * cond = ggml_silu(ctx, t_emb); // [ada_dim, n_batch]

    // Pass through DiT blocks
    for (int i = 0; i < model.n_layers; i++) {
        x = dit_block_forward(ctx, x, cond, model.layers[i], nullptr);
    }

    // Final layer norm + output projection
    // (dots.tts applies final norm + linear -> latent_dim at the end)

    // Output projection: Linear(hidden -> latent_dim)
    ggml_tensor * x_flat = ggml_reshape_2d(ctx, x, DIT_HIDDEN_SIZE, seq_len * n_batch);
    ggml_tensor * out = ggml_mul_mat(ctx, model.out_proj_w, x_flat);
    if (model.out_proj_b) {
        out = ggml_add(ctx, out,
            ggml_repeat(ctx, model.out_proj_b,
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len * n_batch)));
    }
    out = ggml_cont(ctx, ggml_permute(ctx, out, 1, 0, 2, 3));

    // Reshape to [latent_patch_size, latent_dim]
    out = ggml_reshape_2d(ctx, out, VAE_LATENT_DIM, seq_len);

    return out;
}

// ===========================================================================
// Model loading
// ===========================================================================

bool dit_model_load(dit_model & model, ggml_context * ctx) {
    model.layers.resize(model.n_layers);

    for (int i = 0; i < model.n_layers; i++) {
        char name[256];
        dit_block & b = model.layers[i];

        snprintf(name, sizeof(name), "dit.layers.%d.attn.qkv.weight", i);
        b.attn_qkv_weight = ggml_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "dit.layers.%d.attn.qkv.bias", i);
        b.attn_qkv_bias = ggml_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "dit.layers.%d.attn.o.weight", i);
        b.attn_o_weight = ggml_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "dit.layers.%d.attn.o.bias", i);
        b.attn_o_bias = ggml_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "dit.layers.%d.q_norm.weight", i);
        b.q_norm_w = ggml_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "dit.layers.%d.k_norm.weight", i);
        b.k_norm_w = ggml_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "dit.layers.%d.ffn.w1.weight", i);
        b.ffn_w1 = ggml_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "dit.layers.%d.ffn.w2.weight", i);
        b.ffn_w2 = ggml_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "dit.layers.%d.adaln.linear.weight", i);
        b.adaln_linear_w = ggml_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "dit.layers.%d.adaln.linear.bias", i);
        b.adaln_linear_b = ggml_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "dit.layers.%d.attn_norm.weight", i);
        b.attn_norm_w = ggml_get_tensor(ctx, name);

        snprintf(name, sizeof(name), "dit.layers.%d.ffn_norm.weight", i);
        b.ffn_norm_w = ggml_get_tensor(ctx, name);
    }

    model.spk_proj_w  = ggml_get_tensor(ctx, "dit.spk_proj.weight");
    model.out_proj_w  = ggml_get_tensor(ctx, "dit.out_proj.weight");
    model.out_proj_b  = ggml_get_tensor(ctx, "dit.out_proj.bias");

    return true;
}
