// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// dots.tts.cpp - Multi-head attention via ggml_flash_attn_ext
// Proven approach used by llama.cpp. No manual loops, no NaN.
// Now with Q/K/V/O bias support.

#include "dots_tts.h"
#include "dots_tts_util.h"
#include "ggml.h"
#include <cmath>
#include <cstring>

ggml_tensor * dit_attention_multihead(
    ggml_context * ctx,
    ggml_tensor * x,           // [hidden, n_tokens]
    ggml_tensor * q_weight,    // [hidden, hidden]
    ggml_tensor * k_weight,    // [hidden, hidden]
    ggml_tensor * v_weight,    // [hidden, hidden]
    ggml_tensor * o_weight,    // [hidden, hidden]
    ggml_tensor * q_bias,      // [hidden] or nullptr
    ggml_tensor * k_bias,      // [hidden] or nullptr
    ggml_tensor * v_bias,      // [hidden] or nullptr
    ggml_tensor * o_bias,      // [hidden] or nullptr
    int seq_len, int n_batch,
    int n_heads, int head_dim,
    ggml_tensor * q_norm_w, ggml_tensor * k_norm_w)
{
    int hidden = n_heads * head_dim;
    int n_tokens = seq_len * n_batch;

    // QKV projections
    ggml_tensor * q = ggml_mul_mat(ctx, q_weight, x);
    if (q_bias) q = ggml_add(ctx, q, q_bias);
    ggml_tensor * k = ggml_mul_mat(ctx, k_weight, x);
    if (k_bias) k = ggml_add(ctx, k, k_bias);
    ggml_tensor * v = ggml_mul_mat(ctx, v_weight, x);
    if (v_bias) v = ggml_add(ctx, v, v_bias);
    q = ggml_cont(ctx, q); k = ggml_cont(ctx, k); v = ggml_cont(ctx, v);

    // Reshape to [head_dim, n_heads, n_tokens] — format required by flash_attn
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, n_tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, n_tokens);

    // qk_norm
    if (q_norm_w) { q = ggml_rms_norm(ctx, q, 1e-6f); q = ggml_mul(ctx, q, q_norm_w); }
    if (k_norm_w) { k = ggml_rms_norm(ctx, k, 1e-6f); k = ggml_mul(ctx, k, k_norm_w); }

    // RoPE
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    { int * pd = (int*)((char*)pos->data); for (int i=0;i<n_tokens;i++) pd[i]=i%seq_len; }
    q = ggml_rope(ctx, q, pos, head_dim, 0);
    k = ggml_rope(ctx, k, pos, head_dim, 0);

    // Permute for flash_attn: [head_dim, n_tokens, n_heads]
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    // Flash attention: Q @ K^T @ V, all heads in one op
    float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);

    // Reshape back: [head_dim, n_tokens, n_heads] -> [hidden, n_tokens]
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)); // [head_dim, n_heads, n_tokens]
    attn = ggml_reshape_2d(ctx, attn, hidden, n_tokens); // [hidden, n_tokens]

    // Output projection + bias
    attn = ggml_mul_mat(ctx, o_weight, attn);
    if (o_bias) attn = ggml_add(ctx, attn, o_bias);
    return attn;
}
