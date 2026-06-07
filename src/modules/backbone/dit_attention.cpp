// dots.tts.cpp - Proper multi-head attention for DiT
// Uses ggml batch dimension (ne[2]=n_heads) for per-head computation

#include "dots_tts.h"
#include "dots_tts_util.h"
#include "ggml.h"
#include <cmath>

ggml_tensor * dit_attention_multihead(
    ggml_context * ctx,
    ggml_tensor * x,           // [hidden, n_tokens]
    ggml_tensor * q_weight,    // [hidden, hidden]  
    ggml_tensor * k_weight,    // [hidden, hidden]
    ggml_tensor * v_weight,    // [hidden, hidden]
    ggml_tensor * o_weight,    // [hidden, hidden]
    int seq_len, int n_batch,
    int n_heads, int head_dim,
    ggml_tensor * q_norm_w, ggml_tensor * k_norm_w)
{
    int hidden = n_heads * head_dim;
    int n_tokens = seq_len * n_batch;

    // QKV projections
    ggml_tensor * q = ggml_mul_mat(ctx, q_weight, x); // [hidden, n_tokens]
    ggml_tensor * k = ggml_mul_mat(ctx, k_weight, x);
    ggml_tensor * v = ggml_mul_mat(ctx, v_weight, x);
    q = ggml_cont(ctx, q); k = ggml_cont(ctx, k); v = ggml_cont(ctx, v);

    // Reshape to [head_dim, n_heads, n_tokens]
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, n_tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, n_tokens);

    // qk_norm per head
    if (q_norm_w) { q = ggml_rms_norm(ctx, q, 1e-6f); q = ggml_mul(ctx, q, q_norm_w); }
    if (k_norm_w) { k = ggml_rms_norm(ctx, k, 1e-6f); k = ggml_mul(ctx, k, k_norm_w); }

    // RoPE
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    { int * pd = (int*)((char*)pos->data); for (int i=0;i<n_tokens;i++) pd[i]=i%seq_len; }
    q = ggml_rope(ctx, q, pos, head_dim, 0);
    k = ggml_rope(ctx, k, pos, head_dim, 0);

    // Permute for batched matmul: [head_dim, n_tokens, n_heads]
    // mul_mat uses batch dim ne[2] for per-head computation
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [head_dim, n_tokens, n_heads]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    // Scores: K^T @ Q = [n_tokens, head_dim, n_heads]^T @ [head_dim, n_tokens, n_heads]
    // = [n_tokens, n_tokens, n_heads] — per-head attention scores
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q); // K^T @ Q, batched over n_heads
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float)head_dim));
    scores = ggml_soft_max(ctx, scores);

    // Output: scores @ V^T
    // scores: [n_tokens, n_tokens, n_heads]
    // V: [head_dim, n_tokens, n_heads] -> V^T: [n_tokens, head_dim, n_heads]
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [n_tokens, head_dim, n_heads]
    ggml_tensor * attn = ggml_mul_mat(ctx, scores, vt); // [n_tokens, head_dim, n_heads]

    // Reshape back: [n_tokens, head_dim, n_heads] -> [n_tokens, n_heads*head_dim]
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 1, 2, 0, 3)); // [head_dim, n_heads, n_tokens]
    attn = ggml_reshape_2d(ctx, attn, hidden, n_tokens); // [hidden, n_tokens]

    // Output projection
    attn = ggml_mul_mat(ctx, o_weight, attn);
    return attn; // [hidden, n_tokens]
}
