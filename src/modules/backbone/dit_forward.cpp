// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// dots.tts.cpp - DiT forward pass (simplified for ggml)
// All operations use either:
//   1. ggml_mul_mat for linear projections (2D: [out_features, batch*seq])
//   2. Element-wise ops on correctly shaped tensors  
//   3. ggml_rms_norm, ggml_silu, ggml_rope, ggml_soft_max

#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <cstdio>

// Multi-head attention (defined in dit_attention.cpp)
ggml_tensor * dit_attention_multihead(
    ggml_context * ctx, ggml_tensor * x,
    ggml_tensor * q_weight, ggml_tensor * k_weight, ggml_tensor * v_weight,
    ggml_tensor * o_weight, int seq_len, int n_batch,
    int n_heads, int head_dim, ggml_tensor * q_norm_w, ggml_tensor * k_norm_w);

// ===========================================================================
// Timestep embedding
// ===========================================================================

ggml_tensor * dit_timestep_embedding(ggml_context * ctx, ggml_tensor * t, int dim) {
    int half_dim = dim / 2;
    int n_batch = t->ne[0]; // assume t is [n_batch] flat

    ggml_tensor * emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n_batch);
    float * ed = tensor_data(emb);

    float * td = tensor_data(t);
    for (int b = 0; b < n_batch; b++) {
        float t_val = td[b];
        for (int i = 0; i < half_dim; i++) {
            float freq = expf(-logf(10000.0f) * (float)i / (float)(half_dim > 1 ? half_dim - 1 : 1));
            float arg = t_val * freq;
            ed[b * dim + i] = sinf(arg);
            ed[b * dim + half_dim + i] = cosf(arg);
        }
    }

    return emb;
}

// ===========================================================================
// FFN: two linear layers with SiLU in between
//   x [hidden, N] -> w1 [ffn, hidden] -> SiLU -> w2 [hidden, ffn] -> [hidden, N]
// ===========================================================================

static ggml_tensor * dit_ffn(
    ggml_context * ctx,
    ggml_tensor * x,        // [hidden, N]  (flattened batch*seq)
    ggml_tensor * w1,       // [ffn, hidden]
    ggml_tensor * w2        // [hidden, ffn]
) {
    ggml_tensor * h = ggml_mul_mat(ctx, w1, x);
    h = ggml_silu(ctx, h);
    return ggml_mul_mat(ctx, w2, h);
}

// ===========================================================================
// Self-attention (simplified: no KV cache, full self-attn)
//   x [hidden, seq_len*n_batch]
//   qkv_weight [hidden, 3*hidden]
//   o_weight [hidden, hidden]
// ===========================================================================

static ggml_tensor * dit_attention(
    ggml_context * ctx,
    ggml_tensor * x,           // [hidden, seq_len * n_batch]
    ggml_tensor * qkv_weight,  // [3*hidden, hidden] (merged) or nullptr
    ggml_tensor * q_weight,     // [hidden, hidden] (separate)
    ggml_tensor * k_weight,     // [hidden, hidden]
    ggml_tensor * v_weight,     // [hidden, hidden]
    ggml_tensor * o_weight,    // [hidden, hidden]
    int seq_len,
    int n_batch,
    int n_heads,
    int head_dim,
    ggml_tensor * q_norm_w,
    ggml_tensor * k_norm_w
) {
    int hidden  = n_heads * head_dim;
    int n_tokens = seq_len * n_batch;

    // QKV projection: separate or merged
    ggml_tensor * q, * k, * v;
    if (qkv_weight) {
        // Merged QKV: [3*hidden, n_tokens]
        ggml_tensor * qkv = ggml_mul_mat(ctx, qkv_weight, x);
        q = ggml_view_2d(ctx, qkv, hidden, n_tokens, qkv->nb[1], 0);
        k = ggml_view_2d(ctx, qkv, hidden, n_tokens, qkv->nb[1], hidden * sizeof(float));
        v = ggml_view_2d(ctx, qkv, hidden, n_tokens, qkv->nb[1], 2 * hidden * sizeof(float));
    } else {
        // Separate Q, K, V: each [hidden, n_tokens]
        q = ggml_mul_mat(ctx, q_weight, x);
        k = ggml_mul_mat(ctx, k_weight, x);
        v = ggml_mul_mat(ctx, v_weight, x);
    }

    // Ensure contiguity
    q = ggml_cont(ctx, q);
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    // Reshape for RoPE (4D then flatten)
    auto reshape_qkv_4d = [&](ggml_tensor * t) {
        t = ggml_cont(ctx, t);
        ggml_tensor * t4 = ggml_reshape_4d(ctx, t, head_dim, n_heads, seq_len, n_batch);
        // permute to [seq_len, n_heads, head_dim, n_batch]
        t4 = ggml_cont(ctx, ggml_permute(ctx, t4, 2, 0, 1, 3));
        // reshape to [seq_len * n_heads, head_dim, n_batch]
        t4 = ggml_reshape_3d(ctx, t4, head_dim, seq_len * n_heads, n_batch);
        // permute to [head_dim, n_batch, seq_len * n_heads]
        t4 = ggml_cont(ctx, ggml_permute(ctx, t4, 0, 2, 1, 3));
        // flatten to [head_dim, n_batch * seq_len * n_heads]
        return ggml_reshape_2d(ctx, t4, head_dim, n_batch * seq_len * n_heads);
    };

    q = reshape_qkv_4d(q);
    k = reshape_qkv_4d(k);
    v = reshape_qkv_4d(v);

    // qk_norm
    if (q_norm_w) {
        q = ggml_rms_norm(ctx, q, 1e-6f);
        q = ggml_mul(ctx, q, q_norm_w);
    }
    if (k_norm_w) {
        k = ggml_rms_norm(ctx, k, 1e-6f);
        k = ggml_mul(ctx, k, k_norm_w);
    }

    // RoPE: reshape to [head_dim, n_heads, n_tokens] (ggml_rope expects ne[2]=n_tokens)
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, n_tokens);

    // Position IDs: [n_tokens] with values 0..seq_len-1 repeated per batch
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    {
        int * pd = (int *)tensor_data(pos);
        for (int b = 0; b < n_batch; b++) {
            for (int s = 0; s < seq_len; s++) {
                pd[b * seq_len + s] = s;
            }
        }
    }

    q = ggml_rope(ctx, q, pos, head_dim, 0);
    k = ggml_rope(ctx, k, pos, head_dim, 0);

    // Compute attention scores: Q @ K^T / sqrt(d)
    // Q: [n_tokens, n_heads, head_dim] -> [n_heads, head_dim, n_tokens?]
    // For batched attention, reshape to [n_heads*head_dim, n_tokens]
    // Then: scores = K^T @ Q

    // Q: [n_tokens, n_heads, head_dim] -> flatten to [n_heads*head_dim, n_tokens]
    q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3)); // [n_heads, head_dim, n_tokens]
    q = ggml_reshape_2d(ctx, q, hidden, n_tokens); // [hidden, n_tokens]

    k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3)); // [n_heads, head_dim, n_tokens]
    k = ggml_reshape_2d(ctx, k, hidden, n_tokens); // [hidden, n_tokens]

    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    v = ggml_reshape_2d(ctx, v, hidden, n_tokens);

    // scores = K^T @ Q: K [hidden, n_tokens], Q [n_tokens, hidden]? No — we need per-token scores
    // The problem: standard matmul gives [hidden, hidden], not [n_tokens, n_tokens].
    // We need per-head attention. Let's reshape Q to [n_heads, head_dim, n_tokens]
    // and K to [n_heads, head_dim, n_tokens], then compute per-head: [n_heads, n_tokens, n_tokens]
    //
    // Simpler: broadcast by reshaping K to [n_heads*head_dim, n_tokens]
    // but we need K per head to multiply with Q per head giving [head_dim, head_dim] per head
    //
    // Actually: Q [n_heads, head_dim, n_tokens], K^T [n_heads, n_tokens, head_dim]
    // scores [n_heads, n_tokens, n_tokens]
    // 
    // In ggml: reshape Q to [head_dim, n_tokens*n_heads], K to [head_dim, n_tokens*n_heads]
    // but that loses the per-head separation...

    // Let's use a simpler approach: for MVP, treat heads as part of batch
    // Q: [head_dim, n_tokens * n_heads] (transposed)
    // K: [head_dim, n_tokens * n_heads]
    // Then scores = Q^T @ K, giving [n_tokens*n_heads, n_tokens*n_heads]
    // but that's cross-head, not per-head...

    // OK, actual approach: reshape Q to [n_heads*head_dim, n_tokens] and loop per head
    // For MVP, use a direct matmul approach with proper dimensions:
    //
    // Q: [n_tokens, n_heads, head_dim] -> [n_tokens, n_heads*head_dim]
    // K: [n_tokens, n_heads, head_dim] -> [n_tokens, n_heads*head_dim] -> K^T [n_heads*head_dim, n_tokens]
    // scores = Q @ K^T -> [n_tokens, n_tokens]... but this merges all heads
    //
    // For proper per-head attention in ggml, we use ggml_mul_mat with batch dimension:
    // Reshape to [n_heads, head_dim, n_tokens] for both Q and K
    // Treat n_heads as batch dim 1: Q [head_dim, n_tokens, n_heads], K [head_dim, n_tokens, n_heads]
    // K^T: [head_dim, n_heads, n_tokens]? No...

    // SIMPLEST CORRECT APPROACH: use 4D batch matmul
    // Q: [n_tokens, n_heads, head_dim] -> reshape to [n_tokens, 1, n_heads, head_dim]
    // Actually ggml doesn't do 4D batch matmul natively.
    //
    // The actual solution used in llama.cpp: contatenate heads with batch and seq,
    // use [head_dim, n_heads*seq_len] for Q, and [head_dim, n_heads*seq_len] for K,
    // then the matmul gives scores per position per head.
    //
    // How it works: for each token position, we have n_heads query vectors.
    // Q is stored as [head_dim, n_heads * n_tokens]
    // K is stored as [head_dim, n_heads * n_tokens]
    // BUT: K^T @ Q gives [n_heads*n_tokens, n_heads*n_tokens] — cross everything
    //
    // The key trick: reshape Q to [head_dim, n_tokens, n_heads], K to [head_dim, n_tokens, n_heads]
    // Then use ggml_mul_mat_id for batched matmul? No, that's for different weight matrices.
    //
    // For now, let me just implement correctly with a head-loop for MVP:

    // Prepare V: [hidden, n_tokens] — already done

    // Reshape for per-head computation
    // Q: [hidden, n_tokens] -> [n_heads, head_dim, n_tokens]
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
    // [n_heads, head_dim, n_tokens] -> we want [head_dim, n_tokens, n_heads] for matmul with K[head_dim, n_tokens]
    q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3)); // [head_dim, n_tokens, n_heads]
    q = ggml_reshape_2d(ctx, q, n_tokens * n_heads, head_dim); // [n_tokens*n_heads, head_dim]
    q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 0, 2, 3)); // [head_dim, n_tokens*n_heads]

    // K: same
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, n_tokens);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3)); // [head_dim, n_tokens, n_heads]
    // K^T: we need [n_tokens, head_dim, n_heads]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 0, 2, 3)); // [n_tokens, head_dim, n_heads]
    k = ggml_reshape_2d(ctx, k, n_tokens, head_dim * n_heads); // [n_tokens, head_dim*n_heads]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 0, 2, 3)); // [head_dim*n_heads, n_tokens]

    // scores = K^T @ Q: K [head_dim*n_heads, n_tokens], Q [head_dim, n_tokens*n_heads]
    // This is (n_heads * head_dim) x n_tokens times head_dim x (n_tokens * n_heads)
    // = (n_heads * head_dim) x (n_tokens * n_heads)? No, matmul gives [out_cols, out_rows] 
    // where out_cols = K_cols? No, mul_mat(W, X): X [cols, rows], W [out_cols, cols]
    // So scores = K^T @ Q: [head_dim*n_heads, n_tokens] @ [head_dim, n_tokens*n_heads]
    // Wait, the shapes don't match. head_dim*n_heads != head_dim (unless n_heads=1).
    //
    // I'm overcomplicating this. Let me use the approach from whisper.cpp attention:
    // Use ggml_mul_mat with Q [head_dim, n_heads*n_tokens] and K [head_dim, n_heads*n_tokens]
    // scores = K^T @ Q: [head_dim, n_heads*n_tokens]^T @ [head_dim, n_heads*n_tokens]
    // = [n_heads*n_tokens, n_heads*n_tokens]
    // Then softmax, then * V [head_dim, n_heads*n_tokens] -> [head_dim, n_heads*n_tokens]
    //
    // This works but the scores are n_heads*n_tokens x n_heads*n_tokens which is huge.
    // For 1024 tokens and 16 heads: 16384 x 16384 = 268M float. OK for CPU.
    //
    // Let me just do this correctly:

    // Q: [hidden, n_tokens] = [n_heads*head_dim, n_tokens]
    // K: [hidden, n_tokens] = [n_heads*head_dim, n_tokens]
    // scores = Q^T @ K: [n_tokens, n_heads*head_dim]^T @ [n_heads*head_dim, n_tokens]
    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 0, 2, 3));
    Q = ggml_reshape_2d(ctx, Q, n_heads * head_dim, n_tokens); // [hidden, n_tokens]

    ggml_tensor * K = k; // [head_dim*n_heads, n_tokens] — but K is [head_dim*n_heads, n_tokens]

    // Hmm, let me restart. After RoPE:
    // q: [n_tokens, n_heads, head_dim]
    // k: [n_tokens, n_heads, head_dim]

    // Flatten both to [n_heads*head_dim, n_tokens]
    ggml_tensor * Qn = ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3)); // [n_heads, head_dim, n_tokens]
    Qn = ggml_reshape_2d(ctx, Qn, n_heads * head_dim, n_tokens); // [hidden, n_tokens]

    ggml_tensor * Kn = ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3));
    Kn = ggml_reshape_2d(ctx, Kn, n_heads * head_dim, n_tokens);

    // scores = Q^T @ K: [n_tokens, hidden] @ [hidden, n_tokens] -> [n_tokens, n_tokens]
    // But this mixes per-head and per-token. The correct attention is per-head.
    // 
    // ACTUALLY: for multi-head attention with combined QKV, the formula is:
    // Each head: softmax(Q_h @ K_h^T / sqrt(d)) @ V_h
    // Combined: we can't do it with a single matmul.
    // 
    // The standard ggml approach (used in whisper.cpp, stable-diffusion.cpp):
    // 1. Reshape Q: [n_heads, head_dim, n_tokens] -> Q^T is [n_heads, n_tokens, head_dim]
    // 2. Use ggml_mul_mat for per-head: but ggml doesn't support per-head batching natively
    //    It uses ggml_mul_mat_id for batched matmul across the head dimension
    //
    // Wait, is there even a way to do this per-head? Let me check if ggml_mul_mat supports
    // batching when the first dimension is the batch:
    // Q: [n_heads, head_dim, n_tokens] -> [head_dim, n_heads, n_tokens] via permute
    // But mul_mat operates on last two dims...
    
    // Multi-head attention: per-head softmax(Q_h @ K_h^T / sqrt(d)) @ V_h
    // Q, K, V after RoPE: each [hidden, n_tokens] = [n_heads*head_dim, n_tokens]
    // Reshape to [n_heads, head_dim, n_tokens] for per-head processing
    q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3)); // [n_heads, head_dim, n_tokens]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    
    float scale = 1.0f / sqrtf((float)head_dim);
    int n_tokens_total = seq_len * n_batch;
    
    // Allocate output buffer [hidden, n_tokens]
    ggml_tensor * attn_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens_total, hidden);
    float * out_data = (float*)attn_out->data;
    
    for (int h = 0; h < n_heads; h++) {
        // Extract head h: [head_dim, n_tokens]
        ggml_tensor * qh = ggml_view_2d(ctx, q, head_dim, n_tokens_total, q->nb[1], h * head_dim * sizeof(float));
        ggml_tensor * kh = ggml_view_2d(ctx, k, head_dim, n_tokens_total, k->nb[1], h * head_dim * sizeof(float));
        ggml_tensor * vh = ggml_view_2d(ctx, v, head_dim, n_tokens_total, v->nb[1], h * head_dim * sizeof(float));
        
        // scores = Q_h^T @ K_h: Q^T [n_tokens, head_dim] @ K [head_dim, n_tokens] -> [n_tokens, n_tokens]
        ggml_tensor * qht = ggml_cont(ctx, ggml_permute(ctx, qh, 1, 0, 2, 3)); // [n_tokens, head_dim]
        ggml_tensor * scores = ggml_mul_mat(ctx, qht, kh); // Q^T @ K
        scores = ggml_scale(ctx, scores, scale);
        scores = ggml_clamp(ctx, scores, -30.0f, 30.0f);
        scores = ggml_soft_max(ctx, scores);
        
        // output_h = scores @ V_h^T? No: scores [n_tokens, n_tokens], V [head_dim, n_tokens]
        // Want: V @ scores^T? No, standard: attn @ V where attn = [n_tokens, n_tokens], V = [n_tokens, head_dim]
        // We have V_h [head_dim, n_tokens], transpose to [n_tokens, head_dim]
        ggml_tensor * vht = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3)); // [n_tokens, head_dim]
        // scores @ V = [n_tokens, n_tokens] @ [n_tokens, head_dim] -> [n_tokens, head_dim]
        ggml_tensor * out_h = ggml_mul_mat(ctx, scores, vht);
        // Transpose back: [n_tokens, head_dim] -> [head_dim, n_tokens]
        out_h = ggml_cont(ctx, ggml_permute(ctx, out_h, 1, 0, 2, 3));
        
        // Compute this head
        ggml_cgraph * cg = ggml_new_graph(ctx);
        ggml_build_forward_expand(cg, out_h);
        ggml_graph_compute_with_ctx(ctx, cg, 1);
        
        // Copy to output buffer: output[h * head_dim : (h+1) * head_dim, :]
        float * head_data = (float*)out_h->data;
        for (int i = 0; i < head_dim; i++)
            memcpy(out_data + (h * head_dim + i) * n_tokens_total,
                   head_data + i * n_tokens_total, n_tokens_total * sizeof(float));
    }
    
    // Output projection: [hidden, n_tokens] @ o_weight -> [hidden, n_tokens]
    attn_out = ggml_mul_mat(ctx, o_weight, attn_out);

    return attn_out;
}

// ===========================================================================
// DiT Block forward (simplified)
// ===========================================================================

static ggml_tensor * dit_block_forward_simple(
    ggml_context * ctx,
    ggml_tensor * x,           // [hidden, seq_len * n_batch]
    ggml_tensor * cond,        // [ada_dim, n_batch]
    dit_block & block,
    int seq_len,
    int n_batch
) {
    int hidden = DIT_HIDDEN_SIZE;
    int n_tokens = seq_len * n_batch;

    // adaLN: cond @ adaln_linear_w -> [6*hidden, n_batch]
    ggml_tensor * adaln = ggml_mul_mat(ctx, block.adaln_linear_w, cond); // [6*hidden, n_batch]

    // Split into 6 modulations, each [hidden, n_batch]
    int stride = hidden;
    ggml_tensor * shift_msa = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 0);
    ggml_tensor * scale_msa = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], stride * sizeof(float));
    ggml_tensor * gate_msa  = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 2 * stride * sizeof(float));
    ggml_tensor * shift_mlp = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 3 * stride * sizeof(float));
    ggml_tensor * scale_mlp = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 4 * stride * sizeof(float));
    ggml_tensor * gate_mlp  = ggml_view_2d(ctx, adaln, hidden, n_batch, adaln->nb[1], 5 * stride * sizeof(float));

    // Broadcast modulations from [hidden, n_batch] -> [hidden, n_tokens]
    // by repeating seq_len times along the token dimension
    auto broadcast_batch_to_tokens = [&](ggml_tensor * mod) -> ggml_tensor * {
        // mod [hidden, n_batch] -> [hidden, n_batch, seq_len] -> [hidden, n_tokens]
        ggml_tensor * m3 = ggml_reshape_3d(ctx, mod, hidden, n_batch, 1);
        m3 = ggml_cont(ctx, ggml_permute(ctx, m3, 0, 2, 1, 3)); // [hidden, 1, n_batch]
        // Now just reshape: ggml stores row-major, so adding a dimension works
        // Actually need to use ggml_repeat or manual copy
        // For MVP simplicity, create the expanded tensor manually
        ggml_tensor * out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        float * src = tensor_data(mod);
        float * dst = tensor_data(out);
        for (int s = 0; s < seq_len; s++) {
            for (int b = 0; b < n_batch; b++) {
                memcpy(dst + (b * seq_len + s) * hidden, src + b * hidden, hidden * sizeof(float));
            }
        }
        return out;
    };

    ggml_tensor * shift_msa_tok = broadcast_batch_to_tokens(shift_msa);
    ggml_tensor * scale_msa_tok = broadcast_batch_to_tokens(scale_msa);
    ggml_tensor * gate_msa_tok  = broadcast_batch_to_tokens(gate_msa);
    ggml_tensor * shift_mlp_tok = broadcast_batch_to_tokens(shift_mlp);
    ggml_tensor * scale_mlp_tok = broadcast_batch_to_tokens(scale_mlp);
    ggml_tensor * gate_mlp_tok  = broadcast_batch_to_tokens(gate_mlp);

    // --- Self-attention sub-layer ---
    // norm(x) * (1 + scale) + shift
    ggml_tensor * h = ggml_rms_norm(ctx, x, 1e-6f);
    if (block.attn_norm_w) {
        h = ggml_mul(ctx, h, block.attn_norm_w);
    }
    h = ggml_mul(ctx, h, scale_msa_tok); // scale * norm(x)... but we want (1+scale)*norm(x)
    // Add 1 to scale: create ones tensor and add
    ggml_tensor * ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    for (int i = 0; i < hidden * n_tokens; i++) ((float*)tensor_data(ones))[i] = 1.0f;
    ggml_tensor * scale_p1 = ggml_add(ctx, ones, scale_msa_tok);
    h = ggml_mul(ctx, scale_p1, ggml_rms_norm(ctx, x, 1e-6f));
    if (block.attn_norm_w) {
        h = ggml_mul(ctx, h, block.attn_norm_w);
    }
    h = ggml_add(ctx, h, shift_msa_tok);

    // Multi-head attention (manual per-head loop)
    ggml_tensor * attn = dit_attention_multihead(ctx, h,
        block.attn_q_weight, block.attn_k_weight, block.attn_v_weight,
        block.attn_o_weight,
        seq_len, n_batch, DIT_NUM_HEADS, DIT_HEAD_SIZE,
        block.q_norm_w, block.k_norm_w);

    x = ggml_add(ctx, x, ggml_mul(ctx, gate_msa_tok, attn));

    // --- FFN sub-layer ---
    h = ggml_rms_norm(ctx, x, 1e-6f);
    if (block.ffn_norm_w) {
        h = ggml_mul(ctx, h, block.ffn_norm_w);
    }
    scale_p1 = ggml_add(ctx, ones, scale_mlp_tok);
    h = ggml_rms_norm(ctx, x, 1e-6f);
    if (block.ffn_norm_w) {
        h = ggml_mul(ctx, h, block.ffn_norm_w);
    }
    h = ggml_mul(ctx, scale_p1, h);
    h = ggml_add(ctx, h, shift_mlp_tok);

    ggml_tensor * ffn = dit_ffn(ctx, h, block.ffn_w1, block.ffn_w2);
    x = ggml_add(ctx, x, ggml_mul(ctx, gate_mlp_tok, ffn));

    return x;
}

// ===========================================================================
// Full DiT forward
// ===========================================================================

ggml_tensor * dit_forward(
    dit_model & model,
    ggml_context * ctx,
    ggml_tensor * x,           // [seq_len, n_batch, hidden]
    ggml_tensor * t,           // [n_batch] timestep
    ggml_tensor * speaker_emb, dit_dump_ctx * dump) // [speaker_dim, n_batch] or nullptr
{
    int seq_len = x->ne[0];
    int n_batch = x->ne[1];
    int hidden  = DIT_HIDDEN_SIZE;
    int n_tokens = seq_len * n_batch;

    // Flatten to [hidden, n_tokens]
    x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [hidden, seq_len, n_batch]
    x = ggml_reshape_2d(ctx, x, hidden, n_tokens);

    // Timestep embedding: sinusoidal -> MLP (real dots.tts pipeline)
    // Step 1: sinusoidal features [256, n_batch]
    ggml_tensor * t_sin = dit_timestep_embedding(ctx, t, 256);
    
    // Step 2: time_embedder MLP: 256 -> 1024 -> 1024
    ggml_tensor * t_mlp = ggml_mul_mat(ctx, model.t_embed_w1, t_sin); // [1024, n_batch]
    t_mlp = ggml_silu(ctx, t_mlp);
    t_mlp = ggml_mul_mat(ctx, model.t_embed_w2, t_mlp); // [1024, n_batch]
    
    // Step 3: speaker embedding via xvec_proj (2-layer MLP)
    ggml_tensor * cond = t_mlp;
    if (speaker_emb && model.spk_proj_w1) {
        ggml_tensor * spk = ggml_mul_mat(ctx, model.spk_proj_w1, speaker_emb); // [1024, n_batch]
        spk = ggml_silu(ctx, spk);
        cond = ggml_add(ctx, cond, spk);
    }
    
    // Step 4: SiLU on combined conditioning
    cond = ggml_silu(ctx, cond);

    // Input layer (Linear before DiT blocks)
    if (model.input_layer_w) {
        x = ggml_mul_mat(ctx, model.input_layer_w, x);
        if (model.input_layer_b) x = ggml_add(ctx, x, model.input_layer_b);
    if (dump) dump->add("dit_input_layer", x);
    }

    static int dump_count = 0;
    for (int i = 0; i < model.n_layers; i++) {
        x = dit_block_forward_simple(ctx, x, cond, model.layers[i], seq_len, n_batch);
    if (dump && i == 0) dump->add("dit_block0", x);
        if (i == 0 && dump_count == 0) {
            dump_count++;
            ggml_cgraph * dgf = ggml_new_graph(ctx);
            ggml_build_forward_expand(dgf, x);
            ggml_graph_compute_with_ctx(ctx, dgf, 1);
            float * xd = (float*)x->data;
            int n = seq_len * n_batch * DIT_HIDDEN_SIZE;
            float r=0; for(int j=0;j<n;j++) r+=xd[j]*xd[j];
            FILE * df = fopen("debug/dit_block0.bin", "wb");
            if(df){fwrite(xd,sizeof(float),n,df);fclose(df);}
            printf("  DiT block0 out: rms=%.4f\\n", sqrtf(r/n));
        }
    }

    // Output: adaLN modulation + linear projection
    // adaLN: cond -> [shift, scale] each [hidden, n_batch], broadcast to [hidden, n_tokens]
    ggml_tensor * out = x;
    if (model.out_adaln_w) {
        // Compute modulation: cond [ada_dim, n_batch] @ out_adaln_w [ada_dim, 2*hidden] -> [2*hidden, n_batch]
        ggml_tensor * mod = ggml_mul_mat(ctx, model.out_adaln_w, cond); // [2*hidden, n_batch]
        
        // Split into shift [hidden, n_batch] and scale [hidden, n_batch]
        int stride = DIT_HIDDEN_SIZE;
        ggml_tensor * shift = ggml_view_2d(ctx, mod, DIT_HIDDEN_SIZE, n_batch, mod->nb[1], 0);
        ggml_tensor * scale = ggml_view_2d(ctx, mod, DIT_HIDDEN_SIZE, n_batch, mod->nb[1], stride * sizeof(float));
        
        // Broadcast to [hidden, n_tokens] by repeating seq_len times
        // Use manual broadcast: copy values
        int n_tokens = seq_len * n_batch;
        ggml_tensor * shift_tok = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, n_tokens);
        ggml_tensor * scale_tok = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, n_tokens);
        {
            float * ss = tensor_data(shift), * sd = tensor_data(shift_tok);
            float * cs = tensor_data(scale), * cd = tensor_data(scale_tok);
            for (int s = 0; s < seq_len; s++)
                for (int b = 0; b < n_batch; b++) {
                    memcpy(sd + (b * seq_len + s) * DIT_HIDDEN_SIZE,
                           ss + b * DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE * sizeof(float));
                    memcpy(cd + (b * seq_len + s) * DIT_HIDDEN_SIZE,
                           cs + b * DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE * sizeof(float));
                }
        }
        
        // Apply adaLN: norm(x) * (1 + scale) + shift
        out = ggml_rms_norm(ctx, out, 1e-6f);
        // Create ones tensor
        ggml_tensor * ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, n_tokens);
        for (int i = 0; i < DIT_HIDDEN_SIZE * n_tokens; i++) ((float*)tensor_data(ones))[i] = 1.0f;
        ggml_tensor * scale_1 = ggml_add(ctx, ones, scale_tok);
        out = ggml_mul(ctx, out, scale_1);
        out = ggml_add(ctx, out, shift_tok);
    }
    out = ggml_mul_mat(ctx, model.out_proj_w, out);
    if (dump) dump->add("dit_output", out);

    return out; // [latent_dim, n_tokens] = [VAE_LATENT_DIM, seq_len * n_batch]
}
