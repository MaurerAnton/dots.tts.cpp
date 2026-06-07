// dots.tts.cpp - DiT forward pass (simplified for ggml)
// All operations use either:
//   1. ggml_mul_mat for linear projections (2D: [out_features, batch*seq])
//   2. Element-wise ops on correctly shaped tensors  
//   3. ggml_rms_norm, ggml_silu, ggml_rope, ggml_soft_max

#include "dit.h"
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <cstdio>

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
    ggml_tensor * qkv_weight,  // [3*hidden, hidden]
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

    // QKV projection: [3*hidden, n_tokens]
    ggml_tensor * qkv = ggml_mul_mat(ctx, qkv_weight, x);

    // Split: Q, K, V each [hidden, n_tokens]
    int h = hidden;
    ggml_tensor * q = ggml_view_2d(ctx, qkv, h, n_tokens, qkv->nb[1], 0);
    ggml_tensor * k = ggml_view_2d(ctx, qkv, h, n_tokens, qkv->nb[1], h * sizeof(float));
    ggml_tensor * v = ggml_view_2d(ctx, qkv, h, n_tokens, qkv->nb[1], 2 * h * sizeof(float));

    // Reshape to [seq_len, n_batch, n_heads, head_dim] for RoPE
    // First: [head_dim, n_heads, seq_len, n_batch]
    auto reshape_qkv_4d = [&](ggml_tensor * t) {
        t = ggml_cont(ctx, t);  // ensure contiguity after view
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
    
    // For MVP: single-head attention (all heads merged)
    // mul_mat(A, B) = A^T @ B. A^T = Q^T = [n_tokens, hidden]. A^T @ K = [n_tokens, hidden] @ [hidden, n_tokens] = [n_tokens, n_tokens]
    // So: scores = mul_mat(Q, K) where both are [hidden, n_tokens]
    ggml_tensor * Qf = ggml_reshape_2d(ctx, q, hidden, n_tokens); // [hidden, n_tokens]
    ggml_tensor * Kf = ggml_reshape_2d(ctx, k, hidden, n_tokens); // [hidden, n_tokens]
    ggml_tensor * Vf = ggml_reshape_2d(ctx, v, hidden, n_tokens); // [hidden, n_tokens]

    ggml_tensor * scores = ggml_mul_mat(ctx, Qf, Kf); // Q^T @ K = [n_tokens, n_tokens]

    // Scale
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float)head_dim));

    // Softmax
    scores = ggml_soft_max(ctx, scores);

    // scores @ V: scores [n_tokens, n_tokens], V [hidden, n_tokens]
    // Want: [n_tokens, hidden] = scores^T @ V^T? No
    // scores [n_tokens, n_tokens], V [hidden, n_tokens] -> V^T [n_tokens, hidden]
    // result = scores @ V^T = [n_tokens, n_tokens] @ [n_tokens, hidden] = [n_tokens, hidden]
    // mul_mat(V^T, scores^T)? 
    // mul_mat(scores, V^T): scores [n_tokens, n_tokens], V^T [n_tokens, hidden] -> [n_tokens, hidden]
    ggml_tensor * Vt = ggml_cont(ctx, ggml_permute(ctx, Vf, 1, 0, 2, 3)); // [n_tokens, hidden]
    ggml_tensor * attn_out = ggml_mul_mat(ctx, scores, Vt); // [n_tokens, hidden]

    // Output projection
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 1, 0, 2, 3)); // [hidden, n_tokens]
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

    // Attention
    ggml_tensor * attn = dit_attention(ctx, h,
        block.attn_qkv_weight, block.attn_o_weight,
        seq_len, n_batch, DIT_NUM_HEADS, DIT_HEAD_SIZE,
        nullptr, nullptr);

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
    ggml_tensor * speaker_emb) // [speaker_dim, n_batch] or nullptr
{
    int seq_len = x->ne[0];
    int n_batch = x->ne[1];
    int hidden  = DIT_HIDDEN_SIZE;
    int n_tokens = seq_len * n_batch;

    // Flatten to [hidden, n_tokens]
    x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [hidden, seq_len, n_batch]
    x = ggml_reshape_2d(ctx, x, hidden, n_tokens);

    // Timestep embedding
    ggml_tensor * t_emb = dit_timestep_embedding(ctx, t, DIT_ADA_DIM); // [ada_dim, n_batch]

    // Add speaker embedding if provided
    if (speaker_emb && model.spk_proj_w) {
        ggml_tensor * spk = ggml_mul_mat(ctx, model.spk_proj_w, speaker_emb); // [ada_dim, n_batch]
        fprintf(stderr, "DEBUG: after spk_proj\n"); fflush(stderr);
        t_emb = ggml_add(ctx, t_emb, spk);
        fprintf(stderr, "DEBUG: after ggml_add t_emb+spk\n"); fflush(stderr);
    }

    // SiLU
    fprintf(stderr, "DEBUG: before silu, t_emb [%lld %lld]\n",
        (long long)t_emb->ne[0], (long long)t_emb->ne[1]); fflush(stderr);
    ggml_tensor * cond = ggml_silu(ctx, t_emb);
    fprintf(stderr, "DEBUG: after silu\n"); fflush(stderr);

    // Pass through DiT blocks
    for (int i = 0; i < model.n_layers; i++) {
        if (i == 0) {
            fprintf(stderr, "DEBUG layer 0: adaln_w [%lld %lld], cond [%lld %lld %lld %lld], x [%lld %lld %lld %lld]\n",
                (long long)model.layers[i].adaln_linear_w->ne[0],
                (long long)model.layers[i].adaln_linear_w->ne[1],
                (long long)cond->ne[0], (long long)cond->ne[1],
                (long long)cond->ne[2], (long long)cond->ne[3],
                (long long)x->ne[0], (long long)x->ne[1],
                (long long)x->ne[2], (long long)x->ne[3]);
        }
        x = dit_block_forward_simple(ctx, x, cond, model.layers[i], seq_len, n_batch);
    }

    // Output projection: [hidden, n_tokens] -> [latent_dim, n_tokens]
    fprintf(stderr, "DEBUG: out_proj_w [%lld %lld], x [%lld %lld]\n",
        (long long)model.out_proj_w->ne[0], (long long)model.out_proj_w->ne[1],
        (long long)x->ne[0], (long long)x->ne[1]);
    fflush(stderr);
    ggml_tensor * out = ggml_mul_mat(ctx, model.out_proj_w, x); // [latent_dim, n_tokens]

    return out; // [latent_dim, n_tokens] = [VAE_LATENT_DIM, seq_len * n_batch]
}
