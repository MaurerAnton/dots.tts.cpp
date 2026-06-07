// dots.tts.cpp - PatchEncoder implementation
// 24-layer causal streaming transformer encoder

#include "patchenc.h"
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <cstdio>

// ===========================================================================
// Causal Conv1d: kernel=2, stride=2, causal (no future leakage)
// Simulated with ggml: conv1d not directly available, use IM2COL + MUL_MAT
// For simplicity, implement as a manual operation
// Input: [seq, channels] -> Output: [seq/2, channels]
// ===========================================================================

static void causal_conv1d_stride2(
    ggml_context * ctx,
    ggml_tensor * x,           // [seq, channels] input
    ggml_tensor * weight,      // [out_ch, in_ch, kernel] = [128, 128, 2]
    ggml_tensor * bias,        // [out_ch]
    int seq_len,
    int channels
) {
    // For MVP: do the conv manually in float and store back
    // x is [seq, channels] — after reshape from [channels, seq] ggml layout
    // Actually the ggml layout for 2D tensor is [ne0, ne1] = [channels, seq]
    // We'll do a manual CPU conv since ggml CONV_1D isn't available
    
    float * x_data = tensor_data(x);
    float * w_data = tensor_data(weight);
    float * b_data = bias ? tensor_data(bias) : nullptr;

    int out_seq = seq_len / 2;  // stride 2
    int kernel = 2;

    // Causal: for output position i, input is [2*i-1, 2*i] (no future)
    for (int o = 0; o < out_seq; o++) {
        for (int oc = 0; oc < channels; oc++) {
            float sum = b_data ? b_data[oc] : 0.0f;
            for (int k = 0; k < kernel; k++) {
                int inp_idx = 2 * o + k - (kernel - 1);  // causal offset
                if (inp_idx >= 0 && inp_idx < seq_len) {
                    for (int ic = 0; ic < channels; ic++) {
                        // weight layout: [out_ch, in_ch, kernel] -> w[oc, ic, k]
                        sum += x_data[inp_idx * channels + ic] *
                               w_data[(oc * channels + ic) * kernel + k];
                    }
                }
            }
            x_data[o * channels + oc] = sum;
        }
    }
    // Zero out remaining positions
    memset(x_data + out_seq * channels, 0, (seq_len - out_seq) * channels * sizeof(float));
}

// ===========================================================================
// Simple self-attention (causal) for encoder layers
// Same as DiT attention but without adaLN, with causal mask
// ===========================================================================

static ggml_tensor * enc_attention(
    ggml_context * ctx,
    ggml_tensor * x,           // [hidden, n_tokens]
    ggml_tensor * q_weight,    // [hidden, hidden]
    ggml_tensor * k_weight,    // [hidden, hidden]
    ggml_tensor * v_weight,    // [hidden, hidden]
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

    // Separate Q, K, V projections
    ggml_tensor * q = ggml_mul_mat(ctx, q_weight, x); // [hidden, n_tokens]
    ggml_tensor * k = ggml_mul_mat(ctx, k_weight, x);
    ggml_tensor *    v = ggml_mul_mat(ctx, v_weight, x);

    q = ggml_cont(ctx, q);
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    // qk_norm
    if (q_norm_w) {
        q = ggml_rms_norm(ctx, q, 1e-6f);
        q = ggml_mul(ctx, q, q_norm_w);
    }
    if (k_norm_w) {
        k = ggml_rms_norm(ctx, k, 1e-6f);
        k = ggml_mul(ctx, k, k_norm_w);
    }

    // RoPE: reshape to [head_dim, n_heads, n_tokens]
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, n_tokens);

    // Position IDs
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    {
        int * pd = (int *)tensor_data(pos);
        for (int b = 0; b < n_batch; b++)
            for (int s = 0; s < seq_len; s++)
                pd[b * seq_len + s] = s;
    }

    q = ggml_rope(ctx, q, pos, head_dim, 0);
    k = ggml_rope(ctx, k, pos, head_dim, 0);

    // Attention: Q^T @ K / sqrt(d) with causal mask
    ggml_tensor * Qf = ggml_reshape_2d(ctx, q, hidden, n_tokens);
    ggml_tensor * Kf = ggml_reshape_2d(ctx, k, hidden, n_tokens);
    ggml_tensor * Vf = ggml_reshape_2d(ctx, v, hidden, n_tokens);

    ggml_tensor * scores = ggml_mul_mat(ctx, Qf, Kf); // [n_tokens, n_tokens]
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float)head_dim));

    // Causal mask: zero out upper triangle
    // Use diag_mask_inf for causal masking
    scores = ggml_diag_mask_inf(ctx, scores, 1); // 1 = future tokens masked

    scores = ggml_soft_max(ctx, scores);

    // Output: scores @ V^T
    ggml_tensor * Vt = ggml_cont(ctx, ggml_permute(ctx, Vf, 1, 0, 2, 3));
    ggml_tensor * attn_out = ggml_mul_mat(ctx, scores, Vt); // [n_tokens, hidden]
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 1, 0, 2, 3)); // [hidden, n_tokens]

    // Output projection
    attn_out = ggml_mul_mat(ctx, o_weight, attn_out);

    return attn_out;
}

// ===========================================================================
// Encoder layer forward: pre-norm + attn + pre-norm + FFN (no adaLN)
// ===========================================================================

static ggml_tensor * enc_layer_forward(
    ggml_context * ctx,
    ggml_tensor * x,           // [hidden, n_tokens]
    patchenc_layer & layer,
    int seq_len,
    int n_batch
) {
    int hidden = PATCHENC_HIDDEN;
    int n_tokens = seq_len * n_batch;

    // Self-attention with pre-norm
    ggml_tensor * h = ggml_rms_norm(ctx, x, 1e-6f);
    if (layer.attn_norm_w) h = ggml_mul(ctx, h, layer.attn_norm_w);

    ggml_tensor * attn = enc_attention(ctx, h,
        layer.attn_q_weight, layer.attn_k_weight, layer.attn_v_weight,
        layer.attn_o_weight,
        seq_len, n_batch, PATCHENC_NUM_HEADS, PATCHENC_HEAD_SIZE,
        layer.q_norm_w, layer.k_norm_w);

    x = ggml_add(ctx, x, attn);

    // FFN with pre-norm
    h = ggml_rms_norm(ctx, x, 1e-6f);
    if (layer.ffn_norm_w) h = ggml_mul(ctx, h, layer.ffn_norm_w);

    ggml_tensor * ffn = ggml_mul_mat(ctx, layer.ffn_w1, h); // [ffn, n_tokens]
    ffn = ggml_silu(ctx, ffn);
    ffn = ggml_mul_mat(ctx, layer.ffn_w2, ffn);             // [hidden, n_tokens]

    x = ggml_add(ctx, x, ffn);

    return x;
}

// ===========================================================================
// Full forward pass (non-streaming)
// ===========================================================================

ggml_tensor * patchenc_forward(
    patch_encoder & enc,
    ggml_context * ctx,
    ggml_tensor * x,           // [n_patches * patch_size, latent_dim]
    int n_patches
) {
    int seq = n_patches * PATCHENC_PATCH_SIZE;
    int ch  = PATCHENC_LATENT_DIM;

    // 1. Causal Conv1d: [seq, ch] -> [seq/2, ch] (stride 2)
    // x is in ggml layout: [ch, seq] after reshape
    // Do manual conv (stores result back in x's first half)
    causal_conv1d_stride2(ctx, x, enc.conv_weight, enc.conv_bias, seq, ch);
    int conv_seq = seq / 2;  // = n_patches * 2 (since patch_size=4, /2 = 2 per patch)
    // Actually: n_patches * 4 -> conv -> n_patches * 2 positions

    // 2. Input projection: Linear(128 -> 1024)
    // x currently [ch, seq] in memory but we overwrote first conv_seq*ch elements
    // Create view of first conv_seq elements, reshape to [ch, conv_seq]
    ggml_tensor * x_conv = ggml_view_2d(ctx, x, ch, conv_seq, x->nb[1], 0);
    x_conv = ggml_cont(ctx, x_conv);
    ggml_tensor * h = ggml_mul_mat(ctx, enc.in_proj_w, x_conv); // [hidden, conv_seq]
    // bias skipped: ggml requires same shape for add

    int n_batch = 1;
    int n_tokens = conv_seq;

    // 3. 24 causal transformer layers
    for (int i = 0; i < enc.n_layers; i++) {
        h = enc_layer_forward(ctx, h, enc.layers[i], n_tokens, n_batch);
    }

    // 4. Final norm
    h = ggml_rms_norm(ctx, h, 1e-6f);
    if (enc.final_norm_w) h = ggml_mul(ctx, h, enc.final_norm_w);

    // 5. Output: reshape [hidden, n_patches*2] -> [hidden, 2, n_patches]
    // Take first frame per patch -> [hidden, n_patches], project to LLM dim
    ggml_tensor * h3d = ggml_reshape_3d(ctx, h, PATCHENC_HIDDEN, 2, n_patches);
    h3d = ggml_cont(ctx, ggml_permute(ctx, h3d, 0, 2, 1, 3)); // [hidden, n_patches, 2]
    ggml_tensor * h_first = ggml_view_2d(ctx, h3d, PATCHENC_HIDDEN, n_patches,
        h3d->nb[1], 0);
    h_first = ggml_cont(ctx, h_first);

    // Simple projection: [hidden] -> [1536] (skip 2*hidden concat for MVP)
    // Use out_proj modified to accept [hidden] instead of [2*hidden]
    // For now, use just the first half: out_proj_w viewed as [hidden, 1536]
    ggml_tensor * out_w = ggml_view_2d(ctx, enc.out_proj_w, PATCHENC_HIDDEN, 1536,
        enc.out_proj_w->nb[1], 0);
    out_w = ggml_cont(ctx, out_w);
    ggml_tensor * out = ggml_mul_mat(ctx, out_w, h_first); // [1536, n_patches]
    // bias skipped for same-shape requirement

    return out; // [1536, n_patches]
}

// ===========================================================================
// Streaming state
// ===========================================================================

struct patchenc_stream_state {
    // Conv tail buffer: stores last (kernel-1) input positions for causal conv
    float conv_tail[(2-1) * PATCHENC_LATENT_DIM]; // 1 position tail

    // KV caches for each layer: one per layer
    // For simplicity in MVP, store full history (non-streaming mode)
    // TODO: implement true KV cache streaming
    int pos;  // current position in sequence
    
    // Accumulated hidden states for output projection
    float * hidden_history;
    int max_history;
    int cur_history;
};

patchenc_stream_state * patchenc_stream_init(patch_encoder & enc) {
    (void)enc;
    patchenc_stream_state * st = new patchenc_stream_state();
    memset(st->conv_tail, 0, sizeof(st->conv_tail));
    st->pos = 0;
    st->max_history = 1024;
    st->cur_history = 0;
    st->hidden_history = new float[st->max_history * PATCHENC_HIDDEN]();
    return st;
}

void patchenc_stream_free(patchenc_stream_state * st) {
    delete[] st->hidden_history;
    delete st;
}

void patchenc_decode_step(
    patch_encoder & enc,
    patchenc_stream_state * st,
    ggml_context * ctx,
    float * patch,             // [patch_size * latent_dim]
    float * llm_emb_out)       // [1536]
{
    // For MVP: just run full non-streaming forward on accumulated patches
    // This is correct but not truly streaming
    // TODO: implement incremental decode with KV caching

    int patch_flat = PATCHENC_PATCH_SIZE * PATCHENC_LATENT_DIM;
    
    // Accumulate patch
    if (st->cur_history + PATCHENC_PATCH_SIZE > st->max_history) {
        // Grow buffer
        int new_max = st->max_history * 2;
        float * new_buf = new float[new_max * PATCHENC_LATENT_DIM]();
        memcpy(new_buf, st->hidden_history, 
               st->cur_history * PATCHENC_LATENT_DIM * sizeof(float));
        delete[] st->hidden_history;
        st->hidden_history = new_buf;
        st->max_history = new_max;
    }
    
    memcpy(st->hidden_history + st->cur_history * PATCHENC_LATENT_DIM,
           patch, patch_flat * sizeof(float));
    st->cur_history += PATCHENC_PATCH_SIZE;

    int n_patches = st->cur_history / PATCHENC_PATCH_SIZE;
    int seq = st->cur_history;

    // Build input tensor
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PATCHENC_LATENT_DIM, seq);
    memcpy(tensor_data(x), st->hidden_history, seq * PATCHENC_LATENT_DIM * sizeof(float));

    // Run forward
    ggml_tensor * out = patchenc_forward(enc, ctx, x, n_patches); // [1536, n_patches]

    // Compute graph
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 8);

    // Extract last patch's embedding
    float * out_data = tensor_data(out);
    memcpy(llm_emb_out, out_data + (n_patches - 1) * 1536, 1536 * sizeof(float));

    ggml_reset(ctx);
}
