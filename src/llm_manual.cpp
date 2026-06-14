// Manual LLM forward pass — byte-perfect with PyTorch Qwen2/Mistral
// EXACT code from proven test_manual_llm.cpp (commit d35a084)
#include "llm_manual.h"
#include "dit_manual.h"
#include "safetensors.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

struct LayerW { float *an,*fn,*qw,*qb,*kw,*kb,*vw,*vb,*ow,*ob,*gw,*uw,*dw; };

bool load_llm_weights(SafeTensorsFile & sf, LLMWeights & w, int n_layers) {
    w.n_layers = n_layers;
    w.layers = new LLMWeights::Layer[n_layers];
    memset(w.layers, 0, n_layers * sizeof(LLMWeights::Layer));
    
    auto load = [&](const char * name, float *& buf) -> bool {
        const st_tensor_info * info = sf.find(name);
        if (!info) { buf = nullptr; return true; }
        size_t n = 1; for (size_t d : info->shape) n *= d;
        buf = (float*)malloc(n * sizeof(float));
        if (!buf) return false;
        sf.load_raw(*info, buf, n);
        return true;
    };
    
    if (!load("llm.model.embed_tokens.weight", w.embed_w)) return false;
    
    for (int i = 0; i < n_layers; i++) {
        char name[512]; auto & l = w.layers[i];
        snprintf(name,sizeof(name),"llm.model.layers.%d.input_layernorm.weight",i); load(name,l.an);
        snprintf(name,sizeof(name),"llm.model.layers.%d.post_attention_layernorm.weight",i); load(name,l.fn);
        snprintf(name,sizeof(name),"llm.model.layers.%d.self_attn.q_proj.weight",i); load(name,l.qw);
        snprintf(name,sizeof(name),"llm.model.layers.%d.self_attn.q_proj.bias",i); load(name,l.qb);
        snprintf(name,sizeof(name),"llm.model.layers.%d.self_attn.k_proj.weight",i); load(name,l.kw);
        snprintf(name,sizeof(name),"llm.model.layers.%d.self_attn.k_proj.bias",i); load(name,l.kb);
        snprintf(name,sizeof(name),"llm.model.layers.%d.self_attn.v_proj.weight",i); load(name,l.vw);
        snprintf(name,sizeof(name),"llm.model.layers.%d.self_attn.v_proj.bias",i); load(name,l.vb);
        snprintf(name,sizeof(name),"llm.model.layers.%d.self_attn.o_proj.weight",i); load(name,l.ow);
        snprintf(name,sizeof(name),"llm.model.layers.%d.self_attn.o_proj.bias",i); load(name,l.ob);
        snprintf(name,sizeof(name),"llm.model.layers.%d.mlp.gate_proj.weight",i); load(name,l.gw);
        snprintf(name,sizeof(name),"llm.model.layers.%d.mlp.up_proj.weight",i); load(name,l.uw);
        snprintf(name,sizeof(name),"llm.model.layers.%d.mlp.down_proj.weight",i); load(name,l.dw);
    }
    return load("llm.model.norm.weight", w.final_norm);
}

void free_llm_weights(LLMWeights & w) {
    free(w.embed_w); free(w.final_norm);
    for (int i = 0; i < w.n_layers; i++) {
        auto & l = w.layers[i];
        free(l.an);free(l.fn);free(l.qw);free(l.qb);free(l.kw);free(l.kb);
        free(l.vw);free(l.vb);free(l.ow);free(l.ob);free(l.gw);free(l.uw);free(l.dw);
    }
    delete[] w.layers;
}

// EXACT forward pass from working commit d35a084, adapted to LLMWeights struct
void llm_manual_forward(const LLMWeights & w, const int * token_ids, int seq_len,
    float * hiddens_out) {
    const int H=1536, F=8960, NH=12, NKV=2, HD=128, N=w.n_layers;
    int kvdim = NKV * HD;  // 256 for 2 KV heads
    
    // Embeddings
    float * h = (float*)malloc(seq_len * H * sizeof(float));
    float * h2 = (float*)malloc(seq_len * H * sizeof(float));
    for (int t = 0; t < seq_len; t++)
        memcpy(h + t*H, w.embed_w + token_ids[t]*H, H*sizeof(float));
    
    // Buffers (reused across layers)
    float * normed = (float*)malloc(seq_len * H * sizeof(float));
    float * q = (float*)malloc(seq_len * H * sizeof(float));
    float * k = (float*)malloc(seq_len * kvdim * sizeof(float));
    float * v = (float*)malloc(seq_len * kvdim * sizeof(float));
    float * ao = (float*)malloc(seq_len * H * sizeof(float));
    float * attn_out = (float*)malloc(seq_len * H * sizeof(float));
    float * gate = (float*)malloc(seq_len * F * sizeof(float));
    float * up = (float*)malloc(seq_len * F * sizeof(float));
    float * mlp_out = (float*)malloc(seq_len * H * sizeof(float));
    float * scores = (float*)malloc(seq_len * seq_len * sizeof(float));
    float * qh = (float*)malloc(seq_len * HD * sizeof(float));
    float * kh = (float*)malloc(seq_len * HD * sizeof(float));
    float * vh = (float*)malloc(seq_len * HD * sizeof(float));
    
    float scale = 1.0f / sqrtf((float)HD);
    float * final_norm = w.final_norm;
    
    // Copy weight pointers to local struct for EXACT same access pattern as working code
    std::vector<LayerW> layers(N);
    for (int i = 0; i < N; i++) {
        layers[i].an = w.layers[i].an; layers[i].fn = w.layers[i].fn;
        layers[i].qw = w.layers[i].qw; layers[i].qb = w.layers[i].qb;
        layers[i].kw = w.layers[i].kw; layers[i].kb = w.layers[i].kb;
        layers[i].vw = w.layers[i].vw; layers[i].vb = w.layers[i].vb;
        layers[i].ow = w.layers[i].ow; layers[i].ob = w.layers[i].ob;
        layers[i].gw = w.layers[i].gw; layers[i].uw = w.layers[i].uw;
        layers[i].dw = w.layers[i].dw;
    }
    
    for (int i = 0; i < N; i++) {
        LayerW & l = layers[i];
        
        // RMSNorm
        for (int t = 0; t < seq_len; t++)
            manual_rms_norm(normed + t*H, h + t*H, l.an, H);
        
        // QKV projections (K/V: 256 output, Q: 1536 output)
        for (int t = 0; t < seq_len; t++) {
            manual_linear(q + t*H, normed + t*H, l.qw, l.qb, H, H);
            manual_linear(k + t*kvdim, normed + t*H, l.kw, l.kb, H, kvdim);
            manual_linear(v + t*kvdim, normed + t*H, l.vw, l.vb, H, kvdim);
        }
        
        // Multi-head attention
        memset(ao, 0, seq_len * H * sizeof(float));
        for (int head = 0; head < NH; head++) {
            int kv_h = head * NKV / NH;
            for (int t = 0; t < seq_len; t++) for (int d = 0; d < HD; d++) {
                qh[t*HD+d] = q[t*H + head*HD + d];
                kh[t*HD+d] = k[t*kvdim + kv_h*HD + d];
                vh[t*HD+d] = v[t*kvdim + kv_h*HD + d];
            }
            manual_rope_theta(qh, kh, seq_len, HD, 1000000.0f);
            
            // Scores + causal mask + softmax
            for (int ti = 0; ti < seq_len; ti++) {
                for (int tj = 0; tj < seq_len; tj++) {
                    float s = 0; for (int d = 0; d < HD; d++) s += qh[ti*HD+d] * kh[tj*HD+d];
                    scores[ti*seq_len+tj] = s * scale;
                }
                for (int tj = ti+1; tj < seq_len; tj++) scores[ti*seq_len+tj] = -INFINITY;
                manual_softmax(scores + ti*seq_len, seq_len);
            }
            // Weighted sum of V
            for (int ti = 0; ti < seq_len; ti++)
                for (int d = 0; d < HD; d++) {
                    float s = 0; for (int tj = 0; tj < seq_len; tj++) s += scores[ti*seq_len+tj] * vh[tj*HD+d];
                    ao[ti*H + head*HD + d] = s;
                }
        }
        
        // O projection
        for (int t = 0; t < seq_len; t++)
            manual_linear(attn_out + t*H, ao + t*H, l.ow, l.ob, H, H);
        
        // Residual
        for (int j = 0; j < seq_len*H; j++) h2[j] = h[j] + attn_out[j];
        
        // RMSNorm before MLP
        for (int t = 0; t < seq_len; t++)
            manual_rms_norm(normed + t*H, h2 + t*H, l.fn, H);
        
        // MLP: gate_proj → SiLU, up_proj → multiply, down_proj
        for (int t = 0; t < seq_len; t++) {
            manual_linear(gate + t*F, normed + t*H, l.gw, nullptr, H, F);
            manual_linear(up + t*F, normed + t*H, l.uw, nullptr, H, F);
            for (int j = 0; j < F; j++)
                gate[t*F+j] = gate[t*F+j] / (1.0f + expf(-gate[t*F+j])) * up[t*F+j];
        }
        for (int t = 0; t < seq_len; t++)
            manual_linear(mlp_out + t*H, gate + t*F, l.dw, nullptr, F, H);
        
        // Final residual + swap buffers
        for (int j = 0; j < seq_len*H; j++) h[j] = h2[j] + mlp_out[j];
    }
    
    // Final norm
    for (int t = 0; t < seq_len; t++)
        manual_rms_norm(h2 + t*H, h + t*H, final_norm, H);
    memcpy(hiddens_out, h2, seq_len * H * sizeof(float));
    
    free(h); free(h2); free(normed); free(q); free(k); free(v); free(ao); free(attn_out);
    free(gate); free(up); free(mlp_out); free(scores); free(qh); free(kh); free(vh);
}

// === KV Cache implementation ===

void llm_kv_cache_free(LLMKVCache & c) {
    if (c.k_cache) { for (int i = 0; i < c.n_layers; i++) free(c.k_cache[i]); free(c.k_cache); }
    if (c.v_cache) { for (int i = 0; i < c.n_layers; i++) free(c.v_cache[i]); free(c.v_cache); }
    c.k_cache = c.v_cache = nullptr; c.seq_len = 0;
}

static void manual_attention_kv(const float * q_single, const float * k_cached, const float * v_cached,
    int cached_len, const float * qw, const float * qb, const float * kw, const float * kb,
    const float * vw, const float * vb, const float * ow, const float * ob,
    int H, int HD, int head, int n_kv_heads, int NH, int kvdim, float * ao_single) {
    // Q for new token
    float qh[128];
    for (int d = 0; d < HD; d++) qh[d] = q_single[head*HD + d];
    
    // K, V from cache (for this head's kv_head)
    int kv_h = head * n_kv_heads / NH;
    const float * kh_cached = k_cached + kv_h * cached_len * HD;
    const float * vh_cached = v_cached + kv_h * cached_len * HD;
    
    // Also compute K, V for the new single token
    float kh_new[128], vh_new[128];
    for (int d = 0; d < HD; d++) {
        kh_new[d] = q_single[H + kv_h*HD + d];  // K is stored after Q in temp buffer
        vh_new[d] = q_single[H + H + kv_h*HD + d];  // V is stored after K
    }
    // Apply RoPE to Q and new K
    // (simplified: use manual_rope_theta on single-element arrays)
    float qh_r[128], kh_r[128];
    memcpy(qh_r, qh, HD*sizeof(float)); memcpy(kh_r, kh_new, HD*sizeof(float));
    manual_rope_theta(qh_r, kh_r, 1, HD, 1000000.0f);
    
    int total_len = cached_len + 1;
    float scores[256];  // max seq
    float sc = 1.0f / sqrtf((float)HD);
    
    // Scores against cached K
    for (int j = 0; j < cached_len; j++) {
        float s = 0; for (int d = 0; d < HD; d++) s += qh_r[d] * kh_cached[j*HD + d];
        scores[j] = s * sc;
    }
    // Score against new K
    { float s = 0; for (int d = 0; d < HD; d++) s += qh_r[d] * kh_r[d];
      scores[cached_len] = s * sc; }
    
    // Causal: can attend to all cached + new token
    manual_softmax(scores, total_len);
    
    // Weighted sum
    float out[128] = {0};
    for (int j = 0; j < cached_len; j++)
        for (int d = 0; d < HD; d++) out[d] += scores[j] * vh_cached[j*HD + d];
    for (int d = 0; d < HD; d++) out[d] += scores[cached_len] * vh_new[d];
    
    // Store in ao_single at head position
    for (int d = 0; d < HD; d++) ao_single[head*HD + d] = out[d];
}

void llm_kv_cache_init(const LLMWeights & w, const int * token_ids, int n_tok,
    float * hiddens_out, LLMKVCache & cache) {
    const int H=1536, NKV=2, HD=128, KVD=NKV*HD, N=w.n_layers;
    
    // Allocate cache
    cache.n_layers = N; cache.n_kv_heads = NKV; cache.head_dim = HD;
    cache.max_seq = n_tok + 256;  // enough for text tokens + audio calls
    cache.seq_len = n_tok;
    cache.k_cache = (float**)malloc(N * sizeof(float*));
    cache.v_cache = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        cache.k_cache[i] = (float*)calloc(NKV * cache.max_seq * HD, sizeof(float));
        cache.v_cache[i] = (float*)calloc(NKV * cache.max_seq * HD, sizeof(float));
    }
    
    // Use the proven llm_manual_forward to get hidden states
    llm_manual_forward(w, token_ids, n_tok, hiddens_out);
    
    // Now re-run the forward pass just to capture K/V at each layer.
    // This is inefficient but correct.
    float * h = (float*)malloc(n_tok * H * sizeof(float));
    float * h2 = (float*)malloc(n_tok * H * sizeof(float));
    for (int t = 0; t < n_tok; t++)
        memcpy(h + t*H, w.embed_w + token_ids[t]*H, H*sizeof(float));
    
    float * normed = (float*)malloc(n_tok * H * sizeof(float));
    float * q = (float*)malloc(n_tok * H * sizeof(float));
    float * k = (float*)malloc(n_tok * KVD * sizeof(float));
    float * v = (float*)malloc(n_tok * KVD * sizeof(float));
    float * ao = (float*)malloc(n_tok * H * sizeof(float));
    float * attn_out = (float*)malloc(n_tok * H * sizeof(float));
    float * gate = (float*)malloc(n_tok * 8960 * sizeof(float));
    float * up = (float*)malloc(n_tok * 8960 * sizeof(float));
    float * mlp_out = (float*)malloc(n_tok * H * sizeof(float));
    float * scores = (float*)malloc(n_tok * n_tok * sizeof(float));
    float * qh = (float*)malloc(n_tok * HD * sizeof(float));
    float * kh_buf = (float*)malloc(n_tok * HD * sizeof(float));
    float * vh = (float*)malloc(n_tok * HD * sizeof(float));
    float sc = 1.0f / sqrtf((float)HD);
    
    struct LayerWLocal { float *an,*fn,*qw,*qb,*kw,*kb,*vw,*vb,*ow,*ob,*gw,*uw,*dw; };
    std::vector<LayerWLocal> layers(N);
    for (int i = 0; i < N; i++) {
        layers[i].an=w.layers[i].an; layers[i].fn=w.layers[i].fn;
        layers[i].qw=w.layers[i].qw; layers[i].qb=w.layers[i].qb;
        layers[i].kw=w.layers[i].kw; layers[i].kb=w.layers[i].kb;
        layers[i].vw=w.layers[i].vw; layers[i].vb=w.layers[i].vb;
        layers[i].ow=w.layers[i].ow; layers[i].ob=w.layers[i].ob;
        layers[i].gw=w.layers[i].gw; layers[i].uw=w.layers[i].uw;
        layers[i].dw=w.layers[i].dw;
    }
    
    for (int i = 0; i < N; i++) {
        auto & l = layers[i];
        for (int t = 0; t < n_tok; t++) manual_rms_norm(normed + t*H, h + t*H, l.an, H);
        for (int t = 0; t < n_tok; t++) {
            manual_linear(q + t*H, normed + t*H, l.qw, l.qb, H, H);
            manual_linear(k + t*KVD, normed + t*H, l.kw, l.kb, H, KVD);
            manual_linear(v + t*KVD, normed + t*H, l.vw, l.vb, H, KVD);
        }
        // Save K with RoPE applied (as used in attention)
        for (int hh = 0; hh < NKV; hh++) {
            for (int t = 0; t < n_tok; t++) {
                float * kdst = cache.k_cache[i] + hh * cache.max_seq * HD + t * HD;
                float * ksrc = k + t*KVD + hh*HD;
                memcpy(kdst, ksrc, HD * sizeof(float));
                // Apply RoPE at position t
                int half=HD/2; float theta=1.0f; float ts=powf(1000000.0f,-2.0f/HD);
                for(int d=0;d<half;d++){float cs=cosf((float)t*theta),sn=sinf((float)t*theta);
                    int i1=d,i2=d+half;float k0=kdst[i1],k1=kdst[i2];
                    kdst[i1]=k0*cs-k1*sn;kdst[i2]=k1*cs+k0*sn;theta*=ts;}
            }
        }
        // Save V (no RoPE)
        for (int hh = 0; hh < NKV; hh++)
            for (int t = 0; t < n_tok; t++)
                memcpy(cache.v_cache[i] + hh * cache.max_seq * HD + t * HD,
                       v + t*KVD + hh*HD, HD * sizeof(float));
        // Run attention + MLP (to advance h to next layer)
        memset(ao, 0, n_tok * H * sizeof(float));
        for (int head = 0; head < 12; head++) {
            int kv_h = head * NKV / 12;
            for (int t = 0; t < n_tok; t++) for (int d = 0; d < HD; d++) {
                qh[t*HD+d]=q[t*H+head*HD+d]; kh_buf[t*HD+d]=k[t*KVD+kv_h*HD+d]; vh[t*HD+d]=v[t*KVD+kv_h*HD+d]; }
            manual_rope_theta(qh, kh_buf, n_tok, HD, 1000000.0f);
            for (int ti = 0; ti < n_tok; ti++) {
                for (int tj = 0; tj < n_tok; tj++) { float s=0; for(int d=0;d<HD;d++)s+=qh[ti*HD+d]*kh_buf[tj*HD+d]; scores[ti*n_tok+tj]=s*sc; }
                for (int tj = ti+1; tj < n_tok; tj++) scores[ti*n_tok+tj] = -INFINITY;
                manual_softmax(scores + ti*n_tok, n_tok); }
            for (int ti = 0; ti < n_tok; ti++) for (int d = 0; d < HD; d++) {
                float s=0; for(int tj=0;tj<n_tok;tj++)s+=scores[ti*n_tok+tj]*vh[tj*HD+d]; ao[ti*H+head*HD+d]=s; }}
        for (int t = 0; t < n_tok; t++) manual_linear(attn_out + t*H, ao + t*H, l.ow, l.ob, H, H);
        for (int j = 0; j < n_tok*H; j++) h2[j] = h[j] + attn_out[j];
        for (int t = 0; t < n_tok; t++) manual_rms_norm(normed + t*H, h2 + t*H, l.fn, H);
        for (int t = 0; t < n_tok; t++) {
            manual_linear(gate + t*8960, normed + t*H, l.gw, nullptr, H, 8960);
            manual_linear(up + t*8960, normed + t*H, l.uw, nullptr, H, 8960);
            for (int j = 0; j < 8960; j++) gate[t*8960+j] = gate[t*8960+j]/(1.0f+expf(-gate[t*8960+j])) * up[t*8960+j]; }
        for (int t = 0; t < n_tok; t++) manual_linear(mlp_out + t*H, gate + t*8960, l.dw, nullptr, 8960, H);
        for (int j = 0; j < n_tok*H; j++) h[j] = h2[j] + mlp_out[j];
    }
    free(h); free(h2); free(normed); free(q); free(k); free(v); free(ao); free(attn_out);
    free(gate); free(up); free(mlp_out); free(scores); free(qh); free(kh_buf); free(vh);
}

void llm_kv_cache_step(const LLMWeights & w, const float * input_emb,
    LLMKVCache & cache, float * hidden_out) {
    const int H=1536, F=8960, NH=12, NKV=2, HD=128, KVD=NKV*HD, N=w.n_layers;
    int seq_len = cache.seq_len;
    
    float * h = (float*)malloc(H * sizeof(float));
    memcpy(h, input_emb, H * sizeof(float));
    
    float * normed = (float*)malloc(H * sizeof(float));
    float * q = (float*)malloc(H * sizeof(float));
    float * k = (float*)malloc(KVD * sizeof(float));
    float * v = (float*)malloc(KVD * sizeof(float));
    float * ao = (float*)malloc(H * sizeof(float));
    float * attn_out = (float*)malloc(H * sizeof(float));
    float * gate = (float*)malloc(F * sizeof(float));
    float * up = (float*)malloc(F * sizeof(float));
    float * mlp_out = (float*)malloc(H * sizeof(float));
    float * h2 = (float*)malloc(H * sizeof(float));
    
    // Copy weights
    struct LayerWLocal { float *an,*fn,*qw,*qb,*kw,*kb,*vw,*vb,*ow,*ob,*gw,*uw,*dw; };
    std::vector<LayerWLocal> layers(N);
    for (int i = 0; i < N; i++) {
        layers[i].an=w.layers[i].an; layers[i].fn=w.layers[i].fn;
        layers[i].qw=w.layers[i].qw; layers[i].qb=w.layers[i].qb;
        layers[i].kw=w.layers[i].kw; layers[i].kb=w.layers[i].kb;
        layers[i].vw=w.layers[i].vw; layers[i].vb=w.layers[i].vb;
        layers[i].ow=w.layers[i].ow; layers[i].ob=w.layers[i].ob;
        layers[i].gw=w.layers[i].gw; layers[i].uw=w.layers[i].uw;
        layers[i].dw=w.layers[i].dw;
    }
    float * final_norm = w.final_norm;
    
    for (int i = 0; i < N; i++) {
        auto & l = layers[i];
        // RMSNorm (single token)
        manual_rms_norm(normed, h, l.an, H);
        // QKV (single token)
        manual_linear(q, normed, l.qw, l.qb, H, H);
        manual_linear(k, normed, l.kw, l.kb, H, KVD);
        manual_linear(v, normed, l.vw, l.vb, H, KVD);
        
        float * kc = cache.k_cache[i];
        float * vc = cache.v_cache[i];
        
        memset(ao, 0, H * sizeof(float));
        float sc = 1.0f / sqrtf((float)HD);
        
        for (int head = 0; head < NH; head++) {
            int kv_h = head * NKV / NH;
            float qh[128];
            for (int d = 0; d < HD; d++) qh[d] = q[head*HD + d];
            
            // Extract new K for this KV head
            float kh_new[128];
            for (int d = 0; d < HD; d++) kh_new[d] = k[kv_h*HD + d];
            
            // Apply RoPE to Q (position = seq_len) and new K
            float qh_r[128], kh_r[128];
            memcpy(qh_r, qh, HD*sizeof(float)); memcpy(kh_r, kh_new, HD*sizeof(float));
            manual_rope_theta(qh_r, kh_r, 1, HD, 1000000.0f);
            // Adjust RoPE position: the new token is at position seq_len
            // manual_rope_theta uses s=0 for single token. We need position = seq_len.
            // Re-apply with correct position
            { int half=HD/2; float theta=1.0f; float ts=powf(1000000.0f,-2.0f/HD);
              for(int d=0;d<half;d++){float cs=cosf((float)seq_len*theta),sn=sinf((float)seq_len*theta);
                int i1=d,i2=d+half;float q0=qh[i1],q1=qh[i2],k0=kh_new[i1],k1=kh_new[i2];
                qh_r[i1]=q0*cs-q1*sn;qh_r[i2]=q1*cs+q0*sn;
                kh_r[i1]=k0*cs-k1*sn;kh_r[i2]=k1*cs+k0*sn;theta*=ts;}}
            
            // Scores: dot product against ALL cached K (which already have RoPE from init)
            int total_len = seq_len + 1;
            float * scores = (float*)alloca(total_len * sizeof(float));
            
            for (int j = 0; j < seq_len; j++) {
                const float * kj = kc + kv_h * cache.max_seq * HD + j * HD;
                float s = 0; for (int d = 0; d < HD; d++) s += qh_r[d] * kj[d];
                scores[j] = s * sc;
            }
            // Score against new K
            { float s = 0; for (int d = 0; d < HD; d++) s += qh_r[d] * kh_r[d];
              scores[seq_len] = s * sc; }
            
            // Causal: all positions visible
            manual_softmax(scores, total_len);
            
            // Weighted sum against cached V + new V
            float out[128] = {0};
            for (int j = 0; j < seq_len; j++) {
                const float * vj = vc + kv_h * cache.max_seq * HD + j * HD;
                for (int d = 0; d < HD; d++) out[d] += scores[j] * vj[d];
            }
            // V for new token (no RoPE on V)
            for (int d = 0; d < HD; d++) {
                float vn = v[kv_h*HD + d];
                out[d] += scores[seq_len] * vn;
            }
            
            for (int d = 0; d < HD; d++) ao[head*HD + d] = out[d];
        }
        
        // Save K (with RoPE!) and V for this new token
        // K needs RoPE applied at position seq_len
        { int half=HD/2; float theta=1.0f; float ts=powf(1000000.0f,-2.0f/HD);
          for(int h=0;h<NKV;h++){float*kh2=kc+h*cache.max_seq*HD+seq_len*HD;
            memcpy(kh2,k+h*HD,HD*sizeof(float));theta=1.0f;
            for(int d=0;d<half;d++){float cs=cosf((float)seq_len*theta),sn=sinf((float)seq_len*theta);
              int i1=d,i2=d+half;float k0=kh2[i1],k1=kh2[i2];
              kh2[i1]=k0*cs-k1*sn;kh2[i2]=k1*cs+k0*sn;theta*=ts;}}}
        // V: no RoPE needed, save at correct position
        for (int h = 0; h < NKV; h++)
            memcpy(vc + h * cache.max_seq * HD + seq_len * HD, v + h * HD, HD * sizeof(float));
        
        // O projection + residual
        manual_linear(attn_out, ao, l.ow, l.ob, H, H);
        for (int j = 0; j < H; j++) h2[j] = h[j] + attn_out[j];
        
        // MLP
        manual_rms_norm(normed, h2, l.fn, H);
        manual_linear(gate, normed, l.gw, nullptr, H, F);
        manual_linear(up, normed, l.uw, nullptr, H, F);
        for (int j = 0; j < F; j++) gate[j] = gate[j]/(1.0f+expf(-gate[j])) * up[j];
        manual_linear(mlp_out, gate, l.dw, nullptr, F, H);
        for (int j = 0; j < H; j++) h[j] = h2[j] + mlp_out[j];
    }
    cache.seq_len++;
    
    // Final norm
    manual_rms_norm(h2, h, final_norm, H);
    memcpy(hidden_out, h2, H * sizeof(float));
    
    free(h); free(h2); free(normed); free(q); free(k); free(v); free(ao); free(attn_out);
    free(gate); free(up); free(mlp_out);
}
