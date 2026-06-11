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
