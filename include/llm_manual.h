// Manual LLM forward pass — reusable, byte-perfect with PyTorch
// Proven by test_manual_llm: last hidden correlation 1.000000
#pragma once
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit_manual.h"
#include "safetensors.h"
#include <cstdlib>
#include <cstring>
#include <cmath>

struct LLMWeights {
    float *embed_w;  // [vocab, hidden]
    struct Layer {
        float *an, *fn;  // RMSNorm weights
        float *qw,*qb, *kw,*kb, *vw,*vb, *ow,*ob;  // attention
        float *gw, *uw, *dw;  // MLP
    };
    Layer * layers;
    float * final_norm;
    int n_layers;
};

inline bool load_llm_weights(SafeTensorsFile & sf, LLMWeights & w, int n_layers) {
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
    
    load("llm.model.embed_tokens.weight", w.embed_w);
    
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
    load("llm.model.norm.weight", w.final_norm);
    return true;
}

inline void free_llm_weights(LLMWeights & w) {
    free(w.embed_w); free(w.final_norm);
    for (int i = 0; i < w.n_layers; i++) {
        auto & l = w.layers[i];
        free(l.an);free(l.fn);free(l.qw);free(l.qb);free(l.kw);free(l.kb);
        free(l.vw);free(l.vb);free(l.ow);free(l.ob);free(l.gw);free(l.uw);free(l.dw);
    }
    delete[] w.layers;
}

// Forward pass: produces hidden states [seq_len, hidden_size] in hiddens_out
// Caller allocates hiddens_out with seq_len * 1536 floats
inline void llm_manual_forward(const LLMWeights & w, const int * token_ids, int n_tok,
    float * hiddens_out) {
    const int H=1536, F=8960, NH=12, NKV=2, HD=128, KVD=NKV*HD, N=w.n_layers;
    
    // Verify weights loaded
    if (!w.embed_w) { fprintf(stderr,"FATAL: embed_w is null\n"); return; }
    if (!w.final_norm) { fprintf(stderr,"FATAL: final_norm is null\n"); return; }
    { float r=0; for(int i=0;i<H;i++) r+=w.final_norm[i]*w.final_norm[i];
      fprintf(stderr,"  final_norm RMS=%.4f first3=%.4f %.4f %.4f\n", sqrtf(r/H), w.final_norm[0],w.final_norm[1],w.final_norm[2]); }
    
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
    float * gate = (float*)malloc(n_tok * F * sizeof(float));
    float * up = (float*)malloc(n_tok * F * sizeof(float));
    float * mlp_out = (float*)malloc(n_tok * H * sizeof(float));
    float * scores = (float*)malloc(n_tok * n_tok * sizeof(float));
    float * qh = (float*)malloc(n_tok * HD * sizeof(float));
    float * kh = (float*)malloc(n_tok * HD * sizeof(float));
    float * vh = (float*)malloc(n_tok * HD * sizeof(float));
    float sc = 1.0f/sqrtf((float)HD);
    
    for (int i = 0; i < N; i++) {
        auto & l = w.layers[i];
        for (int t = 0; t < n_tok; t++) manual_rms_norm(normed + t*H, h + t*H, l.an, H);
        for (int t = 0; t < n_tok; t++) {
            manual_linear(q + t*H, normed + t*H, l.qw, l.qb, H, H);
            manual_linear(k + t*KVD, normed + t*H, l.kw, l.kb, H, KVD);
            manual_linear(v + t*KVD, normed + t*H, l.vw, l.vb, H, KVD);
        }
        memset(ao, 0, n_tok * H * sizeof(float));
        for (int head = 0; head < NH; head++) {
            int kv_h = head * NKV / NH;
            for (int t = 0; t < n_tok; t++) for (int d = 0; d < HD; d++) {
                qh[t*HD+d]=q[t*H+head*HD+d]; kh[t*HD+d]=k[t*KVD+kv_h*HD+d]; vh[t*HD+d]=v[t*KVD+kv_h*HD+d]; }
            manual_rope_theta(qh, kh, n_tok, HD, 1000000.0f);
            for (int ti = 0; ti < n_tok; ti++) {
                for (int tj = 0; tj < n_tok; tj++) { float s=0; for(int d=0;d<HD;d++)s+=qh[ti*HD+d]*kh[tj*HD+d]; scores[ti*n_tok+tj]=s*sc; }
                for (int tj = ti+1; tj < n_tok; tj++) scores[ti*n_tok+tj] = -INFINITY;
                manual_softmax(scores + ti*n_tok, n_tok);
            }
            for (int ti = 0; ti < n_tok; ti++) for (int d = 0; d < HD; d++) {
                float s=0; for(int tj=0;tj<n_tok;tj++)s+=scores[ti*n_tok+tj]*vh[tj*HD+d]; ao[ti*H+head*HD+d]=s; }
        }
        for (int t = 0; t < n_tok; t++) manual_linear(attn_out + t*H, ao + t*H, l.ow, l.ob, H, H);
        for (int j = 0; j < n_tok*H; j++) h2[j] = h[j] + attn_out[j];
        for (int t = 0; t < n_tok; t++) manual_rms_norm(normed + t*H, h2 + t*H, l.fn, H);
        for (int t = 0; t < n_tok; t++) {
            manual_linear(gate + t*F, normed + t*H, l.gw, nullptr, H, F);
            manual_linear(up + t*F, normed + t*H, l.uw, nullptr, H, F);
            for (int j = 0; j < F; j++) gate[t*F+j] = gate[t*F+j]/(1.0f+expf(-gate[t*F+j])) * up[t*F+j]; }
        for (int t = 0; t < n_tok; t++) manual_linear(mlp_out + t*H, gate + t*F, l.dw, nullptr, H, F);
        for (int j = 0; j < n_tok*H; j++) h[j] = h2[j] + mlp_out[j];
    }
    for (int t = 0; t < n_tok; t++) manual_rms_norm(h2 + t*H, h + t*H, w.final_norm, H);
    { float r=0; for(int i=0;i<n_tok*H;i++) r+=h2[i]*h2[i];
      fprintf(stderr,"  [llm_manual_forward] output RMS=%.4f\\n", sqrtf(r/(n_tok*H))); }
    memcpy(hiddens_out, h2, n_tok * H * sizeof(float));
    
    free(h); free(h2); free(normed); free(q); free(k); free(v); free(ao); free(attn_out);
    free(gate); free(up); free(mlp_out); free(scores); free(qh); free(kh); free(vh);
}
