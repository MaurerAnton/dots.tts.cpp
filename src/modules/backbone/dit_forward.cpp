// SPDX-License-Identifier: GPL-3.0-or-later
// Hybrid dit_forward: manual cond + ggml blocks (faster than pure manual)
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <cstdio>

ggml_tensor * dit_attention_multihead(
    ggml_context * ctx, ggml_tensor * x,
    ggml_tensor * q_weight, ggml_tensor * k_weight, ggml_tensor * v_weight,
    ggml_tensor * o_weight,
    ggml_tensor * q_bias, ggml_tensor * k_bias, ggml_tensor * v_bias,
    ggml_tensor * o_bias,
    int seq_len, int n_batch,
    int n_heads, int head_dim, ggml_tensor * q_norm_w, ggml_tensor * k_norm_w);

static void manual_layernorm(float * out, const float * x, int n) {
    float mean = 0; for (int i = 0; i < n; i++) mean += x[i]; mean /= n;
    float var = 0; for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    float inv_std = 1.0f / sqrtf(var / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv_std;
}
static void manual_linear(float * out, const float * x, const float * w, const float * bias, int in_feat, int out_feat) {
    for (int o = 0; o < out_feat; o++) { float s = bias ? bias[o] : 0.0f;
        for (int i = 0; i < in_feat; i++) s += w[o * in_feat + i] * x[i]; out[o] = s; }
}

// Manual cond computation
static void compute_cond(dit_model & m, float t_val, const float * spk_emb, float * cond) {
    int half=128; float *se=new float[256], *h1=new float[1024];
    for(int i=0;i<half;i++){float f=expf(-logf(10000.0f)*(float)i/(float)half);
        se[i]=cosf(t_val*f); se[half+i]=sinf(t_val*f);}
    {float*w1=tensor_data(m.t_embed_w1);float*b1=m.t_embed_b1?tensor_data(m.t_embed_b1):nullptr;
     for(int o=0;o<1024;o++){float s=b1?b1[o]:0.0f;for(int i=0;i<256;i++)s+=w1[o*256+i]*se[i];h1[o]=s/(1.0f+expf(-s));}}
    {float*w2=tensor_data(m.t_embed_w2);float*b2=m.t_embed_b2?tensor_data(m.t_embed_b2):nullptr;
     for(int o=0;o<1024;o++){float s=b2?b2[o]:0.0f;for(int i=0;i<1024;i++)s+=w2[o*1024+i]*h1[i];cond[o]=s;}}
    if(spk_emb){float*t=new float[1024],*sw1=tensor_data(m.spk_proj_w1),*sb1=m.spk_proj_b1?tensor_data(m.spk_proj_b1):nullptr;
     for(int o=0;o<1024;o++){float s=sb1?sb1[o]:0.0f;for(int i=0;i<512;i++)s+=sw1[o*512+i]*spk_emb[i];t[o]=s;}
     float mean=0;for(int i=0;i<1024;i++)mean+=t[i];mean/=1024;
     float var=0;for(int i=0;i<1024;i++){float d=t[i]-mean;var+=d*d;}var=var/1024+1e-5f;
     float istd=1.0f/sqrtf(var);float*lw=m.spk_ln_w?tensor_data(m.spk_ln_w):nullptr,*lb=m.spk_ln_b?tensor_data(m.spk_ln_b):nullptr;
     for(int i=0;i<1024;i++){float x=(t[i]-mean)*istd;if(lw)x*=lw[i];if(lb)x+=lb[i];cond[i]+=x;}delete[] t;}
    delete[] se; delete[] h1;
}

void dit_forward_raw(dit_model & model, const float * x, int seq_len, float t_val,
    const float * speaker_emb, float * out) {
    int hidden=DIT_HIDDEN_SIZE, n_tokens=seq_len;
    float *cond=new float[1024]; compute_cond(model, t_val, speaker_emb, cond);

    float *h=new float[n_tokens*hidden];
    {float*iw=tensor_data(model.input_layer_w);float*ib=model.input_layer_b?tensor_data(model.input_layer_b):nullptr;
     for(int t=0;t<n_tokens;t++) manual_linear(h+t*hidden,x+t*hidden,iw,ib,hidden,hidden);}

    float *bo=new float[n_tokens*hidden];
    for(int i=0;i<model.n_layers;i++){
        manual_dit_block(h,cond,model.layers[i],bo,n_tokens);
        float*tmp=h;h=bo;bo=tmp;
    }

    float cs[1024];for(int i=0;i<1024;i++)cs[i]=cond[i]/(1.0f+expf(-cond[i]));
    float mod_raw[2*DIT_HIDDEN_SIZE];
    {float*aw=tensor_data(model.out_adaln_w);float*ab=model.out_adaln_b?tensor_data(model.out_adaln_b):nullptr;
     for(int o=0;o<2*hidden;o++){float s=ab?ab[o]:0.0f;for(int i=0;i<DIT_HIDDEN_SIZE;i++)s+=aw[o*DIT_HIDDEN_SIZE+i]*cs[i];mod_raw[o]=s;}}
    float*shift=mod_raw,*scale=mod_raw+hidden;
    float*ln=new float[n_tokens*hidden];
    for(int t=0;t<n_tokens;t++){manual_layernorm(ln+t*hidden,h+t*hidden,hidden);
        for(int i=0;i<hidden;i++)ln[t*hidden+i]=ln[t*hidden+i]*(1.0f+scale[i])+shift[i];}
    float*ow=tensor_data(model.out_proj_w);float*ob=model.out_proj_b?tensor_data(model.out_proj_b):nullptr;
    for(int t=0;t<n_tokens;t++) manual_linear(out+t*VAE_LATENT_DIM,ln+t*hidden,ow,ob,hidden,VAE_LATENT_DIM);
    delete[] h;delete[] bo;delete[] ln;delete[] cond;
}
