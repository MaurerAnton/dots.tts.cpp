// SPDX-License-Identifier: GPL-3.0-or-later
// Hybrid DiT: manual cond/adaLN + ggml attention/FFN (fast!)
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "ggml.h"
#include "ggml-cpu.h"
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

// Manual cond computation (time_embed + speaker)
static void compute_cond(dit_model & m, float t_val, const float * spk_emb, float * cond) {
    int half=128; float se[256], h1[1024];
    for(int i=0;i<half;i++){float f=expf(-logf(10000.0f)*(float)i/(float)half);
        se[i]=cosf(t_val*f); se[half+i]=sinf(t_val*f);}
    {float*w1=tensor_data(m.t_embed_w1);float*b1=m.t_embed_b1?tensor_data(m.t_embed_b1):nullptr;
     for(int o=0;o<1024;o++){float s=b1?b1[o]:0.0f;for(int i=0;i<256;i++)s+=w1[o*256+i]*se[i];h1[o]=s/(1.0f+expf(-s));}}
    {float*w2=tensor_data(m.t_embed_w2);float*b2=m.t_embed_b2?tensor_data(m.t_embed_b2):nullptr;
     for(int o=0;o<1024;o++){float s=b2?b2[o]:0.0f;for(int i=0;i<1024;i++)s+=w2[o*1024+i]*h1[i];cond[o]=s;}}
    if(spk_emb){float t[1024],*sw1=tensor_data(m.spk_proj_w1),*sb1=m.spk_proj_b1?tensor_data(m.spk_proj_b1):nullptr;
     for(int o=0;o<1024;o++){float s=sb1?sb1[o]:0.0f;for(int i=0;i<512;i++)s+=sw1[o*512+i]*spk_emb[i];t[o]=s;}
     float mean=0;for(int i=0;i<1024;i++)mean+=t[i];mean/=1024;
     float var=0;for(int i=0;i<1024;i++){float d=t[i]-mean;var+=d*d;}var=var/1024+1e-5f;
     float istd=1.0f/sqrtf(var);float*lw=m.spk_ln_w?tensor_data(m.spk_ln_w):nullptr,*lb=m.spk_ln_b?tensor_data(m.spk_ln_b):nullptr;
     for(int i=0;i<1024;i++){float x=(t[i]-mean)*istd;if(lw)x*=lw[i];if(lb)x+=lb[i];cond[i]+=x;}}
}

// Hybrid block: manual adaLN + ggml attention/FFN
static ggml_tensor * hybrid_block(ggml_context * ctx, ggml_tensor * x,
    const float * cond, dit_block & block, int seq_len, int n_batch)
{
    int hidden=DIT_HIDDEN_SIZE, n_tokens=seq_len*n_batch;
    
    // Manual adaLN
    float cs[1024]; for(int i=0;i<1024;i++) cs[i]=cond[i]/(1.0f+expf(-cond[i]));
    float adaln_raw[6*DIT_HIDDEN_SIZE];
    {float*aw=tensor_data(block.adaln_linear_w);float*ab=block.adaln_linear_b?tensor_data(block.adaln_linear_b):nullptr;
     for(int o=0;o<6*hidden;o++){float s=ab?ab[o]:0.0f;for(int i=0;i<DIT_HIDDEN_SIZE;i++)s+=aw[o*DIT_HIDDEN_SIZE+i]*cs[i];adaln_raw[o]=s;}}
    
    // Create leaf tensors for adaLN shift/scale/gate
    float *sm=adaln_raw, *scm=adaln_raw+hidden, *gm=adaln_raw+2*hidden;
    float *sml=adaln_raw+3*hidden, *scl=adaln_raw+4*hidden, *gml=adaln_raw+5*hidden;
    
    auto make_leaf_2d = [&](const float * src, int h, int w) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h, w);
        memcpy(t->data, src, h*w*sizeof(float)); return t;
    };
    auto broadcast = [&](const float * src) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        float * d = (float*)t->data;
        for(int s=0;s<seq_len;s++) for(int b=0;b<n_batch;b++)
            memcpy(d+(b*seq_len+s)*hidden, src+b*hidden, hidden*sizeof(float));
        return t;
    };
    
    ggml_tensor * sm_t=broadcast(sm), * scm_t=broadcast(scm), * gm_t=broadcast(gm);
    ggml_tensor * sml_t=broadcast(sml), * scl_t=broadcast(scl), * gml_t=broadcast(gml);
    ggml_tensor * ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    for(int i=0;i<hidden*n_tokens;i++) ((float*)ones->data)[i]=1.0f;
    
    auto layernorm_copy = [&](ggml_tensor * x) -> ggml_tensor * {
        // Create a copy, normalize it, return normalized copy
        ggml_tensor * nx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        float * xd = (float*)x->data, * nd = (float*)nx->data;
        memcpy(nd, xd, hidden*n_tokens*sizeof(float));
        for (int t = 0; t < n_tokens; t++) {
            float mean = 0;
            for (int i = 0; i < hidden; i++) mean += nd[t*hidden + i];
            mean /= hidden;
            float var = 0;
            for (int i = 0; i < hidden; i++) { float d = nd[t*hidden + i] - mean; var += d * d; }
            float istd = 1.0f / sqrtf(var / hidden + 1e-5f);
            for (int i = 0; i < hidden; i++) nd[t*hidden + i] = (nd[t*hidden + i] - mean) * istd;
        }
        return nx;
    };
    
    // Attention
    ggml_tensor * h = layernorm_copy(x);
    if(block.attn_norm_w) h = ggml_mul(ctx, h, block.attn_norm_w);
    h = ggml_mul(ctx, ggml_add(ctx, ones, scm_t), h);
    h = ggml_add(ctx, h, sm_t);
    ggml_tensor * attn = dit_attention_multihead(ctx, h,
        block.attn_q_weight, block.attn_k_weight, block.attn_v_weight, block.attn_o_weight,
        nullptr,nullptr,nullptr, block.attn_o_bias?block.attn_o_bias:nullptr,
        seq_len, n_batch, DIT_NUM_HEADS, DIT_HEAD_SIZE, block.q_norm_w, block.k_norm_w);
    x = ggml_add(ctx, x, ggml_mul(ctx, gm_t, attn));
    
    // FFN
    h = layernorm_copy(x);
    if(block.ffn_norm_w) h = ggml_mul(ctx, h, block.ffn_norm_w);
    h = ggml_mul(ctx, ggml_add(ctx, ones, scl_t), h);
    h = ggml_add(ctx, h, sml_t);
    ggml_tensor * ffn = ggml_mul_mat(ctx, block.ffn_w1, h);
    if(block.ffn_b1) ffn = ggml_add(ctx, ffn, block.ffn_b1);
    ffn = ggml_gelu(ctx, ffn);  // GELU (not SiLU!) for FFN
    ffn = ggml_mul_mat(ctx, block.ffn_w2, ffn);
    if(block.ffn_b2) ffn = ggml_add(ctx, ffn, block.ffn_b2);
    x = ggml_add(ctx, x, ggml_mul(ctx, gml_t, ffn));
    return x;
}

void dit_forward_raw(dit_model & model, const float * x, int seq_len, float t_val,
    const float * speaker_emb, float * out) {
    int hidden=DIT_HIDDEN_SIZE, n_tokens=seq_len;
    
    // Cond
    float cond[1024]; compute_cond(model, t_val, speaker_emb, cond);
    
    // Create ggml context for this forward pass
    ggml_init_params gp = { .mem_size = 256ULL*1024*1024 };
    ggml_context * ctx = ggml_init(gp);
    
    // Input layer
    ggml_tensor * h_ggml = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    float * hd = (float*)h_ggml->data;
    {float*iw=tensor_data(model.input_layer_w);float*ib=model.input_layer_b?tensor_data(model.input_layer_b):nullptr;
     for(int t=0;t<n_tokens;t++) for(int o=0;o<hidden;o++){float s=ib?ib[o]:0.0f;
         for(int i=0;i<hidden;i++) s+=iw[o*hidden+i]*x[t*hidden+i]; hd[t*hidden+o]=s;}}
    
    // Blocks
    for(int i=0;i<model.n_layers;i++)
        h_ggml = hybrid_block(ctx, h_ggml, cond, model.layers[i], n_tokens, 1);
    
    // Output layer (manual)
    { ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, h_ggml);
      ggml_graph_compute_with_ctx(ctx, gf, 1); }
    float * block_out = (float*)h_ggml->data;
    
    float cs[1024]; for(int i=0;i<1024;i++) cs[i]=cond[i]/(1.0f+expf(-cond[i]));
    float mod_raw[2*DIT_HIDDEN_SIZE];
    {float*aw=tensor_data(model.out_adaln_w);float*ab=model.out_adaln_b?tensor_data(model.out_adaln_b):nullptr;
     for(int o=0;o<2*hidden;o++){float s=ab?ab[o]:0.0f;for(int i=0;i<DIT_HIDDEN_SIZE;i++)s+=aw[o*DIT_HIDDEN_SIZE+i]*cs[i];mod_raw[o]=s;}}
    float*shift=mod_raw,*scale=mod_raw+hidden;
    for(int t=0;t<n_tokens;t++){
        float mean=0; for(int i=0;i<hidden;i++) mean+=block_out[t*hidden+i]; mean/=hidden;
        float var=0; for(int i=0;i<hidden;i++){float d=block_out[t*hidden+i]-mean;var+=d*d;}
        float istd=1.0f/sqrtf(var/hidden+1e-5f);
        for(int i=0;i<hidden;i++) block_out[t*hidden+i]=((block_out[t*hidden+i]-mean)*istd)*(1.0f+scale[i])+shift[i];
    }
    float*ow=tensor_data(model.out_proj_w);float*ob=model.out_proj_b?tensor_data(model.out_proj_b):nullptr;
    for(int t=0;t<n_tokens;t++) for(int o=0;o<VAE_LATENT_DIM;o++){float s=ob?ob[o]:0.0f;
        for(int i=0;i<hidden;i++) s+=ow[o*hidden+i]*block_out[t*hidden+i]; out[t*VAE_LATENT_DIM+o]=s;}
    
    ggml_free(ctx);
}
