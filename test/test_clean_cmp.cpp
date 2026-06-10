// SPDX-License-Identifier: GPL-3.0-or-later
// Test: hybrid block (ggml) vs Python for pipeline input
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "safetensors.h"
#include <cstdio>
#include <cstring>
#include <cmath>

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);
ggml_tensor * dit_attention_multihead(
    ggml_context * ctx, ggml_tensor * x,
    ggml_tensor * q_weight, ggml_tensor * k_weight, ggml_tensor * v_weight,
    ggml_tensor * o_weight,
    ggml_tensor * q_bias, ggml_tensor * k_bias, ggml_tensor * v_bias,
    ggml_tensor * o_bias,
    int seq_len, int n_batch,
    int n_heads, int head_dim, ggml_tensor * q_norm_w, ggml_tensor * k_norm_w);

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";
    SafeTensorsFile sf; sf.open(model_path);
    ggml_init_params wp = { .mem_size = 3ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model m; load_dit_weights(sf, w_ctx, m); sf.close();

    float py_il[8192], py_cond[1024], py_b0[8192];
    { FILE * f = fopen("debug/py_il_out.bin","rb"); fread(py_il,sizeof(float),8192,f); fclose(f); }
    { FILE * f = fopen("debug/py_cond.bin","rb"); fread(py_cond,sizeof(float),1024,f); fclose(f); }
    { FILE * f = fopen("debug/py_b0_out.bin","rb"); fread(py_b0,sizeof(float),8192,f); fclose(f); }

    // Build hybrid block
    ggml_init_params gp = { .mem_size = 256*1024*1024 };
    ggml_context * ctx = ggml_init(gp);
    int hidden=1024, n_tokens=8;
    
    // Manual adaLN
    float cs[1024]; for(int i=0;i<1024;i++) cs[i]=py_cond[i]/(1.0f+expf(-py_cond[i]));
    float adaln_raw[6144];
    {float*aw=tensor_data(m.layers[0].adaln_linear_w);float*ab=m.layers[0].adaln_linear_b?tensor_data(m.layers[0].adaln_linear_b):nullptr;
     for(int o=0;o<6144;o++){float s=ab?ab[o]:0.0f;for(int i=0;i<1024;i++)s+=aw[o*1024+i]*cs[i];adaln_raw[o]=s;}}
    float *sm=adaln_raw,*scm=adaln_raw+1024,*gm=adaln_raw+2048;
    float *sml=adaln_raw+3072,*scl=adaln_raw+4096,*gml=adaln_raw+5120;
    
    auto broadcast = [&](const float * src) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        float * d = (float*)t->data;
        for(int s=0;s<n_tokens;s++) memcpy(d+s*hidden, src, hidden*sizeof(float));
        return t;
    };
    ggml_tensor *sm_t=broadcast(sm),*scm_t=broadcast(scm),*gm_t=broadcast(gm);
    ggml_tensor *sml_t=broadcast(sml),*scl_t=broadcast(scl),*gml_t=broadcast(gml);
    ggml_tensor * ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    for(int i=0;i<hidden*n_tokens;i++) ((float*)ones->data)[i]=1.0f;
    
    // Input x as ggml leaf tensor
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    memcpy(x->data, py_il, 8192*sizeof(float));
    
    // Manual LayerNorm
    ggml_tensor * normed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    { float *xd=(float*)x->data,*nd=(float*)normed->data; memcpy(nd,xd,8192*sizeof(float));
      for(int t=0;t<n_tokens;t++){float mean=0;for(int i=0;i<hidden;i++)mean+=nd[t*hidden+i];mean/=hidden;
        float var=0;for(int i=0;i<hidden;i++){float d2=nd[t*hidden+i]-mean;var+=d2*d2;}
        float istd=1.0f/sqrtf(var/hidden+1e-5f);
        for(int i=0;i<hidden;i++)nd[t*hidden+i]=(nd[t*hidden+i]-mean)*istd;}}
    
    // Attention
    ggml_tensor * h = ggml_mul(ctx, ggml_add(ctx, ones, scm_t), normed);
    h = ggml_add(ctx, h, sm_t);
    ggml_tensor * attn = dit_attention_multihead(ctx, h,
        m.layers[0].attn_q_weight,m.layers[0].attn_k_weight,m.layers[0].attn_v_weight,m.layers[0].attn_o_weight,
        nullptr,nullptr,nullptr,m.layers[0].attn_o_bias?m.layers[0].attn_o_bias:nullptr,
        n_tokens,1,16,64,m.layers[0].q_norm_w,m.layers[0].k_norm_w);
    x = ggml_add(ctx, x, ggml_mul(ctx, gm_t, attn));
    
    // FFN
    normed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    { float *xd=(float*)x->data,*nd=(float*)normed->data; memcpy(nd,xd,8192*sizeof(float));
      for(int t=0;t<n_tokens;t++){float mean=0;for(int i=0;i<hidden;i++)mean+=nd[t*hidden+i];mean/=hidden;
        float var=0;for(int i=0;i<hidden;i++){float d2=nd[t*hidden+i]-mean;var+=d2*d2;}
        float istd=1.0f/sqrtf(var/hidden+1e-5f);
        for(int i=0;i<hidden;i++)nd[t*hidden+i]=(nd[t*hidden+i]-mean)*istd;}}
    h = ggml_mul(ctx, ggml_add(ctx, ones, scl_t), normed);
    h = ggml_add(ctx, h, sml_t);
    ggml_tensor * ffn = ggml_mul_mat(ctx, m.layers[0].ffn_w1, h);
    if(m.layers[0].ffn_b1) ffn = ggml_add(ctx, ffn, m.layers[0].ffn_b1);
    ffn = ggml_gelu(ctx, ffn);
    ffn = ggml_mul_mat(ctx, m.layers[0].ffn_w2, ffn);
    if(m.layers[0].ffn_b2) ffn = ggml_add(ctx, ffn, m.layers[0].ffn_b2);
    x = ggml_add(ctx, x, ggml_mul(ctx, gml_t, ffn));
    
    { ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, x);
      ggml_graph_compute_with_ctx(ctx, gf, 1); }
    
    float * cpp_out = (float*)x->data;
    float r_py=0, r_cpp=0, max_d=0;
    for(int i=0;i<8192;i++){
        r_py+=py_b0[i]*py_b0[i]; r_cpp+=cpp_out[i]*cpp_out[i];
        float d=fabsf(py_b0[i]-cpp_out[i]); if(d>max_d)max_d=d;
    }
    printf("Hybrid block 0: Py RMS=%.4f C++ RMS=%.4f max_diff=%.6f\n",
        sqrtf(r_py/8192), sqrtf(r_cpp/8192), max_d);
    printf("%s\n", max_d<0.01?"MATCH!":"DIFFERS");

    ggml_free(ctx); ggml_free(w_ctx);
    return 0;
}
