// SPDX-License-Identifier: GPL-3.0-or-later
// Dump C++ PE layer 0 intermediates for comparison with Python
#include "patchenc.h"
#include "safetensors.h"
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "ggml.h"
#include "ggml-cpu.h"
bool load_patchenc_weights(SafeTensorsFile & sf, ggml_context * w_ctx, patch_encoder & enc);
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

static void dump_f(const char * name, const float * d, int n) {
    float r=0; for(int i=0;i<n;i++) r+=d[i]*d[i];
    printf("  %-30s RMS=%.6f\n", name, sqrtf(r/n));
    char p[256]; snprintf(p,sizeof(p),"/tmp/pe_layer0_cpp/%s.bin",name);
    FILE * f=fopen(p,"wb"); if(f){fwrite(d,sizeof(float),n,f);fclose(f);}
}

int main() {
    const char * mp = getenv("DOTS_TTS_MODEL");
    if(!mp) mp="/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/model.safetensors";
    SafeTensorsFile sf; if(!sf.open(mp)) return 1;
    ggml_init_params gp={.mem_size=2048ULL*1024*1024};
    ggml_context * wc=ggml_init(gp);
    patch_encoder pe; if(!load_patchenc_weights(sf,wc,pe)) return 1;
    sf.close();

    float inp[512]; FILE * f=fopen("/tmp/pe_python_input.bin","rb");
    if(!f) return 1; fread(inp,sizeof(float),512,f); fclose(f);
    dump_f("input",inp,512);

    ggml_init_params cp={.mem_size=256ULL*1024*1024};
    ggml_context * ctx=ggml_init(cp);
    ggml_tensor * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,PATCHENC_LATENT_DIM,PATCHENC_PATCH_SIZE);
    memcpy(tensor_data(x),inp,512*sizeof(float));

    int seq=PATCHENC_PATCH_SIZE, ch=PATCHENC_LATENT_DIM;

    // ds_proj
    { float *xd=(float*)malloc(seq*ch*sizeof(float));
      memcpy(xd,tensor_data(x),seq*ch*sizeof(float));
      float *wd=tensor_data(pe.conv_weight); float *bd=pe.conv_bias?tensor_data(pe.conv_bias):nullptr;
      float *od=tensor_data(x); int os=seq/2;
      for(int o=0;o<os;o++) for(int oc=0;oc<ch;oc++){
        float s=bd?bd[oc]:0;
        for(int ic=0;ic<ch;ic++){int wb=(oc*ch+ic)*2;
          int i0=2*o-1; if(i0>=0&&i0<seq)s+=xd[i0*ch+ic]*wd[wb+0];
          int i1=2*o;   if(i1>=0&&i1<seq)s+=xd[i1*ch+ic]*wd[wb+1];}
        od[o*ch+oc]=s;}
      memset(od+os*ch,0,(seq-os)*ch*sizeof(float));
      dump_f("after_ds_proj",od,os*ch); free(xd); }
    int n_tok=seq/2, hidden=PATCHENC_HIDDEN;

    // in_proj
    ggml_tensor * xc=ggml_view_2d(ctx,x,ch,n_tok,x->nb[1],0); xc=ggml_cont(ctx,xc);
    ggml_tensor * h=ggml_mul_mat(ctx,pe.in_proj_w,xc);
    if(pe.in_proj_b){ggml_tensor*ib=ggml_reshape_2d(ctx,pe.in_proj_b,hidden,1);h=ggml_add(ctx,h,ib);}
    {ggml_cgraph*cg=ggml_new_graph(ctx);ggml_build_forward_expand(cg,h);ggml_graph_compute_with_ctx(ctx,cg,8);}
    dump_f("after_in_proj",tensor_data(h),hidden*n_tok);

    float *hd=(float*)malloc(hidden*n_tok*sizeof(float));
    memcpy(hd,tensor_data(h),hidden*n_tok*sizeof(float));

    // Layer 0
    { patchenc_layer &l=pe.layers[0];
      ggml_reset(ctx);
      ggml_tensor*xi=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,hidden,n_tok);
      memcpy(tensor_data(xi),hd,hidden*n_tok*sizeof(float));

      // attn_norm
      ggml_tensor*xn=ggml_rms_norm(ctx,xi,1e-6f);
      if(l.attn_norm_w)xn=ggml_mul(ctx,xn,l.attn_norm_w);
      {ggml_cgraph*cg=ggml_new_graph(ctx);ggml_build_forward_expand(cg,xn);ggml_graph_compute_with_ctx(ctx,cg,8);}
      dump_f("attn_norm_out",tensor_data(xn),hidden*n_tok);

      // Q/K/V via ggml
      ggml_tensor*q=ggml_mul_mat(ctx,l.attn_q_weight,xn);
      ggml_tensor*k=ggml_mul_mat(ctx,l.attn_k_weight,xn);
      ggml_tensor*v=ggml_mul_mat(ctx,l.attn_v_weight,xn);
      q=ggml_cont(ctx,q);k=ggml_cont(ctx,k);v=ggml_cont(ctx,v);
      {ggml_cgraph*cg=ggml_new_graph(ctx);ggml_build_forward_expand(cg,q);
       ggml_build_forward_expand(cg,k);ggml_build_forward_expand(cg,v);
       ggml_graph_compute_with_ctx(ctx,cg,8);}
      dump_f("q_after_proj",tensor_data(q),hidden*n_tok);
      dump_f("k_after_proj",tensor_data(k),hidden*n_tok);
      dump_f("v_after_proj",tensor_data(v),hidden*n_tok);

      // Per-head qk_norm
      float*qd=(float*)malloc(hidden*n_tok*sizeof(float));
      float*kd=(float*)malloc(hidden*n_tok*sizeof(float));
      float*vd=(float*)malloc(hidden*n_tok*sizeof(float));
      memcpy(qd,tensor_data(q),hidden*n_tok*sizeof(float));
      memcpy(kd,tensor_data(k),hidden*n_tok*sizeof(float));
      memcpy(vd,tensor_data(v),hidden*n_tok*sizeof(float));
      // per-head norm
      for(int t=0;t<n_tok;t++) for(int hh=0;hh<PATCHENC_NUM_HEADS;hh++){
        float rq=0,rk=0; int base=hh*PATCHENC_HEAD_SIZE+t*hidden;
        for(int d=0;d<PATCHENC_HEAD_SIZE;d++){rq+=qd[base+d]*qd[base+d];rk+=kd[base+d]*kd[base+d];}
        rq=sqrtf(rq/PATCHENC_HEAD_SIZE+1e-6f);rk=sqrtf(rk/PATCHENC_HEAD_SIZE+1e-6f);
        for(int d=0;d<PATCHENC_HEAD_SIZE;d++){qd[base+d]/=rq;kd[base+d]/=rk;}}
      // Reshape to [head_dim, n_heads, n_tokens] for comparison
      { float*qh=(float*)malloc(PATCHENC_HEAD_SIZE*PATCHENC_NUM_HEADS*n_tok*sizeof(float));
        float*kh=(float*)malloc(PATCHENC_HEAD_SIZE*PATCHENC_NUM_HEADS*n_tok*sizeof(float));
        for(int hh=0;hh<PATCHENC_NUM_HEADS;hh++) for(int t=0;t<n_tok;t++)
          for(int d=0;d<PATCHENC_HEAD_SIZE;d++){
            qh[d+hh*PATCHENC_HEAD_SIZE+t*PATCHENC_HEAD_SIZE*PATCHENC_NUM_HEADS]=qd[(hh*PATCHENC_HEAD_SIZE+d)+t*hidden];
            kh[d+hh*PATCHENC_HEAD_SIZE+t*PATCHENC_HEAD_SIZE*PATCHENC_NUM_HEADS]=kd[(hh*PATCHENC_HEAD_SIZE+d)+t*hidden];}
        dump_f("q_after_qknorm_reshaped",qh,PATCHENC_HEAD_SIZE*PATCHENC_NUM_HEADS*n_tok);
        dump_f("k_after_qknorm_reshaped",kh,PATCHENC_HEAD_SIZE*PATCHENC_NUM_HEADS*n_tok);
        free(qh);free(kh);}

      // Continue with ggml: RoPE + attention + FFN
      ggml_tensor*qn=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,hidden,n_tok);
      ggml_tensor*kn=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,hidden,n_tok);
      ggml_tensor*vn=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,hidden,n_tok);
      memcpy(tensor_data(qn),qd,hidden*n_tok*sizeof(float));
      memcpy(tensor_data(kn),kd,hidden*n_tok*sizeof(float));
      memcpy(tensor_data(vn),vd,hidden*n_tok*sizeof(float));
      free(qd);free(kd);free(vd);

      // RoPE
      ggml_tensor*q3=ggml_reshape_3d(ctx,qn,PATCHENC_HEAD_SIZE,PATCHENC_NUM_HEADS,n_tok);
      ggml_tensor*k3=ggml_reshape_3d(ctx,kn,PATCHENC_HEAD_SIZE,PATCHENC_NUM_HEADS,n_tok);
      ggml_tensor*pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,n_tok);
      {int*pd=(int*)tensor_data(pos);for(int s=0;s<n_tok;s++)pd[s]=s;}
      q3=ggml_rope(ctx,q3,pos,PATCHENC_HEAD_SIZE,0);
      k3=ggml_rope(ctx,k3,pos,PATCHENC_HEAD_SIZE,0);
      {ggml_cgraph*cg=ggml_new_graph(ctx);ggml_build_forward_expand(cg,q3);
       ggml_build_forward_expand(cg,k3);ggml_graph_compute_with_ctx(ctx,cg,8);}

      // Extract data for CPU per-head attention
      float* qrd=(float*)malloc(hidden*n_tok*sizeof(float));
      float* krd=(float*)malloc(hidden*n_tok*sizeof(float));
      float* vrd=(float*)malloc(hidden*n_tok*sizeof(float));
      memcpy(qrd,tensor_data(q3),hidden*n_tok*sizeof(float));
      memcpy(krd,tensor_data(k3),hidden*n_tok*sizeof(float));
      memcpy(vrd,tensor_data(vn),hidden*n_tok*sizeof(float));

      // Per-head attention CPU (matching Python)
      float* ctx_d=(float*)calloc(hidden*n_tok,sizeof(float));
      float sc=1.0f/sqrtf((float)PATCHENC_HEAD_SIZE);
      float* scr=(float*)malloc(n_tok*n_tok*sizeof(float));
      for(int h=0;h<PATCHENC_NUM_HEADS;h++){
        for(int ti=0;ti<n_tok;ti++){
          for(int tj=0;tj<n_tok;tj++){
            float s=0;for(int d=0;d<PATCHENC_HEAD_SIZE;d++)
              s+=qrd[(h*PATCHENC_HEAD_SIZE+d)+ti*hidden]*krd[(h*PATCHENC_HEAD_SIZE+d)+tj*hidden];
            scr[ti*n_tok+tj]=s*sc;}
          for(int tj=ti+1;tj<n_tok;tj++)scr[ti*n_tok+tj]=-INFINITY;
          float mx=scr[ti*n_tok];for(int tj=1;tj<n_tok;tj++)if(scr[ti*n_tok+tj]>mx)mx=scr[ti*n_tok+tj];
          float se=0;for(int tj=0;tj<n_tok;tj++){scr[ti*n_tok+tj]=expf(scr[ti*n_tok+tj]-mx);se+=scr[ti*n_tok+tj];}
          for(int tj=0;tj<n_tok;tj++)scr[ti*n_tok+tj]/=se;}
        for(int ti=0;ti<n_tok;ti++)for(int d=0;d<PATCHENC_HEAD_SIZE;d++){
          float s=0;for(int tj=0;tj<n_tok;tj++)s+=scr[ti*n_tok+tj]*vrd[(h*PATCHENC_HEAD_SIZE+d)+tj*hidden];
          ctx_d[(h*PATCHENC_HEAD_SIZE+d)+ti*hidden]+=s;}}
      free(scr);free(qrd);free(krd);free(vrd);
      dump_f("attn_context_perhead",ctx_d,hidden*n_tok);

      ggml_tensor*ao=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,hidden,n_tok);
      memcpy(tensor_data(ao),ctx_d,hidden*n_tok*sizeof(float));
      free(ctx_d);
      ao=ggml_mul_mat(ctx,l.attn_o_weight,ao);
      if(l.attn_o_bias){ggml_tensor*ob=ggml_reshape_2d(ctx,l.attn_o_bias,hidden,1);ao=ggml_add(ctx,ao,ob);}
      {ggml_cgraph*cg=ggml_new_graph(ctx);ggml_build_forward_expand(cg,ao);
       ggml_graph_compute_with_ctx(ctx,cg,8);}
      dump_f("o_proj_out_perhead",tensor_data(ao),hidden*n_tok);

      ggml_tensor*h1=ggml_add(ctx,xi,ao);
      {ggml_cgraph*cg=ggml_new_graph(ctx);ggml_build_forward_expand(cg,h1);
       ggml_graph_compute_with_ctx(ctx,cg,8);}
      dump_f("after_attn_residual",tensor_data(h1),hidden*n_tok);

      // FFN
      ggml_tensor*fn=ggml_rms_norm(ctx,h1,1e-6f);
      if(l.ffn_norm_w)fn=ggml_mul(ctx,fn,l.ffn_norm_w);
      {ggml_cgraph*cg=ggml_new_graph(ctx);ggml_build_forward_expand(cg,fn);
       ggml_graph_compute_with_ctx(ctx,cg,8);}
      dump_f("ffn_norm_out",tensor_data(fn),hidden*n_tok);

      ggml_tensor*ff=ggml_mul_mat(ctx,l.ffn_w1,fn);
      if(l.ffn_b1){ggml_tensor*fb=ggml_reshape_2d(ctx,l.ffn_b1,PATCHENC_FFN_SIZE,1);ff=ggml_add(ctx,ff,fb);}
      {ggml_cgraph*cg=ggml_new_graph(ctx);ggml_build_forward_expand(cg,ff);
       ggml_graph_compute_with_ctx(ctx,cg,8);}
      dump_f("ffn_fc1",tensor_data(ff),PATCHENC_FFN_SIZE*n_tok);

      ff=ggml_silu(ctx,ff);
      {ggml_cgraph*cg=ggml_new_graph(ctx);ggml_build_forward_expand(cg,ff);
       ggml_graph_compute_with_ctx(ctx,cg,8);}
      dump_f("ffn_silu",tensor_data(ff),PATCHENC_FFN_SIZE*n_tok);

      ff=ggml_mul_mat(ctx,l.ffn_w2,ff);
      if(l.ffn_b2){ggml_tensor*fb=ggml_reshape_2d(ctx,l.ffn_b2,hidden,1);ff=ggml_add(ctx,ff,fb);}
      {ggml_cgraph*cg=ggml_new_graph(ctx);ggml_build_forward_expand(cg,ff);
       ggml_graph_compute_with_ctx(ctx,cg,8);}
      dump_f("ffn_fc2",tensor_data(ff),hidden*n_tok);

      ggml_tensor*ho=ggml_add(ctx,h1,ff);
      {ggml_cgraph*cg=ggml_new_graph(ctx);ggml_build_forward_expand(cg,ho);
       ggml_graph_compute_with_ctx(ctx,cg,8);}
      dump_f("layer0_out",tensor_data(ho),hidden*n_tok);
    }

    ggml_free(ctx);ggml_free(wc);free(hd);
    return 0;
}
