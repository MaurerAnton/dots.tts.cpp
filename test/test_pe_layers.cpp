// SPDX-License-Identifier: GPL-3.0-or-later
// Layer-by-layer PE dump for Python comparison
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

static void dump_tensor(const char * name, ggml_tensor * t, int n) {
    float * d = tensor_data(t);
    float rms=0; for(int i=0;i<n;i++) rms+=d[i]*d[i];
    rms = sqrtf(rms/n);
    printf("  %s: RMS=%.6f first3=%.6f %.6f %.6f\n", name, rms, d[0],d[1],d[2]);
    char fname[256]; snprintf(fname,sizeof(fname),"/tmp/pe_dump_cpp/%s.bin",name);
    FILE * f = fopen(fname,"wb"); if(f){fwrite(d,sizeof(float),n,f);fclose(f);}
}

static float* dump_cpu(const char * name, float * d, int n) {
    float rms=0; for(int i=0;i<n;i++) rms+=d[i]*d[i];
    rms = sqrtf(rms/n);
    printf("  %s: RMS=%.6f first3=%.6f %.6f %.6f\n", name, rms, d[0],d[1],d[2]);
    char fname[256]; snprintf(fname,sizeof(fname),"/tmp/pe_dump_cpp/%s.bin",name);
    FILE * f = fopen(fname,"wb"); if(f){fwrite(d,sizeof(float),n,f);fclose(f);}
    return d;
}

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/model.safetensors";
    
    SafeTensorsFile sf;
    if (!sf.open(model_path)) { fprintf(stderr, "FAILED open\n"); return 1; }
    
    ggml_init_params gparams = { .mem_size = 2048ULL*1024*1024 };
    ggml_context * w_ctx = ggml_init(gparams);
    
    patch_encoder pe;
    if (!load_patchenc_weights(sf, w_ctx, pe)) { fprintf(stderr, "FAILED pe load\n"); return 1; }
    sf.close();
    
    // Load Python's PE input (denormalized)
    float input[512];
    FILE * f = fopen("/tmp/pe_python_input.bin", "rb");
    if (!f) { fprintf(stderr, "No input file\n"); return 1; }
    fread(input, sizeof(float), 512, f); fclose(f);
    
    // Dump input
    dump_cpu("input", input, 512);
    
    ggml_init_params cparams = { .mem_size = 256ULL*1024*1024 };
    ggml_context * ctx = ggml_init(cparams);
    
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PATCHENC_LATENT_DIM, PATCHENC_PATCH_SIZE);
    memcpy(tensor_data(x), input, 512*sizeof(float));
    
    int seq = PATCHENC_PATCH_SIZE;
    int ch  = PATCHENC_LATENT_DIM;
    
    // 1. ds_proj: manual causal_conv1d_stride2
    // DEBUG: verify weights AFTER conv too
    { float * ww = tensor_data(pe.conv_weight); float * bb = pe.conv_bias ? tensor_data(pe.conv_bias) : nullptr;
      printf("  DEBUG pre:  w[0,0,0]=%.6f w[0,0,1]=%.6f b[0]=%.6f\n",
             ww[0], ww[1], bb?bb[0]:0.0f); }
    {
        float * x_data = tensor_data(x);
        float * w_data = tensor_data(pe.conv_weight);
        float * b_data = pe.conv_bias ? tensor_data(pe.conv_bias) : nullptr;
        int out_seq = seq / 2, kernel = 2;
        
        for (int o = 0; o < out_seq; o++) {
            for (int oc = 0; oc < ch; oc++) {
                float sum = b_data ? b_data[oc] : 0.0f;
                for (int ic = 0; ic < ch; ic++) {
                    int w_base = (oc * ch + ic) * 2;
                    int inp0 = 2 * o - 1;
                    if (inp0 >= 0 && inp0 < seq)
                        sum += x_data[inp0 * ch + ic] * w_data[w_base + 0];
                    int inp1 = 2 * o;
                    if (inp1 >= 0 && inp1 < seq)
                        sum += x_data[inp1 * ch + ic] * w_data[w_base + 1];
                }
                x_data[o * ch + oc] = sum;
            }
        }
        memset(x_data + out_seq * ch, 0, (seq - out_seq) * ch * sizeof(float));
        // Check first output element manually
        { float s = b_data?b_data[0]:0;
          for(int ic=0;ic<ch;ic++) s += x_data[0*ch + ic + 128*0] * w_data[(0*ch+ic)*2+1]; // wrong: uses x_data AFTER overwrite!
          printf("  DEBUG manual check: x[0]*w[0,0,1]+b = %.6f (vs expected %.6f)\n",
                 x_data[0]*w_data[1] + (b_data?b_data[0]:0), -0.539912f); }
        dump_cpu("after_ds_proj", x_data, out_seq * ch);
    }
    
    int conv_seq = seq / 2;
    
    // 2. in_proj
    ggml_tensor * x_conv = ggml_view_2d(ctx, x, ch, conv_seq, x->nb[1], 0);
    x_conv = ggml_cont(ctx, x_conv);
    ggml_tensor * h = ggml_mul_mat(ctx, pe.in_proj_w, x_conv);
    if (pe.in_proj_b) {
        ggml_tensor * ib = ggml_reshape_2d(ctx, pe.in_proj_b, PATCHENC_HIDDEN, 1);
        h = ggml_add(ctx, h, ib);
    }
    // Compute and dump
    { ggml_cgraph * cg = ggml_new_graph(ctx); ggml_build_forward_expand(cg, h);
      ggml_graph_compute_with_ctx(ctx, cg, 8); }
    dump_tensor("after_in_proj", h, PATCHENC_HIDDEN * conv_seq);
    
    int n_batch = 1, n_tokens = conv_seq;
    
    // 3. 24 layers with per-layer dump
    for (int i = 0; i < pe.n_layers; i++) {
        ggml_tensor * layer_in = h;
        
        // Pre-norm + attn
        ggml_tensor * hn = ggml_rms_norm(ctx, h, 1e-6f);
        if (pe.layers[i].attn_norm_w) hn = ggml_mul(ctx, hn, pe.layers[i].attn_norm_w);
        
        ggml_tensor * q = ggml_mul_mat(ctx, pe.layers[i].attn_q_weight, hn);
        ggml_tensor * k = ggml_mul_mat(ctx, pe.layers[i].attn_k_weight, hn);
        ggml_tensor * v = ggml_mul_mat(ctx, pe.layers[i].attn_v_weight, hn);
        q = ggml_cont(ctx, q); k = ggml_cont(ctx, k); v = ggml_cont(ctx, v);
        
        // qk_norm
        if (pe.layers[i].q_norm_w) { q = ggml_rms_norm(ctx, q, 1e-6f); q = ggml_mul(ctx, q, pe.layers[i].q_norm_w); }
        if (pe.layers[i].k_norm_w) { k = ggml_rms_norm(ctx, k, 1e-6f); k = ggml_mul(ctx, k, pe.layers[i].k_norm_w); }
        
        // RoPE
        int hidden = PATCHENC_HIDDEN, n_heads = PATCHENC_NUM_HEADS, head_dim = PATCHENC_HEAD_SIZE;
        q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
        k = ggml_reshape_3d(ctx, k, head_dim, n_heads, n_tokens);
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        { int * pd = (int*)tensor_data(pos); for(int s=0;s<n_tokens;s++) pd[s]=s; }
        q = ggml_rope(ctx, q, pos, head_dim, 0);
        k = ggml_rope(ctx, k, pos, head_dim, 0);
        
        // Attention
        ggml_tensor * Qf = ggml_reshape_2d(ctx, q, hidden, n_tokens);
        ggml_tensor * Kf = ggml_reshape_2d(ctx, k, hidden, n_tokens);
        ggml_tensor * Vf = ggml_reshape_2d(ctx, v, hidden, n_tokens);
        ggml_tensor * scores = ggml_mul_mat(ctx, Qf, Kf);
        scores = ggml_scale(ctx, scores, 1.0f/sqrtf((float)head_dim));
        scores = ggml_diag_mask_inf(ctx, scores, 1);
        scores = ggml_soft_max(ctx, scores);
        ggml_tensor * Vt = ggml_cont(ctx, ggml_permute(ctx, Vf, 1, 0, 2, 3));
        ggml_tensor * attn_out = ggml_mul_mat(ctx, scores, Vt);
        attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 1, 0, 2, 3));
        attn_out = ggml_mul_mat(ctx, pe.layers[i].attn_o_weight, attn_out);
        if (pe.layers[i].attn_o_bias) {
            ggml_tensor * ob = ggml_reshape_2d(ctx, pe.layers[i].attn_o_bias, hidden, 1);
            attn_out = ggml_add(ctx, attn_out, ob);
        }
        h = ggml_add(ctx, h, attn_out);
        
        // FFN
        hn = ggml_rms_norm(ctx, h, 1e-6f);
        if (pe.layers[i].ffn_norm_w) hn = ggml_mul(ctx, hn, pe.layers[i].ffn_norm_w);
        ggml_tensor * ffn = ggml_mul_mat(ctx, pe.layers[i].ffn_w1, hn);
        if (pe.layers[i].ffn_b1) {
            ggml_tensor * fb1 = ggml_reshape_2d(ctx, pe.layers[i].ffn_b1, PATCHENC_FFN_SIZE, 1);
            ffn = ggml_add(ctx, ffn, fb1);
        }
        ffn = ggml_silu(ctx, ffn);
        ffn = ggml_mul_mat(ctx, pe.layers[i].ffn_w2, ffn);
        if (pe.layers[i].ffn_b2) {
            ggml_tensor * fb2 = ggml_reshape_2d(ctx, pe.layers[i].ffn_b2, hidden, 1);
            ffn = ggml_add(ctx, ffn, fb2);
        }
        h = ggml_add(ctx, h, ffn);
        
        // Compute and dump
        { ggml_cgraph * cg = ggml_new_graph(ctx); ggml_build_forward_expand(cg, h);
          ggml_graph_compute_with_ctx(ctx, cg, 8); }
        char lname[64]; snprintf(lname,sizeof(lname),"layer_%d_out",i);
        dump_tensor(lname, h, hidden * n_tokens);
    }
    
    // Final norm
    h = ggml_rms_norm(ctx, h, 1e-6f);
    if (pe.final_norm_w) h = ggml_mul(ctx, h, pe.final_norm_w);
    { ggml_cgraph * cg = ggml_new_graph(ctx); ggml_build_forward_expand(cg, h);
      ggml_graph_compute_with_ctx(ctx, cg, 8); }
    dump_tensor("after_encoder", h, PATCHENC_HIDDEN * n_tokens);
    
    // Output projection
    ggml_tensor * h3d = ggml_reshape_3d(ctx, h, PATCHENC_HIDDEN, 2, 1);
    h3d = ggml_cont(ctx, ggml_permute(ctx, h3d, 0, 2, 1, 3));
    ggml_tensor * h_concat = ggml_reshape_2d(ctx, h3d, PATCHENC_HIDDEN * 2, 1);
    h_concat = ggml_cont(ctx, h_concat);
    ggml_tensor * out = ggml_mul_mat(ctx, pe.out_proj_w, h_concat);
    if (pe.out_proj_b) {
        ggml_tensor * ob = ggml_reshape_2d(ctx, pe.out_proj_b, 1536, 1);
        out = ggml_add(ctx, out, ob);
    }
    { ggml_cgraph * cg = ggml_new_graph(ctx); ggml_build_forward_expand(cg, out);
      ggml_graph_compute_with_ctx(ctx, cg, 8); }
    dump_tensor("final_output", out, 1536);
    
    ggml_free(ctx); ggml_free(w_ctx);
    return 0;
}
