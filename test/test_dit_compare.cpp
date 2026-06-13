// Quick DiT comparison test: C++ vs Python on same input
#include "dots_tts.h"
#include "dit.h"
#include "dit_manual.h"
#include "safetensors.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <cmath>

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);

static void compute_cond(dit_model & m, float t_val, const float * spk_emb, float speaker_scale, float * cond) {
    int half=128; float se[256], h1[1024];
    for(int i=0;i<half;i++){float f=expf(-logf(10000.0f)*(float)i/(float)half);
        se[i]=cosf(t_val*f); se[half+i]=sinf(t_val*f);}
    {float*w1=tensor_data(m.t_embed_w1);float*b1=m.t_embed_b1?tensor_data(m.t_embed_b1):nullptr;
     for(int o=0;o<1024;o++){float s=b1?b1[o]:0.0f;for(int i=0;i<256;i++)s+=w1[o*256+i]*se[i];h1[o]=s/(1.0f+expf(-s));}}
    {float*w2=tensor_data(m.t_embed_w2);float*b2=m.t_embed_b2?tensor_data(m.t_embed_b2):nullptr;
     for(int o=0;o<1024;o++){float s=b2?b2[o]:0.0f;for(int i=0;i<1024;i++)s+=w2[o*1024+i]*h1[i];cond[o]=s;}}
    if(spk_emb){for(int o=0;o<1024;o++)cond[o]=cond[o];} // no-op for speaker
}

static float compute_rms(const float * data, int n) {
    double r = 0; for (int i = 0; i < n; i++) r += (double)data[i]*data[i];
    return (float)sqrt(r / n);
}

static float compute_corr(const float * a, const float * b, int n) {
    double sx=0,sy=0,sxy=0,sx2=0,sy2=0;
    for(int i=0;i<n;i++){sx+=a[i];sy+=b[i];sxy+=(double)a[i]*b[i];sx2+=(double)a[i]*a[i];sy2+=(double)b[i]*b[i];}
    return (float)((n*sxy-sx*sy)/sqrt((n*sx2-sx*sx)*(n*sy2-sy*sy)));
}

int main() {
    setbuf(stdout, NULL);
    
    const char * sf_path = "models/model.safetensors";
    { FILE * f = fopen(sf_path, "rb"); if (!f) { sf_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/e3520f75254d0020a0406db31c51a79d00d22d55/model.safetensors"; f = fopen(sf_path, "rb"); if (f) fclose(f); } }
    
    ggml_init_params wp = { .mem_size = 3ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    
    SafeTensorsFile sf; sf.open(sf_path);
    dit_model dit; load_dit_weights(sf, w_ctx, dit); sf.close();
    
    const int total_len = 5, hidden = 1024, latent_dim = 128, patch_size = 4;
    float * dit_input = new float[total_len * hidden];
    float * py_vel = new float[patch_size * latent_dim];
    bool * mask = new bool[total_len * total_len];
    float * pos_ids = new float[total_len];
    
    { FILE * f = fopen("debug_v4/dit_input_test.bin", "rb"); fread(dit_input, sizeof(float), total_len * hidden, f); fclose(f); }
    { FILE * f = fopen("debug_v4/py_vel_test_v2.bin", "rb"); fread(py_vel, sizeof(float), patch_size * latent_dim, f); fclose(f); }
    { FILE * f = fopen("debug_v4/mask_test.bin", "rb"); float * tmp = new float[total_len*total_len]; fread(tmp, sizeof(float), total_len*total_len, f); fclose(f); for(int i=0;i<total_len*total_len;i++) mask[i]=tmp[i]>0.5f; delete[] tmp; }
    { FILE * f = fopen("debug_v4/pos_ids_test.bin", "rb"); fread(pos_ids, sizeof(float), total_len, f); fclose(f); }
    
    printf("Input loaded. pos0 RMS=%.4f  Py vel RMS=%.4f\n", compute_rms(dit_input, hidden), compute_rms(py_vel, patch_size*latent_dim));
    
    // Compute cond using time embedder
    float cond[1024]; compute_cond(dit, 0.0f, nullptr, 1.5f, cond);
    printf("C++ cond RMS=%.4f\n", compute_rms(cond, 1024));
    
    // C++ DiT forward
    float * h_seq = new float[total_len * hidden];
    {float * iw = tensor_data(dit.input_layer_w); float * ib = dit.input_layer_b ? tensor_data(dit.input_layer_b) : nullptr;
     for (int ti = 0; ti < total_len; ti++) manual_linear(h_seq + ti * hidden, dit_input + ti * hidden, iw, ib, hidden, hidden);}
    
    float * bo_seq = new float[total_len * hidden];
    for (int i = 0; i < dit.n_layers; i++) {
        manual_dit_block(h_seq, cond, dit.layers[i], bo_seq, total_len, mask, pos_ids);
        float * tmp = h_seq; h_seq = bo_seq; bo_seq = tmp;
    }
    
    float cs2[1024]; for (int i = 0; i < hidden; i++) cs2[i] = cond[i] / (1.0f + expf(-cond[i]));
    float mod_raw[2 * hidden];
    {float * aw = tensor_data(dit.out_adaln_w); float * ab = dit.out_adaln_b ? tensor_data(dit.out_adaln_b) : nullptr;
     for (int o = 0; o < 2 * hidden; o++) { float s = ab ? ab[o] : 0.0f; for (int i = 0; i < hidden; i++) s += aw[o * hidden + i] * cs2[i]; mod_raw[o] = s; }}
    float * dit_shift = mod_raw, * dit_scale = mod_raw + hidden;
    float * dit_ln = new float[total_len * hidden];
    for (int ti = 0; ti < total_len; ti++) {
        manual_layernorm(dit_ln + ti * hidden, h_seq + ti * hidden, hidden);
        for (int j = 0; j < hidden; j++) dit_ln[ti * hidden + j] = dit_ln[ti * hidden + j] * (1.0f + dit_scale[j]) + dit_shift[j];
    }
    {float * ow = tensor_data(dit.out_proj_w); float * ob = dit.out_proj_b ? tensor_data(dit.out_proj_b) : nullptr;
     float * out_all = new float[total_len * latent_dim];
     for (int ti = 0; ti < total_len; ti++) manual_linear(out_all + ti * latent_dim, dit_ln + ti * hidden, ow, ob, hidden, latent_dim);
    
     float cpp_vel[512];
     for (int p = 0; p < 4; p++) for (int c = 0; c < 128; c++) cpp_vel[p*128+c] = out_all[(1+p)*128 + c];
     delete[] out_all;
    
     float cpp_rms = compute_rms(cpp_vel, 512);
     float corr = compute_corr(cpp_vel, py_vel, 512);
     float diff_max = 0; for(int i=0;i<512;i++){float d=fabsf(cpp_vel[i]-py_vel[i]);if(d>diff_max)diff_max=d;}
     
     printf("C++ vel RMS=%.4f  Corr=%.6f  Max|diff|=%.6f\n", cpp_rms, corr, diff_max);
     printf("First 4: Py=[%.6f,%.6f,%.6f,%.6f] Cpp=[%.6f,%.6f,%.6f,%.6f]\n",
            py_vel[0],py_vel[1],py_vel[2],py_vel[3],cpp_vel[0],cpp_vel[1],cpp_vel[2],cpp_vel[3]);
    }
    
    delete[] dit_input; delete[] py_vel; delete[] mask; delete[] pos_ids;
    delete[] h_seq; delete[] bo_seq; delete[] dit_ln;
    ggml_free(w_ctx);
    return 0;
}
