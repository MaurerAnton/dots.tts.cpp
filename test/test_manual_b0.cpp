// Test manual block 0 vs Python
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "dit_manual.h"
#include "safetensors.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cmath>

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";
    SafeTensorsFile sf; sf.open(model_path);
    ggml_init_params wp = { .mem_size = 2ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model m; load_dit_weights(sf, w_ctx, m); sf.close();

    auto load_bin = [](const char * name) -> float* {
        char path[256]; snprintf(path, sizeof(path), "debug/py_%s.bin", name);
        FILE * f = fopen(path, "rb"); if(!f) return nullptr;
        fseek(f,0,SEEK_END); int n=ftell(f)/4; fseek(f,0,SEEK_SET);
        float * d = new float[n]; fread(d,sizeof(float),n,f); fclose(f); return d; };

    float * py_x_in = load_bin("b0_x_in");
    float * py_out = load_bin("b0_out");
    if (!py_x_in || !py_out) { fprintf(stderr, "Run tools/dump_dit_py.py first\n"); return 1; }

    // Compute cond manually (t=0, speaker=zeros)
    float t_emb[1024];
    {   float t_val=0.0f; int half=128; float se[256];
        for(int i=0;i<half;i++){ float f=expf(-logf(10000.0f)*(float)i/(float)half);
            se[i]=cosf(t_val*f); se[half+i]=sinf(t_val*f); }
        float * w1=tensor_data(m.t_embed_w1); float * b1=m.t_embed_b1?tensor_data(m.t_embed_b1):nullptr;
        float h1[1024]; for(int o=0;o<1024;o++){ float s=b1?b1[o]:0.0f; for(int i=0;i<256;i++) s+=w1[o*256+i]*se[i]; h1[o]=s/(1.0f+expf(-s)); }
        float * w2=tensor_data(m.t_embed_w2); float * b2=m.t_embed_b2?tensor_data(m.t_embed_b2):nullptr;
        for(int o=0;o<1024;o++){ float s=b2?b2[o]:0.0f; for(int i=0;i<1024;i++) s+=w2[o*1024+i]*h1[i]; t_emb[o]=s; } }
    float spk_vals[1024];
    {   float temp[1024]; float * sw1=tensor_data(m.spk_proj_w1); float * sb1=m.spk_proj_b1?tensor_data(m.spk_proj_b1):nullptr;
        for(int o=0;o<1024;o++){ float s=sb1?sb1[o]:0.0f; for(int i=0;i<512;i++) s+=sw1[o*512+i]*0.0f; temp[o]=s; }
        float mean=0; for(int i=0;i<1024;i++) mean+=temp[i]; mean/=1024;
        float var=0; for(int i=0;i<1024;i++){ float d=temp[i]-mean; var+=d*d; } var=var/1024+1e-5f;
        float istd=1.0f/sqrtf(var); float * lw=m.spk_ln_w?tensor_data(m.spk_ln_w):nullptr;
        float * lb=m.spk_ln_b?tensor_data(m.spk_ln_b):nullptr;
        for(int i=0;i<1024;i++){ float x=(temp[i]-mean)*istd; if(lw)x*=lw[i]; if(lb)x+=lb[i]; spk_vals[i]=x; } }
    float cond[1024]; for(int i=0;i<1024;i++) cond[i]=t_emb[i]+spk_vals[i];

    // Run manual block 0
    float cpp_out[8192];
    manual_dit_block(py_x_in, cond, m.layers[0], cpp_out, 8);

    // Verify LayerNorm manually
    {
        const float * tok0 = py_x_in;
        // Test manual_layernorm directly
        float dir_normed[3]; manual_layernorm(dir_normed, tok0, 1024);
        float mean = 0; for (int i = 0; i < 1024; i++) mean += tok0[i]; mean /= 1024;
        float var = 0; for (int i = 0; i < 1024; i++) { float d = tok0[i] - mean; var += d * d; }
        float inv_std = 1.0f / sqrtf(var / 1024 + 1e-5f);
        float normed[3];
        for (int i = 0; i < 3; i++) normed[i] = (tok0[i] - mean) * inv_std;
        fprintf(stderr, "  test_normed: first3=[%.4f,%.4f,%.4f]\n", normed[0], normed[1], normed[2]);
        fprintf(stderr, "  dir_normed: first3=[%.4f,%.4f,%.4f]\n", dir_normed[0], dir_normed[1], dir_normed[2]);
    }

    // Compare
    float r_cpp=0, r_py=0, md=0;
    for(int i=0;i<8192;i++){ r_cpp+=cpp_out[i]*cpp_out[i]; r_py+=py_out[i]*py_out[i];
        float d=fabsf(cpp_out[i]-py_out[i]); if(d>md) md=d; }
    printf("Manual block 0 vs Python:\n  C++ RMS=%.4f  Py RMS=%.4f  max_diff=%.6f\n",
        sqrtf(r_cpp/8192), sqrtf(r_py/8192), md);
    printf("  C++ first5: "); for(int i=0;i<5;i++) printf("%.4f ", cpp_out[i]);
    printf("\n  Py  first5: "); for(int i=0;i<5;i++) printf("%.4f ", py_out[i]); printf("\n");
    printf("  %s\n", md < 0.01 ? "MATCH!" : "DIFFERS");

    delete[] py_x_in; delete[] py_out; ggml_free(w_ctx);
    return 0;
}
