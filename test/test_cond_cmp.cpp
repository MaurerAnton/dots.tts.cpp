// Quick test: compare cond from compute_cond vs test_full_dit inline
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

    float cond1[1024], cond2[1024];
    float spk_zero[512] = {0};

    // Method 1: test_full_dit inline (verified byte-perfect with Python)
    {
        float t_val = 0.0f;
        int half = 128; float se[256], h1[1024];
        for (int i = 0; i < half; i++) {
            float f = expf(-logf(10000.0f) * (float)i / (float)half);
            se[i] = cosf(t_val * f); se[half + i] = sinf(t_val * f);
        }
        float * w1 = tensor_data(m.t_embed_w1);
        float * b1 = m.t_embed_b1 ? tensor_data(m.t_embed_b1) : nullptr;
        for (int o = 0; o < 1024; o++) {
            float s = b1 ? b1[o] : 0.0f;
            for (int i = 0; i < 256; i++) s += w1[o*256 + i] * se[i];
            h1[o] = s / (1.0f + expf(-s));
        }
        float * w2 = tensor_data(m.t_embed_w2);
        float * b2 = m.t_embed_b2 ? tensor_data(m.t_embed_b2) : nullptr;
        for (int o = 0; o < 1024; o++) {
            float s = b2 ? b2[o] : 0.0f;
            for (int i = 0; i < 1024; i++) s += w2[o*1024 + i] * h1[i];
            cond1[o] = s;
        }
        float temp[1024];
        float * sw1 = tensor_data(m.spk_proj_w1);
        float * sb1 = m.spk_proj_b1 ? tensor_data(m.spk_proj_b1) : nullptr;
        for (int o = 0; o < 1024; o++) {
            float s = sb1 ? sb1[o] : 0.0f;
            for (int i = 0; i < 512; i++) s += sw1[o*512 + i] * spk_zero[i];
            temp[o] = s;
        }
        float mean = 0;
        for (int i = 0; i < 1024; i++) mean += temp[i]; mean /= 1024;
        float var = 0;
        for (int i = 0; i < 1024; i++) { float d = temp[i] - mean; var += d * d; }
        var = var / 1024 + 1e-5f;
        float istd = 1.0f / sqrtf(var);
        float * lw = m.spk_ln_w ? tensor_data(m.spk_ln_w) : nullptr;
        float * lb = m.spk_ln_b ? tensor_data(m.spk_ln_b) : nullptr;
        for (int i = 0; i < 1024; i++) {
            float x = (temp[i] - mean) * istd;
            if (lw) x *= lw[i]; if (lb) x += lb[i];
            cond1[i] += x;
        }
    }

    // Method 2: compute_cond from dit_forward.cpp (declared static, need to duplicate)
    // Actually, compute_cond is static in dit_forward.cpp, we can't call it directly
    // Let's just compare with what dit_forward_raw produces
    // Instead, let's test: does compute_cond produce different output?
    // We'll replicate compute_cond here exactly
    
    {
        float t_val = 0.0f;
        int half=128; float se[256], h1[1024];
        for(int i=0;i<half;i++){float f=expf(-logf(10000.0f)*(float)i/(float)half);
            se[i]=cosf(t_val*f); se[half+i]=sinf(t_val*f);}
        {float*w1=tensor_data(m.t_embed_w1);float*b1=m.t_embed_b1?tensor_data(m.t_embed_b1):nullptr;
         for(int o=0;o<1024;o++){float s=b1?b1[o]:0.0f;for(int i=0;i<256;i++)s+=w1[o*256+i]*se[i];h1[o]=s/(1.0f+expf(-s));}}
        {float*w2=tensor_data(m.t_embed_w2);float*b2=m.t_embed_b2?tensor_data(m.t_embed_b2):nullptr;
         for(int o=0;o<1024;o++){float s=b2?b2[o]:0.0f;for(int i=0;i<1024;i++)s+=w2[o*1024+i]*h1[i];cond2[o]=s;}}
        if(spk_zero){float t[1024],*sw1=tensor_data(m.spk_proj_w1),*sb1=m.spk_proj_b1?tensor_data(m.spk_proj_b1):nullptr;
         for(int o=0;o<1024;o++){float s=sb1?sb1[o]:0.0f;for(int i=0;i<512;i++)s+=sw1[o*512+i]*spk_zero[i];t[o]=s;}
         float mean=0;for(int i=0;i<1024;i++)mean+=t[i];mean/=1024;
         float var=0;for(int i=0;i<1024;i++){float d=t[i]-mean;var+=d*d;}var=var/1024+1e-5f;
         float istd=1.0f/sqrtf(var);float*lw=m.spk_ln_w?tensor_data(m.spk_ln_w):nullptr,*lb=m.spk_ln_b?tensor_data(m.spk_ln_b):nullptr;
         for(int i=0;i<1024;i++){float x=(t[i]-mean)*istd;if(lw)x*=lw[i];if(lb)x+=lb[i];cond2[i]+=x;}}
    }

    // Compare
    float max_diff = 0;
    for (int i = 0; i < 1024; i++) {
        float d = fabsf(cond1[i] - cond2[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("Cond max_diff: %.6f\n", max_diff);
    float r1=0,r2=0; for(int i=0;i<1024;i++){r1+=cond1[i]*cond1[i];r2+=cond2[i]*cond2[i];}
    printf("Cond1 RMS: %.4f, Cond2 RMS: %.4f\n", sqrtf(r1/1024), sqrtf(r2/1024));

    ggml_free(w_ctx);
    return 0;
}
