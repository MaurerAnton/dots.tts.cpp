// Test: DiT velocity at different t values using manual_dit_block (CORRECT)
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

void compute_cond(dit_model & m, float t_val, const float * spk_emb, float * cond) {
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

// Same as e2e_pipeline's flow matching code
void dit_forward_correct(dit_model & model, const float * x, int n_tokens, float t_val,
    const float * speaker_emb, float * out) {
    int hidden = DIT_HIDDEN_SIZE;
    float cond[1024]; compute_cond(model, t_val, speaker_emb, cond);
    float *h = new float[n_tokens*hidden];
    {float*iw=tensor_data(model.input_layer_w);float*ib=model.input_layer_b?tensor_data(model.input_layer_b):nullptr;
     for(int t=0;t<n_tokens;t++) manual_linear(h+t*hidden,x+t*hidden,iw,ib,hidden,hidden);}
    float *bo = new float[n_tokens*hidden];
    for(int i=0;i<model.n_layers;i++){manual_dit_block(h,cond,model.layers[i],bo,n_tokens);float*tmp=h;h=bo;bo=tmp;}
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
    delete[] h;delete[] bo;delete[] ln;
}

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";
    SafeTensorsFile sf; sf.open(model_path);
    ggml_init_params wp = { .mem_size = 2ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model m; load_dit_weights(sf, w_ctx, m); sf.close();

    int n_tokens = 8, hidden = DIT_HIDDEN_SIZE;
    float * x_in = new float[n_tokens * hidden];
    FILE * f_in = fopen("debug/cpp_dit_input.bin", "rb");
    fread(x_in, sizeof(float), n_tokens * hidden, f_in); fclose(f_in);

    float spk_zero[512] = {0};
    float out[8 * VAE_LATENT_DIM];

    float t_vals[] = {0.0f, 0.1f, 0.5f, 0.9f};
    for (int ti = 0; ti < 4; ti++) {
        float t = t_vals[ti];
        dit_forward_correct(m, x_in, n_tokens, t, spk_zero, out);
        float rms = 0;
        for (int i = 0; i < 8 * VAE_LATENT_DIM; i++) rms += out[i] * out[i];
        rms = sqrtf(rms / (8 * VAE_LATENT_DIM));
        printf("t=%.1f: RMS=%.4f, first5=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
               t, rms, out[0], out[1], out[2], out[3], out[4]);
    }

    delete[] x_in;
    ggml_free(w_ctx);
    return 0;
}
