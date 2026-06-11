// Definitive test: Python's exact hidden_proj → C++ DiT → C++ BigVGAN
// Euler update INSIDE step loop (correctly)
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "dit_manual.h"
#include "safetensors.h"
#include "bigvgan_cpp.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);

static void compute_cond(dit_model & m, float t_val, const float * spk_emb, float * cond) {
    int half=128; float se[256], h1[1024];
    for(int i=0;i<half;i++){float f=expf(-logf(10000.0f)*(float)i/(float)half);
        se[i]=cosf(t_val*f); se[half+i]=sinf(t_val*f);}
    {float*w1=tensor_data(m.t_embed_w1);float*b1=m.t_embed_b1?tensor_data(m.t_embed_b1):nullptr;
     for(int o=0;o<1024;o++){float s=b1?b1[o]:0.0f;for(int i=0;i<256;i++)s+=w1[o*256+i]*se[i];h1[o]=s/(1.0f+expf(-s));}}
    {float*w2=tensor_data(m.t_embed_w2);float*b2=m.t_embed_b2?tensor_data(m.t_embed_b2):nullptr;
     for(int o=0;o<1024;o++){float s=b2?b2[o]:0.0f;for(int i=0;i<1024;i++)s+=w2[o*1024+i]*h1[i];cond[o]=s;}}
}

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";
    
    SafeTensorsFile sf; sf.open(model_path);
    ggml_init_params wp = { .mem_size = 3ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model dit; load_dit_weights(sf, w_ctx, dit); sf.close();
    
    BigVGANDecoder bv;
    if (!bigvgan_load("models/vocoder_eff.safetensors", bv)) { fprintf(stderr, "FAILED BigVGAN\n"); return 1; }
    
    const int latent_dim = 128, patch_size = 4, nfe = 10;
    const float dt = 1.0f / nfe;
    int n_calls = 0, total_frames = 0;
    
    // Count Python dump files
    for (int call = 0; ; call++) {
        char fname[256]; snprintf(fname, sizeof(fname), "py_call%d_fmseq.bin", call);
        FILE * f = fopen(fname, "rb"); if (!f) break;
        fseek(f, 0, SEEK_END); if (ftell(f) == 0) { fclose(f); break; }
        fclose(f); n_calls = call + 1;
    }
    if (n_calls == 0) { fprintf(stderr, "No Python dump files\n"); return 1; }
    if (n_calls > 3) n_calls = 3;  // Limit for speed
    fprintf(stderr, "Found %d calls (using 3)\n", n_calls);
    
    float *all_latents = new float[n_calls * patch_size * latent_dim];
    
    for (int call = 0; call < n_calls; call++) {
        // Load Python fm_sequence and noise
        char fname[256]; snprintf(fname, sizeof(fname), "py_call%d_fmseq.bin", call);
        FILE * f = fopen(fname, "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        int fm_seq_len = sz / (DIT_HIDDEN_SIZE * sizeof(float));
        float * fm_seq = new float[fm_seq_len * DIT_HIDDEN_SIZE];
        fread(fm_seq, sizeof(float), fm_seq_len * DIT_HIDDEN_SIZE, f); fclose(f);
        
        int patch_flat = patch_size * latent_dim;
        float *z_t = new float[patch_flat], *v_t = new float[patch_flat];
        snprintf(fname, sizeof(fname), "py_noise_call%d.bin", call);
        f = fopen(fname, "rb");
        if (f) { fread(z_t, sizeof(float), patch_flat, f); fclose(f); }
        
        // Euler steps (UPDATE INSIDE LOOP)
        for (int step = 0; step < nfe; step++) {
            float t = (float)step * dt;
            int cond_seq = fm_seq_len + patch_size;
            int noise_pos = fm_seq_len;
            
            float *dx = new float[cond_seq * DIT_HIDDEN_SIZE]();
            memcpy(dx, fm_seq, fm_seq_len * DIT_HIDDEN_SIZE * sizeof(float));
            // coord_proj(noise)
            for (int i = 0; i < patch_size; i++) {
                float *op = dx + (noise_pos + i) * DIT_HIDDEN_SIZE;
                if (dit.coord_proj_w) {
                    float *cw = tensor_data(dit.coord_proj_w), *nv = z_t + i * latent_dim;
                    for (int j = 0; j < DIT_HIDDEN_SIZE; j++) { float s=0; for(int k=0;k<latent_dim;k++)s+=nv[k]*cw[j*latent_dim+k]; op[j]=s; }
                }
            }
            for (int p = 0; p < cond_seq; p++) { float *pos=dx+p*DIT_HIDDEN_SIZE, r=0; for(int j=0;j<DIT_HIDDEN_SIZE;j++)r+=pos[j]*pos[j];
                r=sqrtf(r/DIT_HIDDEN_SIZE); if(r>10.0f){float s=10.0f/r;for(int j=0;j<DIT_HIDDEN_SIZE;j++)pos[j]*=s;} }
            
            float cond[1024]; compute_cond(dit, t, nullptr, cond);
            float *h = new float[cond_seq*DIT_HIDDEN_SIZE], *bo = new float[cond_seq*DIT_HIDDEN_SIZE];
            {float*iw=tensor_data(dit.input_layer_w);float*ib=dit.input_layer_b?tensor_data(dit.input_layer_b):nullptr;
             for(int ti=0;ti<cond_seq;ti++) manual_linear(h+ti*DIT_HIDDEN_SIZE,dx+ti*DIT_HIDDEN_SIZE,iw,ib,DIT_HIDDEN_SIZE,DIT_HIDDEN_SIZE);}
            for(int i=0;i<dit.n_layers;i++){manual_dit_block(h,cond,dit.layers[i],bo,cond_seq);float*tmp=h;h=bo;bo=tmp;}
            float cs2[1024]; for(int i=0;i<1024;i++) cs2[i]=cond[i]/(1.0f+expf(-cond[i]));
            float mod_raw2[2*DIT_HIDDEN_SIZE];
            {float*aw=tensor_data(dit.out_adaln_w);float*ab=dit.out_adaln_b?tensor_data(dit.out_adaln_b):nullptr;
             for(int o=0;o<2*DIT_HIDDEN_SIZE;o++){float s=ab?ab[o]:0;for(int i=0;i<DIT_HIDDEN_SIZE;i++)s+=aw[o*DIT_HIDDEN_SIZE+i]*cs2[i];mod_raw2[o]=s;}}
            float*dit_shift=mod_raw2,*dit_scale=mod_raw2+DIT_HIDDEN_SIZE;
            float*dit_ln=new float[cond_seq*DIT_HIDDEN_SIZE];
            for(int ti=0;ti<cond_seq;ti++){manual_layernorm(dit_ln+ti*DIT_HIDDEN_SIZE,h+ti*DIT_HIDDEN_SIZE,DIT_HIDDEN_SIZE);
                for(int j=0;j<DIT_HIDDEN_SIZE;j++)dit_ln[ti*DIT_HIDDEN_SIZE+j]=dit_ln[ti*DIT_HIDDEN_SIZE+j]*(1.0f+dit_scale[j])+dit_shift[j];}
            float*ow=tensor_data(dit.out_proj_w);float*ob=dit.out_proj_b?tensor_data(dit.out_proj_b):nullptr;
            float *out_data = new float[cond_seq*latent_dim];
            for(int ti=0;ti<cond_seq;ti++) manual_linear(out_data+ti*latent_dim,dit_ln+ti*DIT_HIDDEN_SIZE,ow,ob,DIT_HIDDEN_SIZE,latent_dim);
            for (int p=0;p<patch_size;p++){int t_idx=noise_pos+p; for(int c=0;c<latent_dim;c++)v_t[p*latent_dim+c]=out_data[t_idx*latent_dim+c];}
            delete[] dx; delete[] out_data; delete[] h; delete[] bo; delete[] dit_ln;
            
            // EULER UPDATE INSIDE LOOP (this was the bug before!)
            for (int i = 0; i < patch_flat; i++) z_t[i] += v_t[i] * dt;
        }
        
        // Save latents for comparison
        snprintf(fname, sizeof(fname), "cpp_v2_call%d_latents.bin", call);
        { FILE * lf = fopen(fname, "wb"); if(lf){fwrite(z_t,sizeof(float),patch_flat,lf);fclose(lf);} }
        
        memcpy(all_latents+total_frames*latent_dim, z_t, patch_size*latent_dim*sizeof(float));
        total_frames += patch_size;
        { float r=0; for(int i=0;i<patch_flat;i++) r+=z_t[i]*z_t[i];
          fprintf(stderr,"Call %d: latent RMS=%.4f (seq=%d)\n", call, sqrtf(r/patch_flat), fm_seq_len+patch_size); }
        
        delete[] z_t; delete[] v_t; delete[] fm_seq;
    }
    
    // BigVGAN decode (with VAE standardization)
    { float r=0; for(int i=0;i<total_frames*latent_dim;i++) r+=all_latents[i]*all_latents[i];
      fprintf(stderr,"Latents pre-std RMS=%.4f\n", sqrtf(r/(total_frames*latent_dim))); }
    // VAE standardization
    static const float VMEAN[128] = {-0.0018,1.5587,0.2704,-0.6201,0.6770,0.0476,-1.0319,0.4992,0.6062,-0.1117,0.1011,1.9314,-0.3252,1.6055,-0.7193,-0.3595,1.7855,-0.4199,1.7570,1.0285,-0.3500,1.1654,-1.7491,-0.7094,-0.5514,-2.2958,0.9232,-0.2191,-0.8963,-2.5472,-3.2895,0.3434,-0.3056,1.2524,1.2189,0.9421,-3.2393,-0.1867,-0.5583,0.5986,0.4958,0.8181,1.2283,0.3208,-0.0927,-0.5480,-0.3843,1.5058,-1.1151,-0.4964,-0.8426,0.6010,-0.3794,-1.3442,1.6066,-0.7866,-0.7095,0.2941,-0.2754,0.5637,1.5513,2.1656,0.4843,0.4405,-0.8961,0.1787,-1.7293,-0.7062,-0.8769,-0.1251,-0.6801,0.3431,-0.6298,0.2136,0.2846,-0.9208,2.1376,-0.2581,-1.8387,-0.6564,-2.3242,1.2945,1.1961,-0.3600,-0.3377,0.8106,-0.2576,1.2125,-2.3581,1.1030,-2.5128,-0.4035,-0.7615,1.6613,-0.4468,-0.7887,-1.2168,1.2327,-0.5016,-0.3032,0.1336,-2.3773,-0.9109,-6.6330,-1.2273,-1.4531,-0.2262,-1.4044,1.7953,-0.9349,0.4821,-2.9635,2.5297,-0.9634,-0.1377,0.2467,-1.2026,2.2112,-0.3299,4.8154,0.6731,-1.4437,-1.1465,-0.0825,-0.0948,-1.3719,0.3122,-0.1676};
    static const float VSTD[128] = {3.6477,2.0494,2.0477,2.3114,1.6807,2.7022,2.9756,1.6971,1.8174,2.3360,1.9619,2.1530,2.2949,2.7127,1.6238,1.9152,3.0338,1.5830,3.2447,1.5701,2.6532,2.5800,2.1421,3.1803,1.9691,5.8602,1.7696,2.0176,2.5675,3.1446,2.3140,3.1461,1.6386,1.6639,5.4171,3.3226,3.7026,1.9677,1.9383,1.4299,1.7438,1.8232,1.5562,3.7779,2.2670,1.9966,1.8933,2.9376,1.5818,1.7817,1.2738,1.2641,1.8378,2.5080,2.2691,2.4595,2.2339,2.9236,2.1710,1.9006,1.9267,2.5662,2.3672,2.6802,1.4845,1.5066,2.4320,1.5267,2.9641,1.9826,1.1071,2.0246,1.4723,1.2689,2.5800,2.5754,2.5913,1.3481,1.5047,1.6610,1.9842,1.9742,1.8364,1.4604,1.3824,1.4949,3.4064,1.5842,1.4575,2.2500,2.2421,1.8668,1.4647,1.4465,1.7577,1.7282,2.0829,1.6112,2.7190,1.4525,1.6201,2.5962,1.7213,22.1080,1.0896,1.7761,2.9203,2.0389,3.5477,2.3818,1.5096,7.3880,1.3896,2.7805,2.0932,1.9355,2.2740,1.3479,1.6040,1.9349,2.5265,2.7129,1.8058,1.3762,1.4585,2.8542,1.5449,2.7214};
    float cm[128]={0}, cs[128]={0};
    for(int f=0;f<total_frames;f++) for(int c=0;c<latent_dim;c++) cm[c]+=all_latents[f*latent_dim+c];
    for(int c=0;c<latent_dim;c++) cm[c]/=total_frames;
    for(int f=0;f<total_frames;f++) for(int c=0;c<latent_dim;c++){float d=all_latents[f*latent_dim+c]-cm[c];cs[c]+=d*d;}
    for(int c=0;c<latent_dim;c++) cs[c]=sqrtf(cs[c]/total_frames+1e-8f);
    for(int f=0;f<total_frames;f++) for(int c=0;c<latent_dim;c++)
        all_latents[f*latent_dim+c]=(all_latents[f*latent_dim+c]-cm[c])/cs[c]*VSTD[c]+VMEAN[c];
    { float r=0; for(int i=0;i<total_frames*latent_dim;i++) r+=all_latents[i]*all_latents[i];
      fprintf(stderr,"Latents post-std RMS=%.4f\n", sqrtf(r/(total_frames*latent_dim))); }
    
    int n_samples = total_frames * 1920;
    float *audio = new float[n_samples];
    bigvgan_decode(bv, all_latents, total_frames, audio, &n_samples);
    
    FILE*wf=fopen("output_pyctx_v2.wav","wb");
    if(wf){int ds=n_samples*2,fs=36+ds;fwrite("RIFF",1,4,wf);fwrite(&fs,4,1,wf);fwrite("WAVE",1,4,wf);
        fwrite("fmt ",1,4,wf);int fz=16;fwrite(&fz,4,1,wf);short af=1,nc=1;fwrite(&af,2,1,wf);fwrite(&nc,2,1,wf);
        int sr=48000;fwrite(&sr,4,1,wf);int br=sr*2;fwrite(&br,4,1,wf);short ba=2,bp=16;fwrite(&ba,2,1,wf);fwrite(&bp,2,1,wf);
        fwrite("data",1,4,wf);fwrite(&ds,4,1,wf);
        for(int i=0;i<n_samples;i++){float s=audio[i]*32767.0f;if(s>32767)s=32767;if(s<-32768)s=-32768;short si=(short)s;fwrite(&si,2,1,wf);}
        fclose(wf);fprintf(stderr,"Written output_pyctx_v2.wav: %d samples\n",n_samples);}
    
    delete[] all_latents; delete[] audio; bigvgan_free(bv); ggml_free(w_ctx);
    return 0;
}
