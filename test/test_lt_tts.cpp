// Full TTS pipeline using LibTorch traced DiT — byte-identical to Python
#include <torch/torch.h>
#include <torch/script.h>
#include "dots_tts.h"
#include "dit.h"
#include "dit_manual.h"
#include "audiovae.h"
#include "bigvgan_cpp.h"
#include "patchenc.h"
#include "safetensors.h"
#include "gpt2_bpe_tokenizer.h"
#include "llm_manual.h"
#include "noise_data_pipeline.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <ctime>

static double now_ms() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1000.0 + ts.tv_nsec/1000000.0; }
bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);
bool load_patchenc_weights(SafeTensorsFile & sf, ggml_context * w_ctx, patch_encoder & enc);

// Denormalize latents
static const float VAE_STD[128] = {3.0316,2.8055,1.7751,1.7465,2.0646,1.9869,2.2711,2.0302,1.8684,2.3117,2.0107,1.9570,2.8696,2.3088,2.0507,2.1352,2.6940,1.7395,1.9994,1.8537,2.0860,2.1602,1.9088,2.6236,1.9415,3.6039,1.7381,1.8368,2.7282,2.0870,2.4967,2.4496,1.6383,1.6716,4.6338,2.4076,3.9610,1.8117,1.8016,1.8020,2.0124,1.7533,1.9128,2.2415,2.0514,1.9496,1.8469,2.1641,1.9519,1.9785,1.7530,1.6008,1.9471,2.6634,2.2622,2.4504,1.9333,2.8137,2.0679,2.1296,1.8197,2.4677,2.2560,2.2627,1.9336,1.6912,2.1193,1.7280,3.0617,1.5928,1.7773,1.8976,1.6118,1.5717,2.4272,3.1342,2.3934,1.8039,1.8262,1.7740,3.5085,1.9350,2.2693,1.8149,1.5992,1.8857,2.5523,2.2707,1.8558,1.6974,1.9261,1.8115,1.5260,1.8547,1.4196,1.7291,1.6253,2.0238,2.5385,2.1883,1.6951,2.5633,2.0287,13.0302,2.1418,1.6957,4.7318,2.3544,2.6665,2.8945,1.7005,6.3084,2.2875,2.7594,2.3732,1.7797,1.9751,2.6090,1.6278,2.5487,2.0741,2.5441,2.0957,1.6018,1.4397,3.2050,1.8041,2.3601};
static const float VAE_MEAN[128] = {-0.9025,-0.6183,-0.0263,-0.0685,-0.1783,-0.5570,-0.6110,-0.3879,-0.3249,-0.2662,0.3478,0.0372,-0.5750,0.0344,-0.5378,0.2272,-0.2407,0.2857,-0.4126,0.0592,-0.3421,-0.5783,0.0266,-0.0334,-0.0829,-0.3520,0.1200,-0.4391,0.3389,-0.4224,-0.3515,-0.0558,0.1107,-0.0028,0.6533,-0.0096,0.3998,0.0979,0.0710,0.1544,-0.1871,0.1712,0.2767,-0.3803,-0.2228,-0.3248,0.0888,0.0382,-0.2751,-0.1028,-0.3058,0.2343,0.0244,-0.2566,-0.0230,0.7069,-0.4051,0.5665,-0.1977,-0.4928,0.1987,-0.3744,0.2669,-0.5638,-0.0391,0.1564,-0.1035,-0.4483,0.6975,0.2761,-0.2020,0.2640,0.0682,0.1176,0.7706,-1.3600,0.0722,-0.1375,0.0337,0.1764,-0.2927,0.3615,-0.1335,0.3452,0.0590,-0.2773,0.0653,0.3692,-0.3961,-0.0325,0.1038,-0.2431,-0.0402,0.3810,-0.0239,-0.3426,-0.0924,-0.2611,0.1166,0.4047,0.1759,-0.6634,-0.0081,2.0526,0.0439,-0.1364,0.1873,-0.0263,-0.7545,0.5526,-0.0752,-0.3717,0.1637,0.7791,0.7361,-0.3620,-0.3784,0.4654,-0.2694,-0.2115,0.3204,0.2995,-0.7347,0.4701,0.1141,0.6925,0.7140,-0.0032};

static void denormalize_latents(float * z, int n_frames, int latent_dim) {
    for (int f = 0; f < n_frames; f++)
        for (int c = 0; c < latent_dim; c++)
            z[f * latent_dim + c] = z[f * latent_dim + c] * VAE_STD[c] + VAE_MEAN[c];
}

int main(int argc, char ** argv) {
    setbuf(stdout, NULL);
    const char * text = argc > 1 ? argv[1] : "Hello world";
    const char * out_path = argc > 2 ? argv[2] : "output_lt.wav";
    int n_calls = argc > 3 ? atoi(argv[3]) : 7;
    
    printf("LibTorch TTS: '%s' → %s (%d calls)\n", text, out_path, n_calls);
    double t0 = now_ms();
    
    // Load traced DiT
    auto dit = torch::jit::load("models/dit_traced.pt");
    printf("  Traced DiT loaded (%d ms)\n", (int)(now_ms()-t0));
    
    // Load DiT weights for coord_proj etc (non-DiT ops)
    const char * model_dir = "models";
    SafeTensorsFile sf; sf.open((std::string(model_dir) + "/model.safetensors").c_str());
    ggml_init_params wp = { .mem_size = 3ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model dit_aux; load_dit_weights(sf, w_ctx, dit_aux);
    
    // Load PatchEncoder
    SafeTensorsFile sf2; sf2.open((std::string(model_dir) + "/model.safetensors").c_str());
    ggml_context * w_ctx_pe = ggml_init(wp);
    patch_encoder pe; load_patchenc_weights(sf2, w_ctx_pe, pe);
    
    // Load LLM + tokenizer
    GPT2BPETokenizer bpe; bpe.load(model_dir);
    std::string tmpl = "[文本]" + std::string(text) + "[文本对应语音]<|audio_gen_start|>";
    auto tok_vec = bpe.encode(tmpl.c_str());
    int n_tok = (int)tok_vec.size();
    int32_t * tokens = tok_vec.data();
    
    SafeTensorsFile sf3; sf3.open((std::string(model_dir) + "/model.safetensors").c_str());
    LLMWeights llm_w; memset(&llm_w, 0, sizeof(llm_w));
    load_llm_weights(sf3, llm_w, 28);
    
    float * lm_hiddens = new float[n_tok * 1536];
    LLMKVCache kv_cache; memset(&kv_cache, 0, sizeof(kv_cache));
    llm_kv_cache_init(llm_w, tokens, n_tok, lm_hiddens, kv_cache);
    
    // Hidden proj for conditioning
    float * last_hidden = lm_hiddens + (n_tok - 1) * 1536;
    float hidden0[1024];
    { float * hw = tensor_data(dit_aux.hidden_proj_w);
      for (int o = 0; o < 1024; o++) { float s = 0; for (int i = 0; i < 1536; i++) s += hw[o*1536+i] * last_hidden[i]; hidden0[o] = s; } }
    
    // Build FM buffer [1, 1024]
    int patch_size = 4, latent_dim = 128, hidden = 1024;
    int fm_capacity = (1 + patch_size) * n_calls + 1;
    float * fm_buffer = new float[fm_capacity * hidden];
    memcpy(fm_buffer, hidden0, hidden * sizeof(float));
    int fm_seq_len = 1;
    
    // Generate latents
    const int NFE = 10;
    float dt = 1.0f / NFE;
    float * all_latents = new float[n_calls * patch_size * latent_dim];
    int total_frames = 0;
    
    ggml_context * gctx_p = nullptr;
    
    for (int call = 0; call < n_calls; call++) {
        int total_len = fm_seq_len + patch_size;
        
        // Build DiT input as torch tensor
        auto dit_input = torch::zeros({1, total_len, 1024});
        float * dit_ptr = dit_input.data_ptr<float>();
        memcpy(dit_ptr, fm_buffer, fm_seq_len * hidden * sizeof(float));
        
        // Noise from pre-computed table
        float z_t[512];
        if (call < 30) memcpy(z_t, NOISE_PIPELINE[call], 512 * sizeof(float));
        else { for (int i = 0; i < 512; i++) z_t[i] = 0; }
        
        // Euler solver
        for (int step = 0; step < NFE; step++) {
            float t_val = (float)step * dt;
            
            // coord_proj for noise
            float * cw = tensor_data(dit_aux.coord_proj_w);
            float * cb = dit_aux.coord_proj_b ? tensor_data(dit_aux.coord_proj_b) : nullptr;
            for (int p = 0; p < patch_size; p++) {
                float * dp = dit_ptr + (fm_seq_len + p) * hidden;
                for (int o = 0; o < hidden; o++) {
                    float s = cb ? cb[o] : 0;
                    for (int i = 0; i < latent_dim; i++) s += cw[o*latent_dim+i] * z_t[p*latent_dim+i];
                    dp[o] = s;
                }
            }
            
            // Run traced DiT
            auto t_tensor = torch::tensor({t_val});
            auto mask = torch::ones({1, total_len, total_len}, torch::kBool);
            auto pos_ids = torch::arange(0, total_len, torch::kFloat32).unsqueeze(0);
            
            auto out = dit.forward({dit_input, t_tensor, mask, pos_ids}).toTensor();  // [1, total_len, 128]
            auto vel = out.slice(1, fm_seq_len, total_len);  // [1, 4, 128]
            float * vp = vel.data_ptr<float>();
            
            // Euler step
            for (int i = 0; i < 512; i++) z_t[i] += vp[i] * dt;
        }
        
        // Store latents
        memcpy(all_latents + call * 512, z_t, 512 * sizeof(float));
        total_frames += patch_size;
        
        // Latent proj for FM buffer
        float * lw = tensor_data(dit_aux.latent_proj_w);
        float * lb = dit_aux.latent_proj_b ? tensor_data(dit_aux.latent_proj_b) : nullptr;
        for (int p = 0; p < patch_size; p++) {
            float * dp = fm_buffer + (fm_seq_len + p) * hidden;
            for (int o = 0; o < hidden; o++) {
                float s = lb ? lb[o] : 0;
                for (int i = 0; i < latent_dim; i++) s += lw[o*latent_dim+i] * z_t[p*latent_dim+i];
                dp[o] = s;
            }
        }
        fm_seq_len += patch_size;
        
        // PatchEncoder + LM for AR feedback
        float z_denorm[512]; memcpy(z_denorm, z_t, 512*sizeof(float));
        denormalize_latents(z_denorm, patch_size, latent_dim);
        
        if (!gctx_p) { ggml_init_params gp2 = { .mem_size = 256ULL*1024*1024 }; gctx_p = ggml_init(gp2); }
        else ggml_reset(gctx_p);
        ggml_tensor * pe_x = ggml_new_tensor_2d(gctx_p, GGML_TYPE_F32, latent_dim, patch_size);
        memcpy(tensor_data(pe_x), z_denorm, 512*sizeof(float));
        ggml_tensor * pe_out = patchenc_forward(pe, gctx_p, pe_x, 1);
        float * pe_data = tensor_data(pe_out);
        
        float new_hidden[1536];
        llm_kv_cache_step(llm_w, pe_data, kv_cache, new_hidden);
        
        // Hidden proj for FM buffer
        float * hw = tensor_data(dit_aux.hidden_proj_w);
        for (int o = 0; o < hidden; o++) {
            float s = 0; for (int i = 0; i < 1536; i++) s += hw[o*1536+i] * new_hidden[i];
            fm_buffer[fm_seq_len * hidden + o] = s;
        }
        fm_seq_len++;
        
        printf("  Call %d/%d: lat RMS=%.4f fm=%d\n", call+1, n_calls, 
            [&]{float r=0;for(int i=0;i<512;i++)r+=z_t[i]*z_t[i];return sqrtf(r/512);}(), fm_seq_len);
    }
    
    // VAE normalization + BigVGAN
    float * latents_vae = new float[total_frames * latent_dim];
    float ch_mean[128]={0}, ch_std[128]={0};
    for (int f = 0; f < total_frames; f++) for (int c = 0; c < latent_dim; c++) ch_mean[c] += all_latents[f*latent_dim+c];
    for (int c = 0; c < latent_dim; c++) ch_mean[c] /= total_frames;
    for (int f = 0; f < total_frames; f++) for (int c = 0; c < latent_dim; c++) { float d = all_latents[f*latent_dim+c]-ch_mean[c]; ch_std[c]+=d*d; }
    for (int c = 0; c < latent_dim; c++) ch_std[c] = sqrtf(ch_std[c]/total_frames+1e-8f);
    for (int f = 0; f < total_frames; f++) for (int c = 0; c < latent_dim; c++)
        latents_vae[f*latent_dim+c] = (all_latents[f*latent_dim+c]-ch_mean[c])/ch_std[c]*VAE_STD[c]+VAE_MEAN[c];
    
    BigVGANDecoder bigvgan;
    if (bigvgan_load("models/vocoder_eff.safetensors", bigvgan)) {
        int real_samples;
        float * real_wav = new float[total_frames * VAE_HOP_SAMPLES];
        bigvgan_decode(bigvgan, latents_vae, total_frames, real_wav, &real_samples);
        
        FILE * wf = fopen(out_path, "wb");
        if (wf) {
            int ds = real_samples * 2, fs = 36 + ds;
            fwrite("RIFF",1,4,wf); fwrite(&fs,4,1,wf); fwrite("WAVE",1,4,wf);
            fwrite("fmt ",1,4,wf); int fz=16; fwrite(&fz,4,1,wf);
            short af=1,nc=1; fwrite(&af,2,1,wf); fwrite(&nc,2,1,wf);
            int sr=48000; fwrite(&sr,4,1,wf); int br=sr*2; fwrite(&br,4,1,wf);
            short ba=2,bp=16; fwrite(&ba,2,1,wf); fwrite(&bp,2,1,wf);
            fwrite("data",1,4,wf); fwrite(&ds,4,1,wf);
            for (int i=0;i<real_samples;i++){float s=real_wav[i]*32767;if(s>32767)s=32767;if(s<-32768)s=-32768;short si=(short)s;fwrite(&si,2,1,wf);}
            fclose(wf);
            printf("  %s: %d samples (%d ms total)\n", out_path, real_samples, (int)(now_ms()-t0));
        }
        delete[] real_wav; bigvgan_free(bigvgan);
    }
    
    delete[] all_latents; delete[] latents_vae; delete[] lm_hiddens; delete[] fm_buffer;
    ggml_free(gctx_p); ggml_free(w_ctx); ggml_free(w_ctx_pe);
    llm_kv_cache_free(kv_cache); free_llm_weights(llm_w);
    return 0;
}
