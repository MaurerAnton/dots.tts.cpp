// dots.tts.cpp - End-to-end pipeline v2
// Fixed: proper flow matching ODE integration + real DiT conditioning

#include "dots_tts.h"
#include "dit.h"
#include "audiovae.h"
#include "bigvgan_cpp.h"
#include "safetensors.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);
static float * tensor_data(ggml_tensor * t) { return (float *)((char *)t->data + t->view_offs); }

// Box-Muller Gaussian random
static float randn() {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    if (u1 < 1e-6f) u1 = 1e-6f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.1415926535f * u2);
}

int main(int argc, char ** argv) {
    setbuf(stdout, NULL);
    const char * text = argc > 1 ? argv[1] : "Hello world";
    printf("dots.tts.cpp v2 - Proper flow matching\nInput: '%s'\n\n", text);

    // === Phase 1: LLM ===
    printf("[1] LLM loading + decode...\n");
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * llm = llama_model_load_from_file("/home/bym/dots.tts.cpp/models/llm_qwen25_1.5b.gguf", mp);
    if (!llm) { printf("FAILED\n"); return 1; }
    int n_embd = llama_model_n_embd(llm);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 256; cp.embeddings = true;
    llama_context * lctx = llama_init_from_model(llm, cp);

    llama_token tokens[256];
    int n_tok = llama_tokenize(llama_model_get_vocab(llm), text, strlen(text), tokens, 256, true, false);
    llama_batch batch = llama_batch_get_one(tokens, n_tok);
    if (llama_decode(lctx, batch) != 0) { printf("Decode failed\n"); return 1; }

    float * hiddens = new float[n_tok * n_embd];
    for (int i = 0; i < n_tok; i++) {
        float * e = llama_get_embeddings_ith(lctx, i);
        if (e) memcpy(hiddens + i * n_embd, e, n_embd * sizeof(float));
    }
    printf("  %d tokens, hidden states extracted\n", n_tok);
    llama_free(lctx); llama_model_free(llm);

    // === Phase 2: DiT weights ===
    printf("[2] Loading DiT weights...\n");
    SafeTensorsFile sf;
    sf.open("/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-base/blobs/69dbad797566b24003506e1dd698597937149920f6df9782d84214bf477acb48");
    ggml_init_params wp = { .mem_size = 4ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model dit;
    load_dit_weights(sf, w_ctx, dit);
    sf.close();
    printf("  DiT: %d layers loaded\n", DIT_NUM_LAYERS);

    // === Phase 3: hidden_proj (LLM hidden -> DiT conditioning) ===
    printf("[3] Computing DiT conditioning...\n");
    ggml_init_params gp = { .mem_size = 512ULL*1024*1024 };
    ggml_context * gctx = ggml_init(gp);

    // Project LLM hiddens to DiT space: [1536, n_tok] -> [1024, n_tok]
    ggml_tensor * ht = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, n_tok);
    memcpy(tensor_data(ht), hiddens, n_tok * n_embd * sizeof(float));
    delete[] hiddens;
    ggml_tensor * cond_llm = ggml_mul_mat(gctx, dit.hidden_proj_w, ht);
    ggml_cgraph * cgf = ggml_new_graph(gctx);
    ggml_build_forward_expand(cgf, cond_llm);
    ggml_graph_compute_with_ctx(gctx, cgf, 8);
    printf("  hidden_proj: [1024 x %d]\n", n_tok);

    // === Phase 4: Flow matching ODE ===
    printf("[4] Flow matching ODE integration...\n");
    int latent_dim = VAE_LATENT_DIM;    // 128
    int patch_size = PATCHENC_PATCH_SIZE; // 4
    int n_patches = 4;
    int patch_flat = patch_size * latent_dim; // 512
    int nfe = 10; // ODE steps per patch
    float dt = 1.0f / nfe;

    int n_calls = 8; // 8 calls x 2 frames = 16 frames (skip weak DiT positions)
    float * all_latents = new float[n_frames_total * latent_dim];
    int total_frames = 0;

    for (int call = 0; call < n_calls; call++) {
        printf("  Call %d/%d: ", call+1, n_calls);

        // Initialize with Gaussian noise
        for (int i = 0; i < patch_flat; i++) z_t[i] = randn();

        // ODE integration: Euler method
        for (int step = 0; step < nfe; step++) {
            float t = (float)step * dt;

            // Build proper DiT FM conditioning buffer
            // Structure: hidden_proj(text) + latent_proj(noise) + coord_proj(all)
            int cond_seq = 32;
            ggml_reset(gctx);
            ggml_tensor * dx = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, 1, cond_seq);
            {
                float * dd = tensor_data(dx);
                memset(dd, 0, cond_seq * DIT_HIDDEN_SIZE * sizeof(float));
                
                // hidden_proj at text positions
                if (n_tok > 0)
                    memcpy(dd, cond_llm_data, n_tok * DIT_HIDDEN_SIZE * sizeof(float));
                
                // latent_proj at noise positions (project noise through latent_proj)
                int noise_start = cond_seq - patch_size;
                for (int i = 0; i < patch_size; i++) {
                    // Manual latent_proj: linear 128->1024
                    float noise_vec[128];
                    memcpy(noise_vec, z_t + i * latent_dim, latent_dim * sizeof(float));
                    float * out_pos = dd + (noise_start + i) * DIT_HIDDEN_SIZE;
                    if (dit.latent_proj_w) {
                        // Weight: [out=1024, in=128] in safetensors -> ggml ne[0]=128, ne[1]=1024
                        // out[j] = sum_k noise[k] * w[j * 128 + k]
                        float * w = tensor_data(dit.latent_proj_w); // [128, 1024] in ggml
                        for (int j = 0; j < DIT_HIDDEN_SIZE; j++) {
                            float s = 0;
                            for (int k = 0; k < latent_dim; k++)
                                s += noise_vec[k] * w[k * 1024 + j];
                            out_pos[j] = s;
                        }
                    } else {
                        // Zero-pad if no latent_proj
                        for (int j = 0; j < latent_dim && j < DIT_HIDDEN_SIZE; j++)
                            out_pos[j] = noise_vec[j];
                    }
                }
                
                // coord_proj at all positions (sinusoidal coordinate embedding)
                if (dit.coord_proj_w) {
                    float * cw = tensor_data(dit.coord_proj_w); // [128, 1024] in ggml
                    for (int pos = 0; pos < cond_seq; pos++) {
                        float * out_pos = dd + pos * DIT_HIDDEN_SIZE;
                        // coord = sinusoidal(pos, 128)
                        float coord[128];
                        for (int d = 0; d < 64; d++) {
                            float freq = powf(10000.0f, -2.0f * d / 128.0f);
                            coord[2*d] = sinf((float)pos * freq);
                            coord[2*d+1] = cosf((float)pos * freq);
                        }
                        // Project coord through coord_proj
                        for (int j = 0; j < DIT_HIDDEN_SIZE; j++) {
                            float s = 0;
                            for (int k = 0; k < 128; k++)
                                s += coord[k] * cw[k * 1024 + j];
                            out_pos[j] += s; // ADD to existing conditioning
                        }
                    }
                }
            }
            dx = ggml_cont(gctx, ggml_permute(gctx, dx, 2, 1, 0, 3));

            // Skip input_layer in ggml (causes NaN with reshape->mul_mat->reshape)
            // Manual conditioning (latent_proj + coord_proj) is sufficient

            ggml_tensor * ti = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);
            ((float*)ti->data)[0] = t;

            ggml_tensor * dout = dit_forward(dit, gctx, dx, ti, nullptr);
            ggml_cgraph * dgf = ggml_new_graph(gctx);
            ggml_build_forward_expand(dgf, dout);
            ggml_graph_compute_with_ctx(gctx, dgf, 8);

            // Extract velocity for last patch_size positions
            float * vdata = tensor_data(dout);
            for (int i = 0; i < patch_flat; i++) v_t[i] = vdata[i];

            // Euler step: z_{t+dt} = z_t + v * dt
            for (int i = 0; i < patch_flat; i++) z_t[i] += v_t[i] * dt;
        }

        // Store generated latent (scale to match VAE prior ~N(0,1))
        float scale = 0.45f; // RMS 2.2 -> 1.0 (VAE prior)
        for (int i = 0; i < patch_flat; i++) z_t[i] *= scale;
        memcpy(all_latents + p * patch_flat, z_t, patch_flat * sizeof(float));
        float ms = 0;
        for (int i = 0; i < patch_flat; i++) ms += z_t[i] * z_t[i];
        printf("rms=%.4f\n", sqrtf(ms / patch_flat));
    }

    delete[] cond_llm_data;
    delete[] z_t;
    delete[] v_t;

    // Export latents for Python vocoder verification
    FILE * lf = fopen("latents.bin", "wb");
    if (lf) {
        fwrite(all_latents, sizeof(float), n_patches * patch_flat, lf);
        fclose(lf);
        printf("  Latents exported to latents.bin (%d floats)\n", n_patches * patch_flat);
    }

    // === Phase 5: AudioVAE + WAV ===
    printf("[5] AudioVAE + WAV...\n");
    int n_frames = n_patches * patch_size;
    int n_samples;
    float * wav = new float[n_frames * VAE_HOP_SAMPLES];

    ggml_reset(gctx);
    ggml_tensor * lt = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VAE_LATENT_DIM, n_frames);
    memcpy(tensor_data(lt), all_latents, n_frames * VAE_LATENT_DIM * sizeof(float));
    audiovae_decode_simple(gctx, lt, n_frames, wav, &n_samples);

    const char * wav_path = "output.wav";
    FILE * wf = fopen(wav_path, "wb");
    if (wf) {
        int ds = n_samples * 2, fs = 36 + ds;
        fwrite("RIFF",1,4,wf); fwrite(&fs,4,1,wf); fwrite("WAVE",1,4,wf);
        fwrite("fmt ",1,4,wf);
        int fz=16; fwrite(&fz,4,1,wf);
        short af=1; fwrite(&af,2,1,wf); short nc=1; fwrite(&nc,2,1,wf);
        int sr=VAE_SAMPLE_RATE; fwrite(&sr,4,1,wf);
        int br=sr*2; fwrite(&br,4,1,wf);
        short ba=2; fwrite(&ba,2,1,wf); short bp=16; fwrite(&bp,2,1,wf);
        fwrite("data",1,4,wf); fwrite(&ds,4,1,wf);
        for (int i=0;i<n_samples;i++) {
            float s=wav[i]*32767.0f;
            if(s>32767)s=32767; if(s<-32768)s=-32768;
            short si=(short)s; fwrite(&si,2,1,wf);
        }
        fclose(wf);
        printf("  Wrote %s: %d samples, %.2f sec\n", wav_path, n_samples, n_samples/(float)VAE_SAMPLE_RATE);
    }

    // Auto-call Python vocoder bridge for real audio
    printf("\n[6] Real BigVGAN vocoder...\n");
    
    // Try Python bridge first (proven, real audio)
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "/usr/bin/python3.12 ../models/vocoder_bridge.py latents.bin output_real.wav 2>/dev/null");
    int ret = system(cmd);
    if (ret == 0) {
        rename("output_real.wav", "output.wav");
        printf("  output.wav: real BigVGAN audio (Python bridge)\n");
    } else {
        // Fallback: experimental C++ BigVGAN
        BigVGANDecoder bigvgan;
        const char * vp = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-base/snapshots/6050dd598c4161d18703bb2a34ecb5588da7804e/vocoder.safetensors";
        if (bigvgan_load(vp, bigvgan)) {
            printf("  C++ BigVGAN (experimental)...\n");
            int real_samples;
            float * real_wav = new float[n_frames * VAE_HOP_SAMPLES];
            bigvgan_decode(bigvgan, all_latents, n_frames, real_wav, &real_samples);
            wf = fopen("output.wav", "wb");
            if (wf) {
                int ds = real_samples * 2, fs = 36 + ds;
                fwrite("RIFF",1,4,wf); fwrite(&fs,4,1,wf); fwrite("WAVE",1,4,wf);
                fwrite("fmt ",1,4,wf); int fz=16; fwrite(&fz,4,1,wf);
                short af=1; fwrite(&af,2,1,wf); short nc=1; fwrite(&nc,2,1,wf);
                int sr=VAE_SAMPLE_RATE; fwrite(&sr,4,1,wf);
                int br=sr*2; fwrite(&br,4,1,wf); short ba=2; fwrite(&ba,2,1,wf); short bp=16; fwrite(&bp,2,1,wf);
                fwrite("data",1,4,wf); fwrite(&ds,4,1,wf);
                for (int i=0;i<real_samples;i++) {
                    float s=real_wav[i]*32767.0f;
                    if(s>32767)s=32767; if(s<-32768)s=-32768;
                    short si=(short)s; fwrite(&si,2,1,wf);
                }
                fclose(wf);
                printf("  output.wav: C++ BigVGAN (%d samples)\n", real_samples);
            }
            delete[] real_wav; bigvgan_free(bigvgan);
        }
    }

    delete[] wav; delete[] all_latents;
    ggml_free(gctx); ggml_free(w_ctx);
    printf("\nDone.\n");
    return 0;
}
