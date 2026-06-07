// dots.tts.cpp - End-to-end pipeline v2
// Fixed: proper flow matching ODE integration + real DiT conditioning

#include "dots_tts.h"
#include "dit.h"
#include "audiovae.h"
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

    float * all_latents = new float[n_patches * patch_flat];
    float * z_t = new float[patch_flat];  // current ODE state
    float * v_t = new float[patch_flat];  // velocity field

    // Copy cond_llm for reuse across patches
    float * cond_llm_data = new float[DIT_HIDDEN_SIZE * n_tok];
    memcpy(cond_llm_data, tensor_data(cond_llm), DIT_HIDDEN_SIZE * n_tok * sizeof(float));

    for (int p = 0; p < n_patches; p++) {
        printf("  Patch %d/%d: ", p+1, n_patches);

        // Initialize with Gaussian noise
        for (int i = 0; i < patch_flat; i++) z_t[i] = randn();

        // ODE integration: Euler method
        for (int step = 0; step < nfe; step++) {
            float t = (float)step * dt;

            // Build DiT input: pad to seq_len=32 for stable attention
            int cond_seq = 32; // minimum for DiT attention
            ggml_reset(gctx);
            ggml_tensor * dx = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, 1, cond_seq);
            {
                float * dd = tensor_data(dx);
                // Zero all
                for (int i = 0; i < cond_seq * DIT_HIDDEN_SIZE; i++) dd[i] = 0.0f;
                // Copy LLM conditioning at start
                if (n_tok > 0)
                    memcpy(dd, cond_llm_data, n_tok * DIT_HIDDEN_SIZE * sizeof(float));
                // Copy noise latent at end (scaled down)
                for (int i = 0; i < patch_size; i++)
                    for (int j = 0; j < latent_dim && j < DIT_HIDDEN_SIZE; j++)
                        dd[(cond_seq - patch_size + i) * DIT_HIDDEN_SIZE + j] = z_t[i * latent_dim + j] * 0.02f;
            }
            dx = ggml_cont(gctx, ggml_permute(gctx, dx, 2, 1, 0, 3));

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

        // Store generated latent
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

    delete[] wav; delete[] all_latents;
    ggml_free(gctx); ggml_free(w_ctx);
    printf("\nDone.\n");
    return 0;
}
