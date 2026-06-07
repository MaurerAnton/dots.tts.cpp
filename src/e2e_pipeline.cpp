// dots.tts.cpp - End-to-end pipeline
// Text -> LLM -> hidden_proj -> DiT -> AudioVAE -> WAV

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

static float * tensor_data(ggml_tensor * t) {
    return (float *)((char *)t->data + t->view_offs);
}

int main(int argc, char ** argv) {
    setbuf(stdout, NULL);
    const char * text = argc > 1 ? argv[1] : "Hello world";

    printf("dots.tts.cpp - End-to-end TTS pipeline\n");
    printf("Input: '%s'\n\n", text);

    // === Phase 1: LLM loads + decodes (no safetensors yet) ===
    printf("[1] Loading LLM...\n");
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * llm = llama_model_load_from_file(
        "/home/bym/dots.tts.cpp/models/llm_qwen25_1.5b.gguf", mp);
    if (!llm) { printf("FAILED\n"); return 1; }
    int n_embd = llama_model_n_embd(llm);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 256; cp.embeddings = true;
    llama_context * lctx = llama_init_from_model(llm, cp);

    printf("[2] Tokenizing + decoding...\n");
    llama_token tokens[256];
    int n_tok = llama_tokenize(llama_model_get_vocab(llm), text, strlen(text),
                                tokens, 256, true, false);
    printf("  %d tokens\n", n_tok);

    llama_batch batch = llama_batch_get_one(tokens, n_tok);
    if (llama_decode(lctx, batch) != 0) { printf("Decode failed\n"); return 1; }
    printf("  Decode OK\n");

    // Extract hidden states
    float * hiddens = new float[n_tok * n_embd];
    for (int i = 0; i < n_tok; i++) {
        float * e = llama_get_embeddings_ith(lctx, i);
        if (e) memcpy(hiddens + i * n_embd, e, n_embd * sizeof(float));
    }
    printf("  Hidden states: %d tokens extracted\n", n_tok);

    // Free LLM resources (no longer needed)
    llama_free(lctx);
    llama_model_free(llm);
    printf("  LLM freed\n");

    // === Phase 2: Load DiT + run TTS ===
    printf("[3] Loading DiT weights from safetensors...\n");
    const char * sf_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-base/blobs/69dbad797566b24003506e1dd698597937149920f6df9782d84214bf477acb48";
    SafeTensorsFile sf;
    if (!sf.open(sf_path)) { printf("FAILED\n"); return 1; }

    struct ggml_init_params wp = { .mem_size = 4ULL*1024*1024*1024, .mem_buffer = nullptr, .no_alloc = false };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model dit;
    if (!load_dit_weights(sf, w_ctx, dit)) { printf("FAILED\n"); return 1; }
    sf.close();
    printf("  DiT: %d layers loaded\n", DIT_NUM_LAYERS);

    // Project hiddens via hidden_proj [1536 -> 1024]
    printf("[4] Running DiT flow matching...\n");
    struct ggml_init_params gp = { .mem_size = 512ULL*1024*1024, .mem_buffer = nullptr, .no_alloc = false };
    ggml_context * gctx = ggml_init(gp);

    ggml_tensor * ht = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, n_tok);
    memcpy(tensor_data(ht), hiddens, n_tok * n_embd * sizeof(float));
    delete[] hiddens;

    printf("  Computing hidden_proj (%lld x %d) @ (%d x %d)...\n",
           (long long)dit.hidden_proj_w->ne[0], (int)dit.hidden_proj_w->ne[1],
           n_embd, n_tok);
    ggml_tensor * cond = ggml_mul_mat(gctx, dit.hidden_proj_w, ht);
    printf("  cond shape: [%lld x %lld]\n", (long long)cond->ne[0], (long long)cond->ne[1]);
    ggml_cgraph * cgf = ggml_new_graph(gctx);
    ggml_build_forward_expand(cgf, cond);
    printf("  Computing hidden_proj graph...\n");
    ggml_graph_compute_with_ctx(gctx, cgf, 8);
    printf("  hidden_proj done\n");

    // Extract cond data before ggml_reset frees it
    int cond_hidden = DIT_HIDDEN_SIZE;
    float * cond_data = new float[cond_hidden * n_tok];
    memcpy(cond_data, tensor_data(cond), cond_hidden * n_tok * sizeof(float));

    // Generate audio latents via DiT
    int n_patches = 4;
    int patch_flat = PATCHENC_PATCH_SIZE * VAE_LATENT_DIM;
    float * all_latents = new float[n_patches * patch_flat];

    for (int p = 0; p < n_patches; p++) {
        printf("  DiT patch %d/%d...\n", p+1, n_patches);
        ggml_reset(gctx);
        int cond_seq = 32;
        ggml_tensor * dx = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, 1, cond_seq);
        memset(tensor_data(dx), 0, cond_seq * DIT_HIDDEN_SIZE * sizeof(float));
        {
            float * dd = tensor_data(dx);
            for (int i = 0; i < n_tok && i < cond_seq; i++)
                memcpy(dd + i * DIT_HIDDEN_SIZE, cond_data + i * cond_hidden, cond_hidden * sizeof(float));
        }
        dx = ggml_cont(gctx, ggml_permute(gctx, dx, 2, 1, 0, 3));

        ggml_tensor * ti = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);
        ((float*)ti->data)[0] = 0.5f;

        ggml_tensor * dout = dit_forward(dit, gctx, dx, ti, nullptr);
        ggml_cgraph * dgf = ggml_new_graph(gctx);
        ggml_build_forward_expand(dgf, dout);
        ggml_graph_compute_with_ctx(gctx, dgf, 8);
        memcpy(all_latents + p * patch_flat, tensor_data(dout), patch_flat * sizeof(float));
    }
    printf("  Generated %d latent patches\n", n_patches);

    // AudioVAE decode
    printf("[5] AudioVAE decode + WAV output...\n");
    int n_frames = n_patches * PATCHENC_PATCH_SIZE;
    int n_samples;
    float * wav = new float[n_frames * VAE_HOP_SAMPLES];

    ggml_reset(gctx);
    ggml_tensor * lt = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VAE_LATENT_DIM, n_frames);
    memcpy(tensor_data(lt), all_latents, n_frames * VAE_LATENT_DIM * sizeof(float));
    audiovae_decode_simple(gctx, lt, n_frames, wav, &n_samples);

    // Write WAV
    const char * wav_path = "output.wav";
    FILE * wf = fopen(wav_path, "wb");
    if (wf) {
        int ds = n_samples * 2;
        int fs = 36 + ds;
        fwrite("RIFF",1,4,wf); fwrite(&fs,4,1,wf); fwrite("WAVE",1,4,wf);
        fwrite("fmt ",1,4,wf);
        int fz=16; fwrite(&fz,4,1,wf);
        short af=1; fwrite(&af,2,1,wf);
        short nc=1; fwrite(&nc,2,1,wf);
        int sr=VAE_SAMPLE_RATE; fwrite(&sr,4,1,wf);
        int br=sr*2; fwrite(&br,4,1,wf);
        short ba=2; fwrite(&ba,2,1,wf);
        short bp=16; fwrite(&bp,2,1,wf);
        fwrite("data",1,4,wf); fwrite(&ds,4,1,wf);
        for (int i=0;i<n_samples;i++) {
            float s=wav[i]*32767.0f;
            if(s>32767)s=32767; if(s<-32768)s=-32768;
            short si=(short)s; fwrite(&si,2,1,wf);
        }
        fclose(wf);
        printf("  Wrote %s: %d samples, %.2f sec @ %d Hz\n",
               wav_path, n_samples, n_samples/(float)VAE_SAMPLE_RATE, VAE_SAMPLE_RATE);
    }

    delete[] wav;
    delete[] all_latents;
    delete[] cond_data;
    ggml_free(gctx);
    ggml_free(w_ctx);

    printf("\nDone.\n");
    return 0;
}
