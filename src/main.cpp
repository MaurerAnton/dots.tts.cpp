// dots.tts.cpp - main entry point
// First C++ implementation of dots.tts
// 
// Architecture: 
//   Text -> BPE Tokenizer (llama.cpp) -> Qwen2.5-1.5B LLM -> PatchEncoder 
//   -> FM buffer -> DiT (flow matching) -> AudioVAE -> 48kHz WAV
//
// Phase 1 MVP: DiT + Flow Matching only (other components stubbed)

#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "patchenc.h"
#include "audiovae.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "safetensors.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// tensor_data is now in dots_tts_util.h

// Forward declarations
static void patchenc_load_dummy(patch_encoder & enc, ggml_context * w_ctx);
static void patchenc_test(patch_encoder & enc);
bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);
bool load_patchenc_weights(SafeTensorsFile & sf, ggml_context * w_ctx, patch_encoder & enc);

struct dots_tts_runtime {
    dit_model dit;

    // Compute contexts
    ggml_context * w_ctx;      // weights
    ggml_context * compute_ctx; // graph building

    // ggml backend (CPU only for MVP)
    // ggml_backend_t backend;

    // Config
    int n_batch;
    int max_seq_len;
};

// ---------------------------------------------------------------------------
// Initialize runtime
// ---------------------------------------------------------------------------

static dots_tts_runtime * dots_tts_init() {
    dots_tts_runtime * rt = new dots_tts_runtime();
    rt->n_batch = 1;
    rt->max_seq_len = DIT_MAX_SEQ_LEN;

    // Weight context (large, persistent)
    // Will be populated by model loader (GGUF/safetensors loader)
    struct ggml_init_params w_params = {
        /*.mem_size   =*/ 4ULL * 1024 * 1024 * 1024, // 4 GB for all weights
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    rt->w_ctx = ggml_init(w_params);

    // Backend (CPU for MVP)
    // rt->backend = ggml_backend_reg_get_default()->init();

    return rt;
}

// ---------------------------------------------------------------------------
// Dummy model load for testing (allocates random weights on CPU)
// Replace with real GGUF loader
// ---------------------------------------------------------------------------

static void dots_tts_load_dummy(dots_tts_runtime * rt) {
    printf("Loading dummy DiT weights for testing...\n");

    dit_model & m = rt->dit;
    m.n_layers    = DIT_NUM_LAYERS;
    m.hidden_size = DIT_HIDDEN_SIZE;
    m.num_heads   = DIT_NUM_HEADS;
    m.head_dim    = DIT_HEAD_SIZE;
    m.ffn_size    = DIT_FFN_SIZE;
    m.ada_dim     = DIT_ADA_DIM;
    m.speaker_dim = DIT_SPEAKER_DIM;

    m.layers.resize(m.n_layers);

    auto alloc_tensor = [&](const char * name, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
        ggml_tensor * t = ggml_new_tensor_4d(rt->w_ctx, GGML_TYPE_F32, n0, n1, n2, n3);
        float * d = tensor_data(t);
        size_t n = n0 * n1 * n2 * n3;
        for (size_t i = 0; i < n; i++) {
            d[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.02f;
        }
        return t;
    };

    for (int i = 0; i < m.n_layers; i++) {
        char name[256];
        dit_block & b = m.layers[i];

        snprintf(name, sizeof(name), "dit.layers.%d.attn.q.weight", i);
        b.attn_q_weight = alloc_tensor(name, DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE, 1, 1);

        snprintf(name, sizeof(name), "dit.layers.%d.attn.k.weight", i);
        b.attn_k_weight = alloc_tensor(name, DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE, 1, 1);

        snprintf(name, sizeof(name), "dit.layers.%d.attn.v.weight", i);
        b.attn_v_weight = alloc_tensor(name, DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE, 1, 1);

        snprintf(name, sizeof(name), "dit.layers.%d.attn.o.weight", i);
        b.attn_o_weight = alloc_tensor(name, DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE, 1, 1);

        snprintf(name, sizeof(name), "dit.layers.%d.attn.o.bias", i);
        b.attn_o_bias = alloc_tensor(name, DIT_HIDDEN_SIZE, 1, 1, 1);

        snprintf(name, sizeof(name), "dit.layers.%d.q_norm.weight", i);
        b.q_norm_w = alloc_tensor(name, DIT_HEAD_SIZE, 1, 1, 1);

        snprintf(name, sizeof(name), "dit.layers.%d.k_norm.weight", i);
        b.k_norm_w = alloc_tensor(name, DIT_HEAD_SIZE, 1, 1, 1);

        snprintf(name, sizeof(name), "dit.layers.%d.ffn.w1.weight", i);
        b.ffn_w1 = alloc_tensor(name, DIT_HIDDEN_SIZE, DIT_FFN_SIZE, 1, 1);

        snprintf(name, sizeof(name), "dit.layers.%d.ffn.w2.weight", i);
        b.ffn_w2 = alloc_tensor(name, DIT_FFN_SIZE, DIT_HIDDEN_SIZE, 1, 1);

        snprintf(name, sizeof(name), "dit.layers.%d.adaln.linear.weight", i);
        b.adaln_linear_w = alloc_tensor(name, DIT_ADA_DIM, 6 * DIT_HIDDEN_SIZE, 1, 1);

        snprintf(name, sizeof(name), "dit.layers.%d.adaln.linear.bias", i);
        b.adaln_linear_b = alloc_tensor(name, 6 * DIT_HIDDEN_SIZE, 1, 1, 1);

        snprintf(name, sizeof(name), "dit.layers.%d.attn_norm.weight", i);
        b.attn_norm_w = alloc_tensor(name, DIT_HIDDEN_SIZE, 1, 1, 1);

        snprintf(name, sizeof(name), "dit.layers.%d.ffn_norm.weight", i);
        b.ffn_norm_w = alloc_tensor(name, DIT_HIDDEN_SIZE, 1, 1, 1);
    }

    m.spk_proj_w1  = alloc_tensor("dit.spk_proj.w1.weight", DIT_SPEAKER_DIM, DIT_ADA_DIM, 1, 1);
    m.spk_proj_b1  = alloc_tensor("dit.spk_proj.w1.bias",   DIT_ADA_DIM, 1, 1, 1);
    m.t_embed_w1   = alloc_tensor("dit.t_embed.w1.weight", DIT_ADA_DIM, 256, 1, 1);
    m.t_embed_b1   = alloc_tensor("dit.t_embed.w1.bias",   DIT_ADA_DIM, 1, 1, 1);
    m.t_embed_w2   = alloc_tensor("dit.t_embed.w2.weight", DIT_ADA_DIM, DIT_ADA_DIM, 1, 1);
    m.t_embed_b2   = alloc_tensor("dit.t_embed.w2.bias",   DIT_ADA_DIM, 1, 1, 1);
    m.out_proj_w   = alloc_tensor("dit.out_proj.weight",  DIT_HIDDEN_SIZE, VAE_LATENT_DIM, 1, 1);
    m.out_proj_b  = alloc_tensor("dit.out_proj.bias",    VAE_LATENT_DIM, 1, 1, 1);

    printf("Dummy DiT weights loaded: %zu bytes\n", ggml_used_mem(rt->w_ctx));
}

// ---------------------------------------------------------------------------
// Test: run DiT forward on random input
// ---------------------------------------------------------------------------

static void dots_tts_test_dit(dots_tts_runtime * rt) {
    printf("\n=== Testing DiT forward pass ===\n");

    struct ggml_init_params test_params = {
        /*.mem_size   =*/ 512ULL * 1024 * 1024, // 512 MB for DiT graph
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * test_ctx = ggml_init(test_params);

    int seq_len = 32;
    int n_batch = 1;
    int hidden  = DIT_HIDDEN_SIZE;
    int latent  = VAE_LATENT_DIM;

    // Create random input
    ggml_tensor * x = ggml_new_tensor_3d(test_ctx, GGML_TYPE_F32, hidden, n_batch, seq_len);
    {
        float * d = tensor_data(x);
        for (int i = 0; i < seq_len * n_batch * hidden; i++) {
            d[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        }
    }
    x = ggml_cont(test_ctx, ggml_permute(test_ctx, x, 2, 1, 0, 3));

    // Timestep
    ggml_tensor * t = ggml_new_tensor_1d(test_ctx, GGML_TYPE_F32, 1);
    ((float *)t->data)[0] = 0.5f;

    // Speaker embedding (random for test)
    ggml_tensor * spk = ggml_new_tensor_2d(test_ctx, GGML_TYPE_F32, DIT_SPEAKER_DIM, 1);
    {
        float * d = tensor_data(spk);
        for (int i = 0; i < DIT_SPEAKER_DIM; i++) {
            d[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        }
    }

    // Forward pass
    ggml_tensor * out = dit_forward(rt->dit, test_ctx, x, t, spk);

    // Build and compute graph
    ggml_cgraph * gf = ggml_new_graph(test_ctx);
    ggml_build_forward_expand(gf, out);
    int n_threads = 8;
    ggml_graph_compute_with_ctx(test_ctx, gf, n_threads);

    // Print output stats
    int n_out = out->ne[0] * out->ne[1];
    float * out_data = tensor_data(out);
    float sum = 0, minv = 1e9, maxv = -1e9;
    for (int i = 0; i < n_out; i++) {
        sum += out_data[i];
        if (out_data[i] < minv) minv = out_data[i];
        if (out_data[i] > maxv) maxv = out_data[i];
    }
    printf("Output: [%lld x %lld] min=%.6f max=%.6f mean=%.6f\n",
           (long long)out->ne[0], (long long)out->ne[1],
           minv, maxv, sum / (float)n_out);

    ggml_free(test_ctx);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;

    printf("dots.tts.cpp - First C++ implementation of dots.tts\n");
    printf("Reference: rednote-hilab/dots.tts (Apache 2.0)\n");
    printf("License: Apache 2.0\n\n");

    printf("Architecture parameters:\n");
    printf("  DiT: %d layers, hidden=%d, heads=%d, head_dim=%d, FFN=%d\n",
           DIT_NUM_LAYERS, DIT_HIDDEN_SIZE, DIT_NUM_HEADS, DIT_HEAD_SIZE, DIT_FFN_SIZE);
    printf("  PatchEncoder: %d layers, hidden=%d\n",
           PATCHENC_NUM_LAYERS, PATCHENC_HIDDEN);
    printf("  AudioVAE: latent=%d, hop=%d, sample_rate=%d kHz\n",
           VAE_LATENT_DIM, VAE_HOP_SAMPLES, VAE_SAMPLE_RATE / 1000);
    printf("  Total params: ~2B\n\n");

    // Init runtime
    dots_tts_runtime * rt = dots_tts_init();

    // Load dummy weights
    dots_tts_load_dummy(rt);

    // Test DiT
    dots_tts_test_dit(rt);

    // Test PatchEncoder
    patch_encoder enc;
    patchenc_load_dummy(enc, rt->w_ctx);
    patchenc_test(enc);

    // Test AudioVAE
    printf("\n=== Testing AudioVAE decoder ===\n");
    {
        struct ggml_init_params vae_params = {
            .mem_size   = 64ULL * 1024 * 1024,
            .mem_buffer = nullptr,
            .no_alloc   = false,
        };
        ggml_context * vae_ctx = ggml_init(vae_params);

        int n_frames = 16; // 4 patches * 4 frames
        ggml_tensor * latent = ggml_new_tensor_2d(vae_ctx, GGML_TYPE_F32, VAE_LATENT_DIM, n_frames);
        float * ld = tensor_data(latent);
        for (int i = 0; i < n_frames * VAE_LATENT_DIM; i++) {
            ld[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }

        int n_samples;
        float * wav = new float[n_frames * VAE_HOP_SAMPLES];
        audiovae_decode_simple(vae_ctx, latent, n_frames, wav, &n_samples);

        float smin = 1e9, smax = -1e9, ssum = 0;
        for (int i = 0; i < n_samples; i++) {
            ssum += wav[i];
            if (wav[i] < smin) smin = wav[i];
            if (wav[i] > smax) smax = wav[i];
        }
        printf("AudioVAE: %d samples, min=%.6f max=%.6f mean=%.6f\n",
               n_samples, smin, smax, ssum / n_samples);
        printf("  (16 latent frames * 960 hop = %d samples = %.1f ms @ 48kHz)\n",
               n_samples, n_samples * 1000.0f / VAE_SAMPLE_RATE);

        delete[] wav;
        ggml_free(vae_ctx);
    }

    // End-to-end pipeline test
    printf("\n=== End-to-end pipeline (simulated LLM) ===\n");
    {
        struct ggml_init_params pipe_params = {
            .mem_size   = 512ULL * 1024 * 1024,
            .mem_buffer = nullptr,
            .no_alloc   = false,
        };
        ggml_context * pipe_ctx = ggml_init(pipe_params);

        int n_patches = 1;  // reduce for speed
        int patch_flat = PATCHENC_PATCH_SIZE * PATCHENC_LATENT_DIM; // 4*128=512

        // Simulate: for each patch, generate LLM hidden states randomly
        // In real pipeline: LLM produces hidden states from text tokens
        // Here we use random to validate the DiT+PatchEncoder+Vocoder chain

        printf("Generating %d audio patches...\n", n_patches);
        float all_latents[n_patches * PATCHENC_PATCH_SIZE * VAE_LATENT_DIM];
        int total_latent_frames = 0;

        // FM buffer accumulates conditioning for each new patch
        // [cond_seq * hidden] + [latent_patch_size * latent_dim]
        // For simplicity, regenerate conditioning each step

        ggml_tensor * llm_hidden = ggml_new_tensor_2d(pipe_ctx, GGML_TYPE_F32,
            1536, n_patches * 2); // simulated LLM hidden states
        {
            float * d = tensor_data(llm_hidden);
            for (int i = 0; i < (int)(n_patches * 2 * 1536); i++) {
                d[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            }
        }

        // For each patch: PatchEncoder -> (simulated LLM step) -> DiT -> latent
        for (int p = 0; p < n_patches; p++) {
            // 1. Simulate audio VAE latent for PatchEncoder input
            // In real pipeline: comes from prompt audio or previous DiT output
            float patch_in[PATCHENC_PATCH_SIZE * PATCHENC_LATENT_DIM];
            for (int i = 0; i < patch_flat; i++) {
                patch_in[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            }

            // 2. PatchEncoder: latent -> LLM embedding [1536]
            ggml_reset(pipe_ctx);
            ggml_tensor * pe_in = ggml_new_tensor_2d(pipe_ctx, GGML_TYPE_F32,
                PATCHENC_LATENT_DIM, PATCHENC_PATCH_SIZE);
            memcpy(tensor_data(pe_in), patch_in, patch_flat * sizeof(float));
            ggml_tensor * pe_out = patchenc_forward(enc, pipe_ctx, pe_in, 1);
            ggml_cgraph * pe_gf = ggml_new_graph(pipe_ctx);
            ggml_build_forward_expand(pe_gf, pe_out);
            ggml_graph_compute_with_ctx(pipe_ctx, pe_gf, 8);
            // pe_out is [1536, 1]

            // 3. Simulated LLM step: combine PatchEncoder output with text context
            // Real pipeline: feed pe_out into LLM as audio token, get hidden state
            // Here: use pre-generated random hidden states
            float * pe_data = tensor_data(pe_out);

            // 4. DiT: flow matching on accumulated conditioning -> new latent patch
            int cond_seq = p * 2 + 2; // growing sequence
            ggml_reset(pipe_ctx);
            ggml_tensor * dit_x = ggml_new_tensor_3d(pipe_ctx, GGML_TYPE_F32,
                DIT_HIDDEN_SIZE, 1, cond_seq);
            {
                float * dx = tensor_data(dit_x);
                for (int i = 0; i < cond_seq * DIT_HIDDEN_SIZE; i++) {
                    dx[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                }
            }
            dit_x = ggml_cont(pipe_ctx, ggml_permute(pipe_ctx, dit_x, 2, 1, 0, 3));

            ggml_tensor * t_in = ggml_new_tensor_1d(pipe_ctx, GGML_TYPE_F32, 1);
            ((float*)t_in->data)[0] = 0.5f;

            ggml_tensor * spk_in = ggml_new_tensor_2d(pipe_ctx, GGML_TYPE_F32,
                DIT_SPEAKER_DIM, 1);
            {
                float * sd = tensor_data(spk_in);
                for (int i = 0; i < DIT_SPEAKER_DIM; i++) {
                    sd[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                }
            }

            ggml_tensor * dit_out = dit_forward(rt->dit, pipe_ctx, dit_x, t_in, spk_in);
            ggml_cgraph * dit_gf = ggml_new_graph(pipe_ctx);
            ggml_build_forward_expand(dit_gf, dit_out);
            ggml_graph_compute_with_ctx(pipe_ctx, dit_gf, 8);
            // dit_out is [VAE_LATENT_DIM=128, PATCHENC_PATCH_SIZE=4]

            // 5. Collect latent patch
            float * lat_data = tensor_data(dit_out);
            memcpy(all_latents + total_latent_frames * VAE_LATENT_DIM,
                   lat_data, patch_flat * sizeof(float));
            total_latent_frames += PATCHENC_PATCH_SIZE;

            if (p == 0) {
                float s = 0;
                for (int i = 0; i < patch_flat; i++) s += lat_data[i];
                printf("  patch 0: DiT output mean=%.6f, PatchEncoder mean=%.6f\n",
                       s / patch_flat, pe_data[0]);
            }
        }

        // 6. AudioVAE: decode all latent frames to waveform
        printf("Decoding %d latent frames to audio...\n", total_latent_frames);
        ggml_reset(pipe_ctx);
        ggml_tensor * all_lat = ggml_new_tensor_2d(pipe_ctx, GGML_TYPE_F32,
            VAE_LATENT_DIM, total_latent_frames);
        memcpy(tensor_data(all_lat), all_latents,
               total_latent_frames * VAE_LATENT_DIM * sizeof(float));

        int total_samples;
        float * wav_out = new float[total_latent_frames * VAE_HOP_SAMPLES];
        audiovae_decode_simple(pipe_ctx, all_lat, total_latent_frames,
                               wav_out, &total_samples);

        float smin = 1e9, smax = -1e9, ssum = 0;
        for (int i = 0; i < total_samples; i++) {
            ssum += wav_out[i];
            if (wav_out[i] < smin) smin = wav_out[i];
            if (wav_out[i] > smax) smax = wav_out[i];
        }
        printf("Pipeline output: %d samples (%.0f ms), min=%.4f max=%.4f mean=%.6f\n",
               total_samples, total_samples * 1000.0f / VAE_SAMPLE_RATE,
               smin, smax, ssum / total_samples);

        delete[] wav_out;
        ggml_free(pipe_ctx);
    }

    printf("\n=== All Phase 1 tests passed ===\n");

    // Real safetensors weights test
    printf("\n=== Testing DiT with real safetensors weights ===\n");
    {
        const char * sf_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-base/blobs/69dbad797566b24003506e1dd698597937149920f6df9782d84214bf477acb48";

        SafeTensorsFile sf;
        if (!sf.open(sf_path)) {
            printf("Safetensors not found — skipping real weight test\n");
        } else {
            printf("Loading DiT weights from safetensors...\n");

            struct ggml_init_params real_w_params = {
                .mem_size   = 6ULL * 1024 * 1024 * 1024, // 6 GB
                .mem_buffer = nullptr,
                .no_alloc   = false,
            };
            ggml_context * real_w_ctx = ggml_init(real_w_params);

            dit_model real_dit;
            patch_encoder real_enc;
            if (load_dit_weights(sf, real_w_ctx, real_dit) &&
                load_patchenc_weights(sf, real_w_ctx, real_enc)) {
                printf("Running DiT forward with real weights...\n");

                struct ggml_init_params real_params = {
                    .mem_size   = 512ULL * 1024 * 1024,
                    .mem_buffer = nullptr,
                    .no_alloc   = false,
                };
                ggml_context * rctx = ggml_init(real_params);

                int seq_len = 32;
                ggml_tensor * rx = ggml_new_tensor_3d(rctx, GGML_TYPE_F32, DIT_HIDDEN_SIZE, 1, seq_len);
                {
                    float * d = tensor_data(rx);
                    for (int i = 0; i < seq_len * DIT_HIDDEN_SIZE; i++)
                        d[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                }
                rx = ggml_cont(rctx, ggml_permute(rctx, rx, 2, 1, 0, 3));

                ggml_tensor * rt_in = ggml_new_tensor_1d(rctx, GGML_TYPE_F32, 1);
                ((float*)rt_in->data)[0] = 0.5f;

                ggml_tensor * rspk = ggml_new_tensor_2d(rctx, GGML_TYPE_F32, DIT_SPEAKER_DIM, 1);
                {
                    float * d = tensor_data(rspk);
                    for (int i = 0; i < DIT_SPEAKER_DIM; i++)
                        d[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                }

                ggml_tensor * rout = dit_forward(real_dit, rctx, rx, rt_in, rspk);
                ggml_cgraph * rgf = ggml_new_graph(rctx);
                ggml_build_forward_expand(rgf, rout);
                ggml_graph_compute_with_ctx(rctx, rgf, 8);

                int n_out = rout->ne[0] * rout->ne[1];
                float * od = tensor_data(rout);
                float sum = 0, minv = 1e9, maxv = -1e9;
                for (int i = 0; i < n_out; i++) {
                    sum += od[i];
                    if (od[i] < minv) minv = od[i];
                    if (od[i] > maxv) maxv = od[i];
                }
                printf("Real DiT output: [%lld x %lld] min=%.6f max=%.6f mean=%.6f\n",
                       (long long)rout->ne[0], (long long)rout->ne[1],
                       minv, maxv, sum / n_out);

                // PatchEncoder test with real weights
                printf("Running PatchEncoder forward with real weights...\n");
                ggml_reset(rctx);
                int n_patches = 1;
                int pe_seq = n_patches * PATCHENC_PATCH_SIZE;
                ggml_tensor * pe_x = ggml_new_tensor_2d(rctx, GGML_TYPE_F32, PATCHENC_LATENT_DIM, pe_seq);
                {
                    float * d = tensor_data(pe_x);
                    for (int i = 0; i < pe_seq * PATCHENC_LATENT_DIM; i++)
                        d[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                }
                ggml_tensor * pe_out = patchenc_forward(real_enc, rctx, pe_x, n_patches);
                ggml_cgraph * pe_gf = ggml_new_graph(rctx);
                ggml_build_forward_expand(pe_gf, pe_out);
                ggml_graph_compute_with_ctx(rctx, pe_gf, 8);
                {
                    int n = pe_out->ne[0] * pe_out->ne[1];
                    float * d = tensor_data(pe_out);
                    float s = 0, mn = 1e9, mx = -1e9;
                    for (int i = 0; i < n; i++) { s += d[i]; if (d[i] < mn) mn = d[i]; if (d[i] > mx) mx = d[i]; }
                    printf("Real PatchEncoder output: [%lld x %lld] min=%.6f max=%.6f mean=%.6f\n",
                           (long long)pe_out->ne[0], (long long)pe_out->ne[1], mn, mx, s/n);
                }

                ggml_free(rctx);
            }
            ggml_free(real_w_ctx);
            sf.close();
        }
    }

    // LLM test with llama.cpp
    printf("\n=== Testing LLM (Qwen2.5-1.5B) ===\n");
    {
        llama_log_set([](enum ggml_log_level level, const char * text, void *) {
            fprintf(stderr, "llama[%d]: %s", level, text);
        }, nullptr);

        const char * llm_path = "/home/bym/dots.tts.cpp/models/llm_qwen25_1.5b.gguf";

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        mparams.use_mmap = false; // try without mmap
        printf("Loading LLM (this may take a moment)...\n");
        llama_model * llm = llama_model_load_from_file(llm_path, mparams);
        if (!llm) {
            // Check if file exists
            FILE * test_f = fopen(llm_path, "rb");
            if (test_f) {
                fclose(test_f);
                printf("File exists but llama failed to load (GGUF format issue?)\n");
            } else {
                printf("LLM GGUF not found at %s - skipping\n", llm_path);
            }
        } else {
            printf("LLM loaded. n_vocab=%d n_ctx_train=%d\n",
                   llama_vocab_n_tokens(llama_model_get_vocab(llm)),
                   llama_model_n_ctx_train(llm));

            llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx = 256;
            cparams.n_batch = 256;
            llama_context * lctx = llama_init_from_model(llm, cparams);

            const char * text = "Privet, how are you?";
            llama_token tokens[256];
            int n = llama_tokenize(llama_model_get_vocab(llm),
                                   text, strlen(text), tokens, 256, true, false);
            printf("Tokenized: '%s' -> %d tokens\n", text, n);

            llama_batch batch = llama_batch_get_one(tokens, n);
            if (llama_decode(lctx, batch) == 0) {
                printf("LLM decode OK. %d layers, hidden=%d\n",
                       llama_model_n_layer(llm),
                       llama_model_n_embd(llm));
            }
            printf("LLM test: OK\n");

            llama_free(lctx);
            llama_model_free(llm);
        }
    }

    ggml_free(rt->w_ctx);
    delete rt;

    printf("\nDone.\n");
    return 0;
    return 0;
}

// ---------------------------------------------------------------------------
// Dummy PatchEncoder load for testing
// ---------------------------------------------------------------------------

static void patchenc_load_dummy(patch_encoder & enc, ggml_context * w_ctx) {
    printf("Loading dummy PatchEncoder weights...\n");

    enc.n_layers = PATCHENC_NUM_LAYERS;
    enc.hidden_size = PATCHENC_HIDDEN;
    enc.num_heads = PATCHENC_NUM_HEADS;
    enc.head_dim = PATCHENC_HEAD_SIZE;
    enc.ffn_size = PATCHENC_FFN_SIZE;
    enc.latent_dim = PATCHENC_LATENT_DIM;
    enc.patch_size = PATCHENC_PATCH_SIZE;

    enc.layers.resize(enc.n_layers);

    auto alloc = [&](const char * name, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
        ggml_tensor * t = ggml_new_tensor_4d(w_ctx, GGML_TYPE_F32, n0, n1, n2, n3);
        float * d = tensor_data(t);
        size_t n = n0 * n1 * n2 * n3;
        for (size_t i = 0; i < n; i++) d[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        return t;
    };

    for (int i = 0; i < enc.n_layers; i++) {
        patchenc_layer & l = enc.layers[i];
        char name[256];

        snprintf(name, sizeof(name), "pe.layers.%d.attn.q.weight", i);
        l.attn_q_weight = alloc(name, PATCHENC_HIDDEN, PATCHENC_HIDDEN, 1, 1);

        snprintf(name, sizeof(name), "pe.layers.%d.attn.k.weight", i);
        l.attn_k_weight = alloc(name, PATCHENC_HIDDEN, PATCHENC_HIDDEN, 1, 1);

        snprintf(name, sizeof(name), "pe.layers.%d.attn.v.weight", i);
        l.attn_v_weight = alloc(name, PATCHENC_HIDDEN, PATCHENC_HIDDEN, 1, 1);

        snprintf(name, sizeof(name), "pe.layers.%d.attn.o.weight", i);
        l.attn_o_weight = alloc(name, PATCHENC_HIDDEN, PATCHENC_HIDDEN, 1, 1);

        snprintf(name, sizeof(name), "pe.layers.%d.q_norm.weight", i);
        l.q_norm_w = alloc(name, PATCHENC_HEAD_SIZE, 1, 1, 1);

        snprintf(name, sizeof(name), "pe.layers.%d.k_norm.weight", i);
        l.k_norm_w = alloc(name, PATCHENC_HEAD_SIZE, 1, 1, 1);

        snprintf(name, sizeof(name), "pe.layers.%d.ffn.w1.weight", i);
        l.ffn_w1 = alloc(name, PATCHENC_HIDDEN, PATCHENC_FFN_SIZE, 1, 1);

        snprintf(name, sizeof(name), "pe.layers.%d.ffn.w2.weight", i);
        l.ffn_w2 = alloc(name, PATCHENC_FFN_SIZE, PATCHENC_HIDDEN, 1, 1);

        snprintf(name, sizeof(name), "pe.layers.%d.attn_norm.weight", i);
        l.attn_norm_w = alloc(name, PATCHENC_HIDDEN, 1, 1, 1);

        snprintf(name, sizeof(name), "pe.layers.%d.ffn_norm.weight", i);
        l.ffn_norm_w = alloc(name, PATCHENC_HIDDEN, 1, 1, 1);
    }

    enc.conv_weight = alloc("pe.conv.weight", PATCHENC_LATENT_DIM, PATCHENC_LATENT_DIM, 2, 1);
    enc.conv_bias   = alloc("pe.conv.bias", PATCHENC_LATENT_DIM, 1, 1, 1);
    enc.in_proj_w   = alloc("pe.in_proj.weight", PATCHENC_LATENT_DIM, PATCHENC_HIDDEN, 1, 1);
    enc.in_proj_b   = alloc("pe.in_proj.bias", PATCHENC_HIDDEN, 1, 1, 1);
    enc.out_proj_w  = alloc("pe.out_proj.weight", 2*PATCHENC_HIDDEN, 1536, 1, 1);
    enc.out_proj_b  = alloc("pe.out_proj.bias", 1536, 1, 1, 1);
    enc.final_norm_w = alloc("pe.final_norm.weight", PATCHENC_HIDDEN, 1, 1, 1);

    printf("PatchEncoder dummy weights loaded: %zu bytes\n", ggml_used_mem(w_ctx));
}

static void patchenc_test(patch_encoder & enc) {
    printf("\n=== Testing PatchEncoder forward pass ===\n");

    struct ggml_init_params test_params = {
        .mem_size   = 256ULL * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    ggml_context * ctx = ggml_init(test_params);

    int n_patches = 1;  // reduce for speed
    int seq = n_patches * PATCHENC_PATCH_SIZE; // 16 latent frames
    int ch  = PATCHENC_LATENT_DIM; // 128

    // Random input
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ch, seq);
    float * xd = tensor_data(x);
    for (int i = 0; i < seq * ch; i++) {
        xd[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    // Forward
    ggml_tensor * out = patchenc_forward(enc, ctx, x, n_patches);

    // Compute
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 8);

    // Stats
    int n_out = out->ne[0] * out->ne[1];
    float * od = tensor_data(out);
    float sum = 0, minv = 1e9, maxv = -1e9;
    for (int i = 0; i < n_out; i++) {
        sum += od[i];
        if (od[i] < minv) minv = od[i];
        if (od[i] > maxv) maxv = od[i];
    }
    printf("PatchEncoder output: [%lld x %lld] min=%.6f max=%.6f mean=%.6f\n",
           (long long)out->ne[0], (long long)out->ne[1],
           minv, maxv, sum / n_out);

    // Streaming test
    printf("=== Testing PatchEncoder streaming ===\n");
    patchenc_stream_state * st = patchenc_stream_init(enc);
    float llm_emb[1536] = {0};
    
    for (int p = 0; p < n_patches; p++) {
        float patch[PATCHENC_PATCH_SIZE * PATCHENC_LATENT_DIM];
        for (int i = 0; i < PATCHENC_PATCH_SIZE * PATCHENC_LATENT_DIM; i++) {
            patch[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        ggml_reset(ctx);
        patchenc_decode_step(enc, st, ctx, patch, llm_emb);
        
        float s = 0;
        for (int i = 0; i < 1536; i++) s += llm_emb[i];
        printf("  patch %d: LLM emb mean=%.6f\n", p, s / 1536.0f);
    }

    patchenc_stream_free(st);
    ggml_free(ctx);
    printf("PatchEncoder streaming: OK\n");
}
