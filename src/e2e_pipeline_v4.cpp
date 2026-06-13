// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// dots.tts.cpp - E2E TTS pipeline v4 — matching Python _generate_latents_stream + _decode
// Key differences from v3:
//   FM buffer: interleaved [hidden, latent_patch(4 pos), hidden, latent_patch(4 pos), ...]
//   Noise generated inside solver, NOT placed in fm_sequence
//   Denormalize latents before PatchEncoder (matching io_helper.denormalize)
//   PatchEncoder streaming state (conv_tail)
//   Attention mask + position IDs for DiT
//   EOS detection via eos_proj

#include "dots_tts.h"
#include "dit.h"
#include "dit_manual.h"
#include "audiovae.h"
#include "bigvgan_cpp.h"
#include "patchenc.h"
#include "safetensors.h"
#include "gpt2_bpe_tokenizer.h"
#include "speaker_embedding.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "llm_manual.h"
#include "mt19937.h"
#include "zig_normal.h"
#include "noise_data_pipeline.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <ctime>
#include <thread>
#include <algorithm>

// Constants matching Python core.py
static const int HIDDEN_PATCH_SIZE = 1;   // core.hidden_patch_size
static const int LATENT_PATCH_SIZE = 4;   // config.patch_size
static const int PATCH_STRIDE = HIDDEN_PATCH_SIZE + LATENT_PATCH_SIZE;  // 5
static const int NFE = 10;

// VAE statistics (from latent_stats.pt)
static const float VAE_MEAN[128] = {-0.9025,-0.6183,-0.0263,-0.0685,-0.1783,-0.5570,-0.6110,-0.3879,-0.3249,-0.2662,0.3478,0.0372,-0.5750,0.0344,-0.5378,0.2272,-0.2407,0.2857,-0.4126,0.0592,-0.3421,-0.5783,0.0266,-0.0334,-0.0829,-0.3520,0.1200,-0.4391,0.3389,-0.4224,-0.3515,-0.0558,0.1107,-0.0028,0.6533,-0.0096,0.3998,0.0979,0.0710,0.1544,-0.1871,0.1712,0.2767,-0.3803,-0.2228,-0.3248,0.0888,0.0382,-0.2751,-0.1028,-0.3058,0.2343,0.0244,-0.2566,-0.0230,0.7069,-0.4051,0.5665,-0.1977,-0.4928,0.1987,-0.3744,0.2669,-0.5638,-0.0391,0.1564,-0.1035,-0.4483,0.6975,0.2761,-0.2020,0.2640,0.0682,0.1176,0.7706,-1.3600,0.0722,-0.1375,0.0337,0.1764,-0.2927,0.3615,-0.1335,0.3452,0.0590,-0.2773,0.0653,0.3692,-0.3961,-0.0325,0.1038,-0.2431,-0.0402,0.3810,-0.0239,-0.3426,-0.0924,-0.2611,0.1166,0.4047,0.1759,-0.6634,-0.0081,2.0526,0.0439,-0.1364,0.1873,-0.0263,-0.7545,0.5526,-0.0752,-0.3717,0.1637,0.7791,0.7361,-0.3620,-0.3784,0.4654,-0.2694,-0.2115,0.3204,0.2995,-0.7347,0.4701,0.1141,0.6925,0.7140,-0.0032};
static const float VAE_STD[128] = {3.0316,2.8055,1.7751,1.7465,2.0646,1.9869,2.2711,2.0302,1.8684,2.3117,2.0107,1.9570,2.8696,2.3088,2.0507,2.1352,2.6940,1.7395,1.9994,1.8537,2.0860,2.1602,1.9088,2.6236,1.9415,3.6039,1.7381,1.8368,2.7282,2.0870,2.4967,2.4496,1.6383,1.6716,4.6338,2.4076,3.9610,1.8117,1.8016,1.8020,2.0124,1.7533,1.9128,2.2415,2.0514,1.9496,1.8469,2.1641,1.9519,1.9785,1.7530,1.6008,1.9471,2.6634,2.2622,2.4504,1.9333,2.8137,2.0679,2.1296,1.8197,2.4677,2.2560,2.2627,1.9336,1.6912,2.1193,1.7280,3.0617,1.5928,1.7773,1.8976,1.6118,1.5717,2.4272,3.1342,2.3934,1.8039,1.8262,1.7740,3.5085,1.9350,2.2693,1.8149,1.5992,1.8857,2.5523,2.2707,1.8558,1.6974,1.9261,1.8115,1.5260,1.8547,1.4196,1.7291,1.6253,2.0238,2.5385,2.1883,1.6951,2.5633,2.0287,13.0302,2.1418,1.6957,4.7318,2.3544,2.6665,2.8945,1.7005,6.3084,2.2875,2.7594,2.3732,1.7797,1.9751,2.6090,1.6278,2.5487,2.0741,2.5441,2.0957,1.6018,1.4397,3.2050,1.8041,2.3601};
static const float VAE_VAR[128] = {9.1906,7.8708,3.1510,3.0503,4.2626,3.9478,5.1579,4.1224,3.4909,5.3440,4.0429,3.8299,8.2341,5.3306,4.2054,4.5591,7.2576,3.0259,3.9969,3.4362,4.3514,4.6665,3.6435,6.8833,3.7694,12.9868,3.0210,3.3738,7.4431,4.3556,6.2335,6.0006,2.6840,2.7942,21.4761,5.7965,15.6895,3.2823,3.2458,3.2472,4.0498,3.0741,3.6588,5.0243,4.2082,3.8014,3.4110,4.6833,3.8099,3.9145,3.0730,2.5626,3.7912,7.0937,5.1176,6.0045,3.7376,7.9169,4.2762,4.5352,3.3114,6.0895,5.0895,5.1185,3.7388,2.8602,4.4914,2.9874,9.3740,2.5370,3.1588,3.6010,2.5979,2.4702,5.8908,9.8208,5.7284,3.2541,3.3350,3.1476,12.3096,3.7442,5.1494,3.2939,2.5574,3.5559,6.5142,5.1541,3.4440,2.8816,3.7099,3.2815,2.3292,3.4399,2.0157,2.9916,2.6416,4.0958,6.4440,4.7887,2.8734,6.5685,4.1157,169.7849,4.5873,2.8754,22.3899,5.5432,7.1102,8.3781,2.8917,39.7959,5.2330,7.6143,5.6321,3.1673,3.9010,6.8069,2.6497,6.4959,4.3019,6.4736,4.3920,2.5658,2.0727,10.2720,3.2550,5.5701};

static const int N_SPEAKER = 0;  // no speaker embedding

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);
bool load_patchenc_weights(SafeTensorsFile & sf, ggml_context * w_ctx, patch_encoder & enc);
bool extract_embeddings(const char * gguf_path, const char * out_path);
static double now_ms() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1000.0 + ts.tv_nsec/1000000.0; }

// compute_cond: time embedding + speaker projection → cond[1024]
static void compute_cond(dit_model & m, float t_val, const float * spk_emb, float speaker_scale, float * cond) {
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
     for(int i=0;i<1024;i++){float x=(t[i]-mean)*istd;if(lw)x*=lw[i];if(lb)x+=lb[i];cond[i]+=x*speaker_scale;}}
}

// Denormalize latents: z * std + mean (matching io_helper.denormalize)
static void denormalize_latents(float * z, int patch_size, int latent_dim) {
    for (int p = 0; p < patch_size; p++)
        for (int c = 0; c < latent_dim; c++)
            z[p * latent_dim + c] = z[p * latent_dim + c] * VAE_STD[c] + VAE_MEAN[c];
}

// Apply latent_proj: [patch_size, latent_dim] → [patch_size, DIT_HIDDEN_SIZE]
static void apply_latent_proj(dit_model & dit, const float * latents, int patch_size, int latent_dim, float * out) {
    float * lw = tensor_data(dit.latent_proj_w);
    float * lb = dit.latent_proj_b ? tensor_data(dit.latent_proj_b) : nullptr;
    for (int p = 0; p < patch_size; p++) {
        for (int o = 0; o < DIT_HIDDEN_SIZE; o++) {
            float s = lb ? lb[o] : 0.0f;
            for (int i = 0; i < latent_dim; i++) s += lw[o * latent_dim + i] * latents[p * latent_dim + i];
            out[p * DIT_HIDDEN_SIZE + o] = s;
        }
    }
}

// Apply hidden_proj: [llm_hidden] → [DIT_HIDDEN_SIZE]
static void apply_hidden_proj(dit_model & dit, const float * hidden, int llm_dim, float * out) {
    float * hw = tensor_data(dit.hidden_proj_w);
    float * hb = dit.hidden_proj_b ? tensor_data(dit.hidden_proj_b) : nullptr;
    for (int o = 0; o < DIT_HIDDEN_SIZE; o++) {
        float s = hb ? hb[o] : 0.0f;
        for (int i = 0; i < llm_dim; i++) s += hw[o * llm_dim + i] * hidden[i];
        out[o] = s;
    }
}

// Apply coord_proj: [latent_dim] → [DIT_HIDDEN_SIZE]
static void apply_coord_proj(dit_model & dit, const float * noise_vec, int latent_dim, float * out) {
    float * cw = tensor_data(dit.coord_proj_w);
    float * cb = dit.coord_proj_b ? tensor_data(dit.coord_proj_b) : nullptr;
    for (int o = 0; o < DIT_HIDDEN_SIZE; o++) {
        float s = cb ? cb[o] : 0.0f;
        for (int i = 0; i < latent_dim; i++) s += cw[o * latent_dim + i] * noise_vec[i];
        out[o] = s;
    }
}

// Build attention mask for FM DiT (matching _build_fm_attn_mask)
// attn_mask: [total_len * total_len] bool (true = attend)
// fm_seq_len: current history length
// patch_size: latent patch size (noise positions at end)
// hidden_patch_size: always 1
static void build_fm_attn_mask(bool * mask, int total_len, int fm_seq_len, int patch_size, int hidden_patch_size) {
    int latent_start = total_len - patch_size;
    int block_start = fm_seq_len > hidden_patch_size ? fm_seq_len - hidden_patch_size : 0;
    
    memset(mask, 0, total_len * total_len * sizeof(bool));
    
    // Block [0:block_start] → causal
    for (int i = 0; i < block_start; i++)
        for (int j = 0; j <= i; j++)
            mask[i * total_len + j] = true;
    
    // Block [block_start:fm_seq_len] → all history + latent queries
    for (int i = block_start; i < fm_seq_len; i++) {
        for (int j = 0; j < fm_seq_len; j++) mask[i * total_len + j] = true;
        for (int j = latent_start; j < total_len; j++) mask[i * total_len + j] = true;
    }
    
    // Latent queries → all history + self (bidirectional)
    for (int i = latent_start; i < total_len; i++) {
        for (int j = 0; j < fm_seq_len; j++) mask[i * total_len + j] = true;
        for (int j = latent_start; j < total_len; j++) mask[i * total_len + j] = true;
    }
    
    // Padding positions between fm_seq_len and latent_start: self-attention only
    for (int i = fm_seq_len; i < latent_start; i++)
        mask[i * total_len + i] = true;
}

// Build position IDs (matching _build_fm_pos_ids)
static void build_fm_pos_ids(float * pos_ids, int total_len, int fm_seq_len, int patch_size) {
    int latent_start = total_len - patch_size;
    for (int i = 0; i < fm_seq_len; i++) pos_ids[i] = (float)i;
    for (int i = latent_start; i < total_len; i++) pos_ids[i] = (float)(fm_seq_len + i - latent_start);
    // Other positions remain 0 (padding)
    for (int i = fm_seq_len; i < latent_start; i++) pos_ids[i] = 0.0f;
}

// Run DiT on prepared sequence (matching fm_solver_step internals)
// input_sequence: total_len x DIT_HIDDEN_SIZE (interleaved history, noise at last patch_size positions)
// cond: [1024] time conditioning
// Returns velocity at latent_start positions [patch_size x latent_dim]
static void dit_solver_step(dit_model & dit, ggml_context * w_ctx, const float * input_sequence, 
    int total_len, int patch_size, const float * cond, float * velocity) {
    
    int latent_start = total_len - patch_size;
    
    // Input layer: project input_sequence
    float * h = new float[total_len * DIT_HIDDEN_SIZE];
    float * iw = tensor_data(dit.input_layer_w);
    float * ib = dit.input_layer_b ? tensor_data(dit.input_layer_b) : nullptr;
    for (int ti = 0; ti < total_len; ti++)
        manual_linear(h + ti * DIT_HIDDEN_SIZE, input_sequence + ti * DIT_HIDDEN_SIZE, iw, ib, DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE);
    
    // DiT blocks
    float * bo = new float[total_len * DIT_HIDDEN_SIZE];
    for (int i = 0; i < dit.n_layers; i++) {
        manual_dit_block(h, cond, dit.layers[i], bo, total_len);
        float * tmp = h; h = bo; bo = tmp;
    }
    
    // Out adaLN + LayerNorm + Out projection
    float cs2[1024];
    for (int i = 0; i < DIT_HIDDEN_SIZE; i++) cs2[i] = cond[i] / (1.0f + expf(-cond[i]));
    
    float mod_raw[2 * DIT_HIDDEN_SIZE];
    float * aw = tensor_data(dit.out_adaln_w);
    float * ab = dit.out_adaln_b ? tensor_data(dit.out_adaln_b) : nullptr;
    for (int o = 0; o < 2 * DIT_HIDDEN_SIZE; o++) {
        float s = ab ? ab[o] : 0.0f;
        for (int i = 0; i < DIT_HIDDEN_SIZE; i++) s += aw[o * DIT_HIDDEN_SIZE + i] * cs2[i];
        mod_raw[o] = s;
    }
    float * dit_shift = mod_raw, * dit_scale = mod_raw + DIT_HIDDEN_SIZE;
    
    float * dit_ln = new float[total_len * DIT_HIDDEN_SIZE];
    for (int ti = 0; ti < total_len; ti++) {
        manual_layernorm(dit_ln + ti * DIT_HIDDEN_SIZE, h + ti * DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE);
        for (int j = 0; j < DIT_HIDDEN_SIZE; j++)
            dit_ln[ti * DIT_HIDDEN_SIZE + j] = dit_ln[ti * DIT_HIDDEN_SIZE + j] * (1.0f + dit_scale[j]) + dit_shift[j];
    }
    
    // Out projection to latent dim
    float * ow = tensor_data(dit.out_proj_w);
    float * ob = dit.out_proj_b ? tensor_data(dit.out_proj_b) : nullptr;
    float * out_all = new float[total_len * VAE_LATENT_DIM];
    for (int ti = 0; ti < total_len; ti++)
        manual_linear(out_all + ti * VAE_LATENT_DIM, dit_ln + ti * DIT_HIDDEN_SIZE, ow, ob, DIT_HIDDEN_SIZE, VAE_LATENT_DIM);
    
    // Extract velocity at latent_start positions
    for (int p = 0; p < patch_size; p++)
        for (int c = 0; c < VAE_LATENT_DIM; c++)
            velocity[p * VAE_LATENT_DIM + c] = out_all[(latent_start + p) * VAE_LATENT_DIM + c];
    
    delete[] h; delete[] bo; delete[] dit_ln; delete[] out_all;
}

int main(int argc, char ** argv) {
    setbuf(stdout, NULL);
    double t1 = now_ms(), t_load = 0, t_gen_start = 0;

    const char * text = "Hello world", * out_path = "output.wav";
    const char * model_dir = getenv("DOTS_TTS_MODEL") ? getenv("DOTS_TTS_MODEL") : "models";
    const char * model_sf_path = nullptr;
    int n_calls_max = 8; uint32_t force_seed = 0;
    bool use_force_seed = false, dump_debug = false, benchmark = false;
    float cfg_scale = 1.0f, eos_threshold = 0.8f;
    bool fast_mode = false;  // skip CFG for speed
    int speaker_scale_int = 0;  // 0=no speaker

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i+1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--model") == 0 && i+1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--length") == 0 && i+1 < argc) n_calls_max = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) { force_seed = (uint32_t)atoi(argv[++i]); use_force_seed = true; }
        else if (strcmp(argv[i], "--speaker") == 0 && i+1 < argc) { speaker_scale_int = 1; }
        else if (strcmp(argv[i], "--dump") == 0) dump_debug = true;
        else if (strcmp(argv[i], "--cfg") == 0 && i+1 < argc) cfg_scale = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--benchmark") == 0) benchmark = true;
        else if (strcmp(argv[i], "--fast") == 0) fast_mode = true;
        else if (argv[i][0] != '-') text = argv[i];
    }

    if (n_calls_max > 100) n_calls_max = 100;
    int n_threads = std::thread::hardware_concurrency();
    if (n_threads < 1) n_threads = 1;

    const int latent_dim = VAE_LATENT_DIM, patch_size = LATENT_PATCH_SIZE, llm_dim = 1536;
    const int patch_flat = patch_size * latent_dim;  // 512
    const float speaker_scale = 1.5f;
    const float dt = 1.0f / NFE;

    printf("dots.tts.cpp v4 — matching Python _generate_latents_stream\n");
    printf("  Text: '%s'  Max calls: %d  Threads: %d\n\n", text, n_calls_max, n_threads);

    // Resolve model path
    std::string sf_path = std::string(model_dir) + "/model.safetensors";
    { FILE * f = fopen(sf_path.c_str(), "rb"); if (!f) {
        sf_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/e3520f75254d0020a0406db31c51a79d00d22d55/model.safetensors";
        f = fopen(sf_path.c_str(), "rb"); if (f) fclose(f);
    } else fclose(f); }
    model_sf_path = sf_path.c_str();

    // === Tokenization ===
    printf("[1] Tokenizing...\n");
    GPT2BPETokenizer bpe_tok;
    if (!bpe_tok.load(model_dir)) { fprintf(stderr, "FAILED: tokenizer\n"); return 1; }
    
    // Build generation schedule (matching build_generation_schedule)
    // Text template: "[文本]Hello world[文本对应语音]<|audio_gen_start|>"
    std::string template_text = "[文本]" + std::string(text) + "[文本对应语音]<|audio_gen_start|>";
    auto token_ids_vec = bpe_tok.encode(template_text.c_str());
    int n_text_tok = (int)token_ids_vec.size();
    if (n_text_tok == 0) { fprintf(stderr, "Empty tokenization\n"); return 1; }
    
    // Audio placeholder token: use token ID 151669 (matching Python's audio_span_token_ids)
    // Actually, the last token in the text is <|audio_gen_start|> with ID 151668
    // Audio placeholders start after this: ID 151669
    int audio_placeholder_id = 151669;
    int max_audio_spans = 500;  // matching max_audio_tokens=500
    int n_schedule = n_text_tok + max_audio_spans;  // 11 + 500 = 511
    
    int32_t * schedule = new int32_t[n_schedule];
    memcpy(schedule, token_ids_vec.data(), n_text_tok * sizeof(int32_t));
    for (int i = n_text_tok; i < n_schedule; i++) schedule[i] = audio_placeholder_id;
    
    printf("  Schedule: %d tokens (%d text + %d audio placeholders)\n", n_schedule, n_text_tok, max_audio_spans);
    
    uint32_t hash = use_force_seed ? force_seed : 5381;
    if (!use_force_seed) for (int i = 0; i < n_text_tok; i++) hash = ((hash << 5) + hash) + schedule[i];
    srand(hash);
    MT19937 mt; mt.seed(hash);
    printf("  Seed: %u\n", hash);

    // === LM: Prefill text tokens ===
    printf("[2] LM prefill (%d text tokens)...\n", n_text_tok);
    SafeTensorsFile sf_llm; sf_llm.open(model_sf_path);
    LLMWeights llm_w; memset(&llm_w, 0, sizeof(llm_w));
    if (!load_llm_weights(sf_llm, llm_w, 28)) { fprintf(stderr, "FAILED: LLM weights\n"); return 1; }
    sf_llm.close();

    double tlm = now_ms();
    float * lm_hiddens = new float[n_text_tok * llm_dim];
    LLMKVCache kv_cache; memset(&kv_cache, 0, sizeof(kv_cache));
    llm_kv_cache_init(llm_w, schedule, n_text_tok, lm_hiddens, kv_cache);
    printf("  LM prefill: %d ms\n", (int)(now_ms()-tlm));
    
    { float r = 0; for (int i = 0; i < n_text_tok * llm_dim; i++) r += lm_hiddens[i] * lm_hiddens[i];
      printf("  LM hiddens RMS=%.4f\n", sqrtf(r / (n_text_tok * llm_dim))); }

    // === Load DiT + PatchEncoder ===
    printf("[3] Loading DiT + PatchEncoder...\n");
    SafeTensorsFile sf; sf.open(model_sf_path);
    ggml_init_params wp_dit = { .mem_size = 3ULL*1024*1024*1024 };
    ggml_context * w_ctx_dit = ggml_init(wp_dit);
    dit_model dit; load_dit_weights(sf, w_ctx_dit, dit); sf.close();

    SafeTensorsFile sf2; sf2.open(model_sf_path);
    ggml_init_params wp_pe = { .mem_size = 3ULL*1024*1024*1024 };
    ggml_context * w_ctx_pe = ggml_init(wp_pe);
    patch_encoder pe; load_patchenc_weights(sf2, w_ctx_pe, pe); sf2.close();
    printf("  DiT: %d layers, PE: loaded\n", dit.n_layers);

    // === Initial FM buffer: hidden_proj(last_text_hidden) ===
    printf("[4] Building initial FM buffer...\n");
    float spk_emb[512] = {0};
    
    int fm_capacity = PATCH_STRIDE * n_calls_max + 1;  // enough for max calls
    float * fm_buffer = new float[fm_capacity * DIT_HIDDEN_SIZE];
    float * fm_cfg_buffer = new float[fm_capacity * DIT_HIDDEN_SIZE];
    int fm_seq_len = 0;
    
    // Append initial hidden (hidden_proj of last text token)
    float hidden_0[DIT_HIDDEN_SIZE];
    float * last_hidden = lm_hiddens + (n_text_tok - 1) * llm_dim;
    apply_hidden_proj(dit, last_hidden, llm_dim, hidden_0);
    memcpy(fm_buffer, hidden_0, DIT_HIDDEN_SIZE * sizeof(float));
    
    // CFG buffer: null version (hidden_proj of zeros)
    float zero_hidden[1536] = {0};
    apply_hidden_proj(dit, zero_hidden, llm_dim, fm_cfg_buffer);
    fm_seq_len = 1;
    
    { float r = 0; for (int i = 0; i < DIT_HIDDEN_SIZE; i++) r += hidden_0[i] * hidden_0[i];
      printf("  FM buffer[0]: hidden RMS=%.4f\n", sqrtf(r / DIT_HIDDEN_SIZE)); }

    // === Decode loop ===
    printf("[5] Decode loop (max %d calls)...\n", n_calls_max);
    
    // Streaming state
    ggml_context * gctx_p = nullptr;
    float * all_latents = new float[n_calls_max * patch_flat];
    float * all_latents_denorm = new float[n_calls_max * patch_flat];
    int total_frames = 0;
    
    // PatchEncoder streaming state (conv_tail for causal conv)
    // ds_proj: Conv1d(in_dim=128, out_dim=128, kernel=2, stride=2, causal)
    // left_padding = kernel_size - stride = 2 - 2 = 0? Actually causal conv has left_padding = kernel-1 = 1
    int conv_tail_size = 128;  // in_channels
    float * conv_tail = new float[conv_tail_size]();
    
    // EOS
    bool end_flag = false;
    
    for (int call = 0; call < n_calls_max && !end_flag; call++) {
        double tcall = now_ms();
        printf("  Call %d/%d: ", call+1, n_calls_max);
        
        // === Prepare DiT input ===
        int history_bucket = fm_seq_len;  // no rounding when optimize is disabled
        int total_len = history_bucket + patch_size;
        
        // Build attn_mask and pos_ids
        bool * attn_mask = new bool[total_len * total_len];
        float * pos_ids = new float[total_len];
        build_fm_attn_mask(attn_mask, total_len, fm_seq_len, patch_size, HIDDEN_PATCH_SIZE);
        build_fm_pos_ids(pos_ids, total_len, fm_seq_len, patch_size);
        
        // Allocate DiT input sequence
        float * dit_input = new float[total_len * DIT_HIDDEN_SIZE]();
        float * dit_cfg_input = new float[total_len * DIT_HIDDEN_SIZE]();
        memcpy(dit_input, fm_buffer, fm_seq_len * DIT_HIDDEN_SIZE * sizeof(float));
        memcpy(dit_cfg_input, fm_cfg_buffer, fm_seq_len * DIT_HIDDEN_SIZE * sizeof(float));
        
        // === Euler solver ===
        float * z_t = new float[patch_flat];
        float * v_t = new float[patch_flat];
        
        // Initial noise
        if (call < 30) memcpy(z_t, NOISE_PIPELINE[call], patch_flat * sizeof(float));
        else for (int i = 0; i < patch_flat; i++) { float z = zig_normal(mt); if(z>5)z=5; if(z<-5)z=-5; z_t[i]=z; }
        
        for (int step = 0; step < NFE; step++) {
            float t = (float)step * dt;
            
            // Apply coord_proj to noise and place at end of sequence
            for (int p = 0; p < patch_size; p++) {
                apply_coord_proj(dit, z_t + p * latent_dim, latent_dim,
                    dit_input + (total_len - patch_size + p) * DIT_HIDDEN_SIZE);
                // For CFG buffer: same noise (noise is unconditional)
                memcpy(dit_cfg_input + (total_len - patch_size + p) * DIT_HIDDEN_SIZE,
                       dit_input + (total_len - patch_size + p) * DIT_HIDDEN_SIZE,
                       DIT_HIDDEN_SIZE * sizeof(float));
            }
            
            // Compute time conditioning
            float cond[1024];
            compute_cond(dit, t, nullptr, speaker_scale, cond);
            
            // Dump first step for debugging
            if (step == 0 && call == 0) {
                { FILE * f = fopen("debug_v4/cpp_dit_input.bin", "wb"); fwrite(dit_input, sizeof(float), total_len * DIT_HIDDEN_SIZE, f); fclose(f); }
                { FILE * f = fopen("debug_v4/cpp_cond.bin", "wb"); fwrite(cond, sizeof(float), 1024, f); fclose(f); }
                { float * mask_f = new float[total_len*total_len]; for(int i=0;i<total_len*total_len;i++) mask_f[i]=attn_mask[i]?1.0f:0.0f;
                  FILE * f = fopen("debug_v4/cpp_mask.bin", "wb"); fwrite(mask_f, sizeof(float), total_len*total_len, f); fclose(f); delete[] mask_f; }
                { FILE * f = fopen("debug_v4/cpp_pos_ids.bin", "wb"); fwrite(pos_ids, sizeof(float), total_len, f); fclose(f); }
                { FILE * f = fopen("debug_v4/cpp_noise_raw.bin", "wb"); fwrite(z_t, sizeof(float), patch_flat, f); fclose(f); }
            }
            
            // Run DiT on conditional input (dit_input → vt_c)
            float * vt_c = new float[patch_flat];
            {
                float * h_seq = new float[total_len * DIT_HIDDEN_SIZE];
                float * iw = tensor_data(dit.input_layer_w);
                float * ib = dit.input_layer_b ? tensor_data(dit.input_layer_b) : nullptr;
                for (int ti = 0; ti < total_len; ti++)
                    manual_linear(h_seq + ti * DIT_HIDDEN_SIZE, dit_input + ti * DIT_HIDDEN_SIZE, iw, ib, DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE);
                
                float * bo_seq = new float[total_len * DIT_HIDDEN_SIZE];
                for (int i = 0; i < dit.n_layers; i++) {
                    manual_dit_block(h_seq, cond, dit.layers[i], bo_seq, total_len, attn_mask, pos_ids);
                    float * tmp = h_seq; h_seq = bo_seq; bo_seq = tmp;
                }
                
                float cs2[1024];
                for (int i = 0; i < DIT_HIDDEN_SIZE; i++) cs2[i] = cond[i] / (1.0f + expf(-cond[i]));
                float mod_raw[2 * DIT_HIDDEN_SIZE];
                float * aw = tensor_data(dit.out_adaln_w);
                float * ab = dit.out_adaln_b ? tensor_data(dit.out_adaln_b) : nullptr;
                for (int o = 0; o < 2 * DIT_HIDDEN_SIZE; o++) {
                    float s = ab ? ab[o] : 0.0f;
                    for (int i = 0; i < DIT_HIDDEN_SIZE; i++) s += aw[o * DIT_HIDDEN_SIZE + i] * cs2[i];
                    mod_raw[o] = s;
                }
                float * dit_shift = mod_raw, * dit_scale = mod_raw + DIT_HIDDEN_SIZE;
                float * dit_ln = new float[total_len * DIT_HIDDEN_SIZE];
                for (int ti = 0; ti < total_len; ti++) {
                    manual_layernorm(dit_ln + ti * DIT_HIDDEN_SIZE, h_seq + ti * DIT_HIDDEN_SIZE, DIT_HIDDEN_SIZE);
                    for (int j = 0; j < DIT_HIDDEN_SIZE; j++)
                        dit_ln[ti * DIT_HIDDEN_SIZE + j] = dit_ln[ti * DIT_HIDDEN_SIZE + j] * (1.0f + dit_scale[j]) + dit_shift[j];
                }
                float * ow = tensor_data(dit.out_proj_w);
                float * ob = dit.out_proj_b ? tensor_data(dit.out_proj_b) : nullptr;
                float * out_all = new float[total_len * VAE_LATENT_DIM];
                for (int ti = 0; ti < total_len; ti++)
                    manual_linear(out_all + ti * VAE_LATENT_DIM, dit_ln + ti * DIT_HIDDEN_SIZE, ow, ob, DIT_HIDDEN_SIZE, VAE_LATENT_DIM);
                
                delete[] h_seq; delete[] bo_seq; delete[] dit_ln;
                
                int latent_start = total_len - patch_size;
                for (int p = 0; p < patch_size; p++)
                    for (int c = 0; c < VAE_LATENT_DIM; c++)
                        vt_c[p * VAE_LATENT_DIM + c] = out_all[(latent_start + p) * VAE_LATENT_DIM + c];
                delete[] out_all;
            }
            
            // Run DiT on unconditional input (dit_cfg_input → vt_u)
            float * vt_u = nullptr;
            if (!fast_mode) {  // Full CFG: vt = vt_c + g*(vt_c - vt_u)
                vt_u = new float[patch_flat];
                // Reuse the same DiT code path (simplified: use same function)
                float* h_seq=new float[total_len*DIT_HIDDEN_SIZE];
                float*iw=tensor_data(dit.input_layer_w);float*ib=dit.input_layer_b?tensor_data(dit.input_layer_b):nullptr;
                for(int ti=0;ti<total_len;ti++) manual_linear(h_seq+ti*DIT_HIDDEN_SIZE,dit_cfg_input+ti*DIT_HIDDEN_SIZE,iw,ib,DIT_HIDDEN_SIZE,DIT_HIDDEN_SIZE);
                float*bo_seq=new float[total_len*DIT_HIDDEN_SIZE];
                for(int i=0;i<dit.n_layers;i++){manual_dit_block(h_seq,cond,dit.layers[i],bo_seq,total_len,attn_mask,pos_ids);float*tmp=h_seq;h_seq=bo_seq;bo_seq=tmp;}
                float cs2u[1024]; for(int i=0;i<DIT_HIDDEN_SIZE;i++)cs2u[i]=cond[i]/(1.0f+expf(-cond[i]));
                float mru[2*DIT_HIDDEN_SIZE];
                {float*aw=tensor_data(dit.out_adaln_w);float*ab=dit.out_adaln_b?tensor_data(dit.out_adaln_b):nullptr;
                 for(int o=0;o<2*DIT_HIDDEN_SIZE;o++){float s=ab?ab[o]:0.0f;for(int i=0;i<DIT_HIDDEN_SIZE;i++)s+=aw[o*DIT_HIDDEN_SIZE+i]*cs2u[i];mru[o]=s;}}
                float*su=mru,*scu=mru+DIT_HIDDEN_SIZE;
                float*dlu=new float[total_len*DIT_HIDDEN_SIZE];
                for(int ti=0;ti<total_len;ti++){manual_layernorm(dlu+ti*DIT_HIDDEN_SIZE,h_seq+ti*DIT_HIDDEN_SIZE,DIT_HIDDEN_SIZE);
                    for(int j=0;j<DIT_HIDDEN_SIZE;j++)dlu[ti*DIT_HIDDEN_SIZE+j]=dlu[ti*DIT_HIDDEN_SIZE+j]*(1.0f+scu[j])+su[j];}
                float*ow=tensor_data(dit.out_proj_w);float*ob=dit.out_proj_b?tensor_data(dit.out_proj_b):nullptr;
                float*ou=new float[total_len*VAE_LATENT_DIM];
                for(int ti=0;ti<total_len;ti++) manual_linear(ou+ti*VAE_LATENT_DIM,dlu+ti*DIT_HIDDEN_SIZE,ow,ob,DIT_HIDDEN_SIZE,VAE_LATENT_DIM);
                delete[] h_seq;delete[] bo_seq;delete[] dlu;
                int ls=total_len-patch_size;
                for(int p=0;p<patch_size;p++) for(int c=0;c<VAE_LATENT_DIM;c++) vt_u[p*VAE_LATENT_DIM+c]=ou[(ls+p)*VAE_LATENT_DIM+c];
                delete[] ou;
            }
            
            // CFG blending: vt = vt_c + cfg_scale * (vt_c - vt_u)
            if (vt_u) {
                for (int i = 0; i < patch_flat; i++) {
                    v_t[i] = vt_c[i] + cfg_scale * (vt_c[i] - vt_u[i]);
                }
                // Dump velocities at step 0 call 0
                if (step == 0 && call == 0) {
                    { FILE * f = fopen("debug_v4/cpp_vt_c.bin", "wb"); fwrite(vt_c, sizeof(float), patch_flat, f); fclose(f); }
                    { FILE * f = fopen("debug_v4/cpp_vt_u.bin", "wb"); fwrite(vt_u, sizeof(float), patch_flat, f); fclose(f); }
                    { FILE * f = fopen("debug_v4/cpp_vt_final.bin", "wb"); fwrite(v_t, sizeof(float), patch_flat, f); fclose(f); }
                }
                delete[] vt_u;
            } else {
                memcpy(v_t, vt_c, patch_flat * sizeof(float));
            }
            delete[] vt_c;
            
            // Euler step
            for (int i = 0; i < patch_flat; i++) z_t[i] += v_t[i] * dt;
            
            // Dump latents after each step (first call only)
            if (call == 0) {
                char fn[64]; snprintf(fn, sizeof(fn), "debug_v4/cpp_z_step%d.bin", step);
                FILE * f = fopen(fn, "wb"); if(f){fwrite(z_t,sizeof(float),patch_flat,f);fclose(f);}
            }
        }
        
        delete[] dit_input; delete[] dit_cfg_input;
        delete[] attn_mask; delete[] pos_ids;
        
        // Store raw latents (z-space, NOT denormalized!)
        memcpy(all_latents + call * patch_flat, z_t, patch_flat * sizeof(float));
        
        // Denormalize for PatchEncoder only
        float * z_denorm = new float[patch_flat];
        memcpy(z_denorm, z_t, patch_flat * sizeof(float));
        denormalize_latents(z_denorm, patch_size, latent_dim);
        memcpy(all_latents_denorm + call * patch_flat, z_denorm, patch_flat * sizeof(float));
        
        delete[] z_t; delete[] v_t;
        total_frames += patch_size;
        
        { float r = 0; for (int i = 0; i < patch_flat; i++) r += z_denorm[i] * z_denorm[i];
          printf("lat RMS=%.4f  ", sqrtf(r / patch_flat)); }
        
        // === Append latent_patch to FM buffer (uses RAW z-space latents!) ===
        float * latent_proj_out = new float[patch_size * DIT_HIDDEN_SIZE];
        apply_latent_proj(dit, all_latents + call * patch_flat, patch_size, latent_dim, latent_proj_out);
        memcpy(fm_buffer + fm_seq_len * DIT_HIDDEN_SIZE, latent_proj_out, patch_size * DIT_HIDDEN_SIZE * sizeof(float));
        memcpy(fm_cfg_buffer + fm_seq_len * DIT_HIDDEN_SIZE, latent_proj_out, patch_size * DIT_HIDDEN_SIZE * sizeof(float));
        fm_seq_len += patch_size;
        delete[] latent_proj_out;
        
        // === PatchEncoder: process denormalized latents ===
        if (!gctx_p) { ggml_init_params gp2 = { .mem_size = 256ULL*1024*1024 }; gctx_p = ggml_init(gp2); }
        else ggml_reset(gctx_p);
        
        ggml_tensor * pe_x = ggml_new_tensor_2d(gctx_p, GGML_TYPE_F32, latent_dim, patch_size);
        memcpy(tensor_data(pe_x), z_denorm, patch_flat * sizeof(float));
        ggml_tensor * pe_out = patchenc_forward(pe, gctx_p, pe_x, 1);
        float * pe_data = tensor_data(pe_out);
        
        // Debug dumps
        if (dump_debug) {
            { char fn[64]; snprintf(fn, sizeof(fn), "debug_v4/lat_call%d.bin", call);
              FILE * f = fopen(fn, "wb"); if(f){fwrite(z_denorm,sizeof(float),patch_flat,f);fclose(f);} }
            { char fn[64]; snprintf(fn, sizeof(fn), "debug_v4/pe_call%d.bin", call);
              FILE * f = fopen(fn, "wb"); if(f){fwrite(pe_data,sizeof(float),llm_dim,f);fclose(f);} }
        }
        
        // === LM step on PatchEncoder output ===
        float new_hidden[1536];
        llm_kv_cache_step(llm_w, pe_data, kv_cache, new_hidden);
        
        // === Append hidden to FM buffer ===
        float new_hidden_proj[DIT_HIDDEN_SIZE];
        apply_hidden_proj(dit, new_hidden, llm_dim, new_hidden_proj);
        memcpy(fm_buffer + fm_seq_len * DIT_HIDDEN_SIZE, new_hidden_proj, DIT_HIDDEN_SIZE * sizeof(float));
        
        // CFG: null hidden_proj (hidden_proj of zeros)
        float null_hp[DIT_HIDDEN_SIZE];
        apply_hidden_proj(dit, zero_hidden, llm_dim, null_hp);
        memcpy(fm_cfg_buffer + fm_seq_len * DIT_HIDDEN_SIZE, null_hp, DIT_HIDDEN_SIZE * sizeof(float));
        fm_seq_len++;
        
        // Dump FM buffer
        if (dump_debug) {
            { char fn[64]; snprintf(fn, sizeof(fn), "debug_v4/fmseq_call%d.bin", call);
              FILE * f = fopen(fn, "wb"); if(f){fwrite(fm_buffer,sizeof(float),fm_seq_len*DIT_HIDDEN_SIZE,f);fclose(f);} }
        }
        
        printf("fm=%d  ", fm_seq_len);
        
        // === EOS detection (full 2-layer MLP: Linear(1536→1536) + SiLU + Linear(1536→2)) ===
        float * eos_w1 = dit.eos_proj_w1 ? tensor_data(dit.eos_proj_w1) : nullptr;
        float * eos_b1 = dit.eos_proj_b1 ? tensor_data(dit.eos_proj_b1) : nullptr;
        float * eos_w2 = dit.eos_proj_w2 ? tensor_data(dit.eos_proj_w2) : nullptr;
        float * eos_b2 = dit.eos_proj_b2 ? tensor_data(dit.eos_proj_b2) : nullptr;
        if (eos_w2 && eos_w1) {
            float h1[1536];
            for (int o = 0; o < llm_dim; o++) {
                float s = eos_b1 ? eos_b1[o] : 0.0f;
                for (int i = 0; i < llm_dim; i++) s += eos_w1[o * llm_dim + i] * new_hidden[i];
                h1[o] = s / (1.0f + expf(-s));  // sigmoid(s)
                h1[o] = s * h1[o];  // s * sigmoid(s) = SiLU(s)
            }
            float eos_logits[2];
            for (int o = 0; o < 2; o++) {
                float s = eos_b2 ? eos_b2[o] : 0.0f;
                for (int i = 0; i < llm_dim; i++) s += eos_w2[o * llm_dim + i] * h1[i];
                eos_logits[o] = s;
            }
            float max_l = fmaxf(eos_logits[0], eos_logits[1]);
            float sum = expf(eos_logits[0] - max_l) + expf(eos_logits[1] - max_l);
            float eos_prob = expf(eos_logits[1] - max_l) / sum;
            printf("EOS=%.4f  ", eos_prob);
            if (eos_prob > eos_threshold) {
                printf("\n  EOS triggered (prob=%.4f > threshold=%.2f)\n", eos_prob, eos_threshold);
                end_flag = true;
            }
        }
        
        delete[] z_denorm;
    }
    
    // === VAE denormalization ===
    printf("[6] VAE denormalization (%d frames)...\n", total_frames);
    float ch_mean[128] = {0}, ch_std[128] = {0};
    for (int f = 0; f < total_frames; f++) for (int c = 0; c < latent_dim; c++) ch_mean[c] += all_latents[f * latent_dim + c];
    for (int c = 0; c < latent_dim; c++) ch_mean[c] /= total_frames;
    for (int f = 0; f < total_frames; f++) for (int c = 0; c < latent_dim; c++) {
        float d = all_latents[f * latent_dim + c] - ch_mean[c]; ch_std[c] += d * d; }
    for (int c = 0; c < latent_dim; c++) ch_std[c] = sqrtf(ch_std[c] / total_frames + 1e-8f);
    
    float * latents_vae = new float[total_frames * latent_dim];
    for (int f = 0; f < total_frames; f++) for (int c = 0; c < latent_dim; c++)
        latents_vae[f * latent_dim + c] = (all_latents[f * latent_dim + c] - ch_mean[c]) / ch_std[c] * VAE_STD[c] + VAE_MEAN[c];
    
    { float r = 0; for (int i = 0; i < total_frames * latent_dim; i++) r += all_latents[i] * all_latents[i];
      printf("  Raw latents RMS=%.4f\n", sqrtf(r / (total_frames * latent_dim))); }
    if (dump_debug) { FILE * f=fopen("debug_v4/raw_latents.bin","wb"); if(f){fwrite(all_latents,sizeof(float),total_frames*latent_dim,f);fclose(f);} }
    
    // === BigVGAN decode ===
    printf("[7] Vocoder...\n");
    BigVGANDecoder bigvgan;
    if (bigvgan_load("models/vocoder_eff.safetensors", bigvgan)) {
        int real_samples;
        float * real_wav = new float[total_frames * VAE_HOP_SAMPLES];
        bigvgan_decode(bigvgan, latents_vae, total_frames, real_wav, &real_samples);
        
        FILE * wf = fopen(out_path, "wb");
        if (wf) {
            int ds = real_samples * 2, fs = 36 + ds;
            fwrite("RIFF", 1, 4, wf); fwrite(&fs, 4, 1, wf); fwrite("WAVE", 1, 4, wf);
            fwrite("fmt ", 1, 4, wf); int fz = 16; fwrite(&fz, 4, 1, wf);
            short af = 1, nc = 1; fwrite(&af, 2, 1, wf); fwrite(&nc, 2, 1, wf);
            int sr = VAE_SAMPLE_RATE; fwrite(&sr, 4, 1, wf);
            int br = sr * 2; fwrite(&br, 4, 1, wf);
            short ba = 2, bp = 16; fwrite(&ba, 2, 1, wf); fwrite(&bp, 2, 1, wf);
            fwrite("data", 1, 4, wf); fwrite(&ds, 4, 1, wf);
            for (int i = 0; i < real_samples; i++) {
                float s = real_wav[i] * 32767.0f;
                if (s > 32767) s = 32767; if (s < -32768) s = -32768;
                short si = (short)s; fwrite(&si, 2, 1, wf);
            }
            fclose(wf);
            printf("  %s: %d samples\n", out_path, real_samples);
        }
        delete[] real_wav; bigvgan_free(bigvgan);
    }
    
    // Cleanup
    delete[] schedule;
    delete[] lm_hiddens;
    delete[] all_latents;
    delete[] all_latents_denorm;
    delete[] latents_vae;
    delete[] fm_buffer;
    delete[] fm_cfg_buffer;
    delete[] conv_tail;
    if (gctx_p) ggml_free(gctx_p);
    ggml_free(w_ctx_dit); ggml_free(w_ctx_pe);
    llm_kv_cache_free(kv_cache);
    free_llm_weights(llm_w);
    
    if (benchmark) { double t2 = now_ms(); printf("\n=== BENCHMARK ===\n  Load: %d ms  Gen: %d ms  Total: %d ms\n", (int)t_load, (int)(t2-t_gen_start), (int)(t2-t1)); }
    printf("\nDone.\n");
    return 0;
}
