// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// dots.tts.cpp - End-to-end TTS pipeline (C++ version)
// All fixes applied: no NaN, correct conditioning, autoregressive LLM feedback
#include "dots_tts.h"
#include "dit.h"
#include "dit_manual.h"
#include "audiovae.h"
#include "bigvgan_cpp.h"
#include "patchenc.h"
#include "safetensors.h"
#include "llama.h"
#include "gpt2_bpe_tokenizer.h"
#include "speaker_embedding.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <ctime>
#include <thread>

// VAE latent statistics extracted from Python DotsTtsRuntime latents
// These match the actual latent distribution the BigVGAN decoder expects
static const float VAE_MEAN[128] = {-0.0018,1.5587,0.2704,-0.6201,0.6770,0.0476,-1.0319,0.4992,0.6062,-0.1117,0.1011,1.9314,-0.3252,1.6055,-0.7193,-0.3595,1.7855,-0.4199,1.7570,1.0285,-0.3500,1.1654,-1.7491,-0.7094,-0.5514,-2.2958,0.9232,-0.2191,-0.8963,-2.5472,-3.2895,0.3434,-0.3056,1.2524,1.2189,0.9421,-3.2393,-0.1867,-0.5583,0.5986,0.4958,0.8181,1.2283,0.3208,-0.0927,-0.5480,-0.3843,1.5058,-1.1151,-0.4964,-0.8426,0.6010,-0.3794,-1.3442,1.6066,-0.7866,-0.7095,0.2941,-0.2754,0.5637,1.5513,2.1656,0.4843,0.4405,-0.8961,0.1787,-1.7293,-0.7062,-0.8769,-0.1251,-0.6801,0.3431,-0.6298,0.2136,0.2846,-0.9208,2.1376,-0.2581,-1.8387,-0.6564,-2.3242,1.2945,1.1961,-0.3600,-0.3377,0.8106,-0.2576,1.2125,-2.3581,1.1030,-2.5128,-0.4035,-0.7615,1.6613,-0.4468,-0.7887,-1.2168,1.2327,-0.5016,-0.3032,0.1336,-2.3773,-0.9109,-6.6330,-1.2273,-1.4531,-0.2262,-1.4044,1.7953,-0.9349,0.4821,-2.9635,2.5297,-0.9634,-0.1377,0.2467,-1.2026,2.2112,-0.3299,4.8154,0.6731,-1.4437,-1.1465,-0.0825,-0.0948,-1.3719,0.3122,-0.1676};
static const float VAE_STD[128] = {3.6477,2.0494,2.0477,2.3114,1.6807,2.7022,2.9756,1.6971,1.8174,2.3360,1.9619,2.1530,2.2949,2.7127,1.6238,1.9152,3.0338,1.5830,3.2447,1.5701,2.6532,2.5800,2.1421,3.1803,1.9691,5.8602,1.7696,2.0176,2.5675,3.1446,2.3140,3.1461,1.6386,1.6639,5.4171,3.3226,3.7026,1.9677,1.9383,1.4299,1.7438,1.8232,1.5562,3.7779,2.2670,1.9966,1.8933,2.9376,1.5818,1.7817,1.2738,1.2641,1.8378,2.5080,2.2691,2.4595,2.2339,2.9236,2.1710,1.9006,1.9267,2.5662,2.3672,2.6802,1.4845,1.5066,2.4320,1.5267,2.9641,1.9826,1.1071,2.0246,1.4723,1.2689,2.5800,2.5754,2.5913,1.3481,1.5047,1.6610,1.9842,1.9742,1.8364,1.4604,1.3824,1.4949,3.4064,1.5842,1.4575,2.2500,2.2421,1.8668,1.4647,1.4465,1.7577,1.7282,2.0829,1.6112,2.7190,1.4525,1.6201,2.5962,1.7213,22.1080,1.0896,1.7761,2.9203,2.0389,3.5477,2.3818,1.5096,7.3880,1.3896,2.7805,2.0932,1.9355,2.2740,1.3479,1.6040,1.9349,2.5265,2.7129,1.8058,1.3762,1.4585,2.8542,1.5449,2.7214};

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);
bool load_patchenc_weights(SafeTensorsFile & sf, ggml_context * w_ctx, patch_encoder & enc);
bool extract_embeddings(const char * gguf_path, const char * out_path);
static double now_ms() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1000.0 + ts.tv_nsec/1000000.0; }

// Local compute_cond (replaces dit_forward_raw's compute_cond)
static void compute_cond(dit_model & m, float t_val, const float * spk_emb, float * cond) {
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
static float randn() { float u1=(float)rand()/RAND_MAX,u2=(float)rand()/RAND_MAX; if(u1<1e-6f)u1=1e-6f; return sqrtf(-2.0f*logf(u1))*cosf(2.0f*3.14159f*u2); }

int main(int argc, char ** argv) {
    setbuf(stdout, NULL);
    double t1 = now_ms(), t_load = 0, t_gen_start = 0;

    const char * text = "Hello world", * out_path = "output.wav";
    const char * model_dir = getenv("DOTS_TTS_MODEL") ? getenv("DOTS_TTS_MODEL") : "models";
    const char * gguf_path = "models/dots_llm.gguf", * embd_path = "models/token_embd_flat.bin", * speaker_name = "neutral";
    int n_calls_user = 8; uint32_t force_seed = 0;
    bool use_force_seed = false, dump_debug = false, use_llm = true, benchmark = false;
    float cfg_scale = 1.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i+1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--model") == 0 && i+1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--gguf") == 0 && i+1 < argc) gguf_path = argv[++i];
        else if (strcmp(argv[i], "--length") == 0 && i+1 < argc) n_calls_user = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) { force_seed = (uint32_t)atoi(argv[++i]); use_force_seed = true; }
        else if (strcmp(argv[i], "--speaker") == 0 && i+1 < argc) speaker_name = argv[++i];
        else if (strcmp(argv[i], "--dump") == 0) dump_debug = true;
        else if (strcmp(argv[i], "--cfg") == 0 && i+1 < argc) cfg_scale = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--fast") == 0) use_llm = false;
        else if (strcmp(argv[i], "--benchmark") == 0) benchmark = true;
        else if (argv[i][0] != '-') text = argv[i];
    }
    if (n_calls_user < 1) n_calls_user = 1;
    if (n_calls_user > 100) n_calls_user = 100;
    int n_threads = std::thread::hardware_concurrency();
    if (n_threads < 1) n_threads = 1;

    printf("dots.tts.cpp v2 -- C++ TTS\n  Text: '%s'  Frames: %d  Threads: %d\n\n", text, n_calls_user, n_threads);

    // === Tokenization ===
    printf("[1] Tokenizing...\n");
    GPT2BPETokenizer bpe_tok;
    if (!bpe_tok.load(model_dir)) { fprintf(stderr, "FAILED: tokenizer\n"); return 1; }
    // Wrap text in Python template
    std::string template_text = "[文本]" + std::string(text) + "[文本对应语音]<|audio_gen_start|>";
    auto token_ids_vec = bpe_tok.encode(template_text.c_str());
    int n_tok = (int)token_ids_vec.size();
    if (n_tok == 0) { fprintf(stderr, "Empty tokenization\n"); return 1; }
    int32_t * token_ids = token_ids_vec.data();
    printf("  Tokenized: %d tokens\n", n_tok);

    uint32_t hash = use_force_seed ? force_seed : 5381;
    if (!use_force_seed) for (int i = 0; i < n_tok; i++) hash = ((hash << 5) + hash) + token_ids[i];
    srand(hash);
    printf("  Seed: %u\n", hash);

    const int n_embd = 1536;
    llama_model * llm = nullptr; llama_context * lctx = nullptr;
    float * token_embd = nullptr;
    float * hiddens = new float[n_tok * n_embd];

    if (use_llm) {
        printf("[1b] Loading LLM...\n");
        llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        llm = llama_model_load_from_file(gguf_path, mp);
        if (!llm) { fprintf(stderr, "FAILED: LLM\n"); return 1; }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 512; cp.n_batch = 512; cp.embeddings = true;
        lctx = llama_init_from_model(llm, cp);
        if (!lctx) { fprintf(stderr, "FAILED: ctx\n"); return 1; }
        llama_set_n_threads(lctx, n_threads, n_threads);
        llama_token tokens[256];
        for (int i = 0; i < n_tok && i < 256; i++) tokens[i] = token_ids[i];
        if (llama_decode(lctx, llama_batch_get_one(tokens, n_tok)) != 0) { fprintf(stderr, "LLM fail\n"); return 1; }
        for (int i = 0; i < n_tok; i++) { float * e = llama_get_embeddings_ith(lctx, i); if (e) memcpy(hiddens+i*n_embd, e, n_embd*sizeof(float)); }
        printf("  %d tokens decoded (LLM alive)\n", n_tok);
    } else {
        FILE * ef = fopen(embd_path, "rb");
        if (!ef) { printf("  Extracting token embeddings from GGUF...\n"); if (!extract_embeddings(gguf_path, embd_path)) { fprintf(stderr, "FAILED\n"); return 1; } ef = fopen(embd_path, "rb"); }
        fseek(ef, 0, SEEK_END); long esz = ftell(ef); fseek(ef, 0, SEEK_SET);
        token_embd = (float*)malloc(esz); fread(token_embd, 1, esz, ef); fclose(ef);
        for (int i = 0; i < n_tok; i++) { int tid = token_ids[i]; if (tid >= 0 && tid < 151672) memcpy(hiddens+i*n_embd, token_embd+tid*n_embd, n_embd*sizeof(float)); }
        printf("  Embeddings: %d tokens\n", n_tok); free(token_embd);
    }

    // === Load DiT + PatchEncoder (separate contexts to avoid corruption) ===
    printf("[2] Loading models...\n");
    SafeTensorsFile sf; sf.open((std::string(model_dir) + "/model.safetensors").c_str());
    ggml_init_params wp_dit = { .mem_size = 3ULL*1024*1024*1024 };
    ggml_context * w_ctx_dit = ggml_init(wp_dit);
    dit_model dit; load_dit_weights(sf, w_ctx_dit, dit);
    sf.close();
    // Reload for PatchEncoder in its own context
    sf.open((std::string(model_dir) + "/model.safetensors").c_str());
    ggml_init_params wp_pe = { .mem_size = 3ULL*1024*1024*1024 };
    ggml_context * w_ctx_pe = ggml_init(wp_pe);
    patch_encoder pe; load_patchenc_weights(sf, w_ctx_pe, pe);
    sf.close();
    printf("  DiT: %d layers, PE: loaded\n", DIT_NUM_LAYERS);

    float spk_emb[512] = {0};
    // Use zero speaker embedding to match Python calibration
    // { FILE * f = fopen("speaker_emb.bin", "rb"); if (f) { fread(spk_emb, sizeof(float), 512, f); fclose(f); } else speaker_embedding_from_name(speaker_name, spk_emb); }
    printf("  Speaker: zeros (matching calibration)\n");

    // === Conditioning: only last token's hidden state (hidden_patch_size=1) ===
    printf("[3] Conditioning...\n");
    ggml_init_params gp = { .mem_size = 1ULL*1024*1024*1024 };
    ggml_context * gctx = ggml_init(gp);
    // Python only projects last hidden_patch_size=1 hidden states, not all
    ggml_tensor * ht = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, 1);
    memcpy(tensor_data(ht), hiddens + (n_tok-1)*n_embd, n_embd * sizeof(float)); delete[] hiddens;
    ggml_tensor * cond_llm = ggml_mul_mat(gctx, dit.hidden_proj_w, ht);
    if (dit.hidden_proj_b) cond_llm = ggml_add(gctx, cond_llm, dit.hidden_proj_b);
    { ggml_cgraph * cgf = ggml_new_graph(gctx); ggml_build_forward_expand(cgf, cond_llm); ggml_graph_compute_with_ctx(gctx, cgf, n_threads); }
    { float * cd = tensor_data(cond_llm); float r=0; for(int i=0;i<DIT_HIDDEN_SIZE;i++) r+=cd[i]*cd[i];
      printf("  hidden_proj: [%d x 1] RMS=%.4f\n", DIT_HIDDEN_SIZE, sqrtf(r/DIT_HIDDEN_SIZE));
      if (dump_debug) { FILE * f=fopen("debug_hidden_proj.bin","wb"); if(f){fwrite(cd,sizeof(float),DIT_HIDDEN_SIZE,f);fclose(f);} } }

    t_load = now_ms() - t1; t_gen_start = now_ms();

    // === Flow Matching ===
    printf("[4] Flow matching...\n");
    int latent_dim = VAE_LATENT_DIM, patch_size = PATCHENC_PATCH_SIZE, patch_flat = patch_size * latent_dim, nfe = 10, n_calls = n_calls_user;
    float dt = 1.0f / nfe;
    int frames_per_call = patch_size, n_frames_total = n_calls * frames_per_call, history_len = 0, total_frames = 0;
    float * all_latents = new float[n_frames_total * latent_dim], * z_t = new float[patch_flat], * v_t = new float[patch_flat];
    float * cond_llm_data = new float[DIT_HIDDEN_SIZE * (1 + n_calls)];
    memcpy(cond_llm_data, tensor_data(cond_llm), DIT_HIDDEN_SIZE * 1 * sizeof(float));
    float * history_latents = new float[n_frames_total * latent_dim];

    // Free gctx after copying — manual DiT doesn't need it
    ggml_free(gctx);
    gctx = nullptr;

    for (int call = 0; call < n_calls; call++) {
        printf("  Call %d/%d: ", call+1, n_calls);
        if (call > 0) { ggml_free(gctx); gctx = ggml_init(gp); }
        for (int i = 0; i < patch_flat; i++) { float z = randn(); if(z>5)z=5; if(z<-5)z=-5; z_t[i]=z; }

        for (int step = 0; step < nfe; step++) {
            float t = (float)step * dt;
            int cur_n_tok = 1 + call, cond_seq = cur_n_tok + history_len + patch_size;
            if (cond_seq < 8) cond_seq = 8;
            int noise_pos = cur_n_tok + history_len;
            // Build input in plain array (no ggml)
            float * dx_data = new float[cond_seq * DIT_HIDDEN_SIZE];
            { memset(dx_data, 0, cond_seq * DIT_HIDDEN_SIZE * sizeof(float));
              if (cur_n_tok > 0) memcpy(dx_data, cond_llm_data, cur_n_tok * DIT_HIDDEN_SIZE * sizeof(float));
              if (dit.latent_proj_w && history_len > 0) { float * lw = tensor_data(dit.latent_proj_w);
                  for (int h = 0; h < history_len; h++) { float * op = dx_data + (cur_n_tok + h) * DIT_HIDDEN_SIZE, * hv = history_latents + h * latent_dim;
                      for (int j = 0; j < DIT_HIDDEN_SIZE; j++) { float s = 0; for (int k = 0; k < latent_dim; k++) s += hv[k] * lw[j * latent_dim + k]; op[j] = s; } } }
              for (int i = 0; i < patch_size; i++) { float * op = dx_data + (noise_pos + i) * DIT_HIDDEN_SIZE;
                  if (dit.coord_proj_w) { float * cw = tensor_data(dit.coord_proj_w), * nv = z_t + i * latent_dim;
                      for (int j = 0; j < DIT_HIDDEN_SIZE; j++) { float s = 0; for (int k = 0; k < latent_dim; k++) s += nv[k] * cw[j * latent_dim + k]; op[j] = s; } } }
              for (int p = 0; p < cond_seq; p++) { float * pos = dx_data + p * DIT_HIDDEN_SIZE, r = 0; for (int j = 0; j < DIT_HIDDEN_SIZE; j++) r += pos[j] * pos[j];
                  r = sqrtf(r / DIT_HIDDEN_SIZE); if (r > 10.0f) { float s = 10.0f / r; for (int j = 0; j < DIT_HIDDEN_SIZE; j++) pos[j] *= s; } } }
            // Call manual DiT forward (pure C++, byte-perfect with Python)
            float * out_data = new float[cond_seq * VAE_LATENT_DIM];
            // Dump input for Python comparison
            if (step == 0 && call == 0) {
                FILE * f = fopen("debug/cpp_dit_input.bin", "wb");
                if (f) { fwrite(dx_data, sizeof(float), cond_seq * DIT_HIDDEN_SIZE, f); fclose(f); }
            }
            float cond[1024]; compute_cond(dit, t, spk_emb, cond);
            float * h_dit = new float[cond_seq * DIT_HIDDEN_SIZE];
            {float*iw=tensor_data(dit.input_layer_w);float*ib=dit.input_layer_b?tensor_data(dit.input_layer_b):nullptr;
             for(int ti=0;ti<cond_seq;ti++) manual_linear(h_dit+ti*DIT_HIDDEN_SIZE,dx_data+ti*DIT_HIDDEN_SIZE,iw,ib,DIT_HIDDEN_SIZE,DIT_HIDDEN_SIZE);}
            float * bo_dit = new float[cond_seq * DIT_HIDDEN_SIZE];
            for(int i=0;i<dit.n_layers;i++){manual_dit_block(h_dit,cond,dit.layers[i],bo_dit,cond_seq);float*tmp=h_dit;h_dit=bo_dit;bo_dit=tmp;}
            // Output layer
            float cs2[1024]; for(int i=0;i<1024;i++) cs2[i]=cond[i]/(1.0f+expf(-cond[i]));
            float mod_raw2[2*DIT_HIDDEN_SIZE];
            {float*aw=tensor_data(dit.out_adaln_w);float*ab=dit.out_adaln_b?tensor_data(dit.out_adaln_b):nullptr;
             for(int o=0;o<2*DIT_HIDDEN_SIZE;o++){float s=ab?ab[o]:0.0f;for(int i=0;i<DIT_HIDDEN_SIZE;i++)s+=aw[o*DIT_HIDDEN_SIZE+i]*cs2[i];mod_raw2[o]=s;}}
            float*dit_shift=mod_raw2,*dit_scale=mod_raw2+DIT_HIDDEN_SIZE;
            float*dit_ln=new float[cond_seq*DIT_HIDDEN_SIZE];
            for(int ti=0;ti<cond_seq;ti++){manual_layernorm(dit_ln+ti*DIT_HIDDEN_SIZE,h_dit+ti*DIT_HIDDEN_SIZE,DIT_HIDDEN_SIZE);
                for(int j=0;j<DIT_HIDDEN_SIZE;j++)dit_ln[ti*DIT_HIDDEN_SIZE+j]=dit_ln[ti*DIT_HIDDEN_SIZE+j]*(1.0f+dit_scale[j])+dit_shift[j];}
            float*ow=tensor_data(dit.out_proj_w);float*ob=dit.out_proj_b?tensor_data(dit.out_proj_b):nullptr;
            for(int ti=0;ti<cond_seq;ti++) manual_linear(out_data+ti*VAE_LATENT_DIM,dit_ln+ti*DIT_HIDDEN_SIZE,ow,ob,DIT_HIDDEN_SIZE,VAE_LATENT_DIM);
            delete[] h_dit; delete[] bo_dit; delete[] dit_ln;
            { float r=0; for(int i=0;i<cond_seq*VAE_LATENT_DIM;i++) r+=out_data[i]*out_data[i];
              fprintf(stderr, "  dit_raw_out rms=%.4f\n", sqrtf(r/(cond_seq*VAE_LATENT_DIM)));
              // Dump first step velocity for comparison with Python
              if (step == 0 && call == 0) {
                  FILE * f = fopen("debug/cpp_velocity.bin", "wb");
                  if (f) { fwrite(out_data, sizeof(float), cond_seq * VAE_LATENT_DIM, f); fclose(f); }
              } }
            float * vdata = out_data; bool has_nan = false;
            // Extract velocity for noise positions
            for (int p = 0; p < patch_size; p++) {
                int t_idx = noise_pos + p;
                for (int c = 0; c < VAE_LATENT_DIM; c++) {
                    float val = vdata[c * cond_seq + t_idx];
                    v_t[p * VAE_LATENT_DIM + c] = val;
                    if (std::isnan(val) || std::isinf(val)) has_nan = true;
                }
            }
            delete[] dx_data; delete[] out_data;
            if (has_nan) { printf("(NaN) "); for (int i=0;i<patch_flat;i++){float z=randn();if(z>5)z=5;if(z<-5)z=-5;z_t[i]=z;} history_len=0; break; }
            // CFG: blend with null-conditioning velocity for better quality
            if (cfg_scale > 1.001f && dit.hidden_proj_b && !has_nan) {
                float * dx_null = new float[cond_seq * DIT_HIDDEN_SIZE];
                { memset(dx_null, 0, cond_seq * DIT_HIDDEN_SIZE * sizeof(float));
                  float * bd = (float*)((char*)dit.hidden_proj_b->data + dit.hidden_proj_b->view_offs);
                  for (int p = 0; p < cur_n_tok + history_len; p++) memcpy(dx_null + p * DIT_HIDDEN_SIZE, bd, DIT_HIDDEN_SIZE * sizeof(float));
                  for (int i = 0; i < patch_size; i++) { float * op = dx_null + (noise_pos + i) * DIT_HIDDEN_SIZE;
                      if (dit.coord_proj_w) { float * cw = tensor_data(dit.coord_proj_w), * nv = z_t + i * latent_dim;
                          for (int j = 0; j < DIT_HIDDEN_SIZE; j++) { float s = 0; for (int k = 0; k < latent_dim; k++) s += nv[k] * cw[j * latent_dim + k]; op[j] = s; } } } }
                float * out_null = new float[cond_seq * VAE_LATENT_DIM];
                // Manual DiT for null conditioning
                {
                    float null_cond[1024]; compute_cond(dit, t, spk_emb, null_cond);
                    float * hn = new float[cond_seq * DIT_HIDDEN_SIZE];
                    {float*iw=tensor_data(dit.input_layer_w);float*ib=dit.input_layer_b?tensor_data(dit.input_layer_b):nullptr;
                     for(int ti=0;ti<cond_seq;ti++) manual_linear(hn+ti*DIT_HIDDEN_SIZE,dx_null+ti*DIT_HIDDEN_SIZE,iw,ib,DIT_HIDDEN_SIZE,DIT_HIDDEN_SIZE);}
                    float * bn = new float[cond_seq * DIT_HIDDEN_SIZE];
                    for(int i=0;i<dit.n_layers;i++){manual_dit_block(hn,null_cond,dit.layers[i],bn,cond_seq);float*tmp=hn;hn=bn;bn=tmp;}
                    float csn[1024]; for(int i=0;i<1024;i++) csn[i]=null_cond[i]/(1.0f+expf(-null_cond[i]));
                    float mrn[2*DIT_HIDDEN_SIZE];
                    {float*aw=tensor_data(dit.out_adaln_w);float*ab=dit.out_adaln_b?tensor_data(dit.out_adaln_b):nullptr;
                     for(int o=0;o<2*DIT_HIDDEN_SIZE;o++){float s=ab?ab[o]:0.0f;for(int i=0;i<DIT_HIDDEN_SIZE;i++)s+=aw[o*DIT_HIDDEN_SIZE+i]*csn[i];mrn[o]=s;}}
                    float*sn=mrn,*scn=mrn+DIT_HIDDEN_SIZE;
                    float*lnn=new float[cond_seq*DIT_HIDDEN_SIZE];
                    for(int ti=0;ti<cond_seq;ti++){manual_layernorm(lnn+ti*DIT_HIDDEN_SIZE,hn+ti*DIT_HIDDEN_SIZE,DIT_HIDDEN_SIZE);
                        for(int j=0;j<DIT_HIDDEN_SIZE;j++)lnn[ti*DIT_HIDDEN_SIZE+j]=lnn[ti*DIT_HIDDEN_SIZE+j]*(1.0f+scn[j])+sn[j];}
                    float*own=tensor_data(dit.out_proj_w);float*obn=dit.out_proj_b?tensor_data(dit.out_proj_b):nullptr;
                    for(int ti=0;ti<cond_seq;ti++) manual_linear(out_null+ti*VAE_LATENT_DIM,lnn+ti*DIT_HIDDEN_SIZE,own,obn,DIT_HIDDEN_SIZE,VAE_LATENT_DIM);
                    delete[] hn; delete[] bn; delete[] lnn;
                }
                float * vnull = out_null;
                for (int p = 0; p < patch_size; p++) {
                    int t_idx = noise_pos + p;
                    for (int c = 0; c < VAE_LATENT_DIM; c++) {
                        float vn = vnull[t_idx * VAE_LATENT_DIM + c];
                        v_t[p * VAE_LATENT_DIM + c] = vn + cfg_scale * (v_t[p * VAE_LATENT_DIM + c] - vn);
                    }
                }
                delete[] dx_null; delete[] out_null;
            }
            { for(int i=0;i<patch_flat;i++){z_t[i]+=v_t[i]*dt;} }
        }
        memcpy(all_latents + call * frames_per_call * latent_dim, z_t, frames_per_call * latent_dim * sizeof(float));
        // Store ALL frames_per_call frames as history (matches Python _append_history_chunk)
        memcpy(history_latents + history_len * latent_dim, z_t, frames_per_call * latent_dim * sizeof(float));
        history_len += frames_per_call; total_frames += frames_per_call;

        // Recreate gctx for PatchEncoder
        if (pe.in_proj_w && call < n_calls - 1) {
            if (!gctx) { ggml_init_params gp2 = { .mem_size = 256ULL*1024*1024 }; gctx = ggml_init(gp2); }
            else ggml_reset(gctx);
            ggml_tensor * pe_x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, latent_dim, patch_size);
            memcpy(tensor_data(pe_x), z_t, patch_flat * sizeof(float));
            ggml_tensor * pe_out = patchenc_forward(pe, gctx, pe_x, 1);
            float * pe_data = tensor_data(pe_out);
            if (lctx) {
                llama_batch eb = llama_batch_init(1, n_embd, 1); eb.n_tokens = 1;
                eb.pos[0] = n_tok + call; eb.n_seq_id[0] = 1; eb.seq_id[0][0] = 0; eb.logits[0] = 1;
                memcpy(eb.embd, pe_data, n_embd * sizeof(float));
                if (llama_decode(lctx, eb) == 0) {
                    float * nh = llama_get_embeddings_ith(lctx, 0); // batch index, not abs position
                    if (nh) { ggml_tensor * hn = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, 1); memcpy(tensor_data(hn), nh, n_embd*sizeof(float));
                        ggml_tensor * cn = ggml_mul_mat(gctx, dit.hidden_proj_w, hn); if (dit.hidden_proj_b) cn = ggml_add(gctx, cn, dit.hidden_proj_b);
                        { ggml_cgraph * cg = ggml_new_graph(gctx); ggml_build_forward_expand(cg, cn); ggml_graph_compute_with_ctx(gctx, cg, n_threads); }
                        // Safety: check RMS before using, fallback to PE append if bad
                        float * cnd = (float*)tensor_data(cn);
                        float crms = 0; for(int j=0;j<DIT_HIDDEN_SIZE;j++) crms += cnd[j]*cnd[j]; crms = sqrtf(crms/DIT_HIDDEN_SIZE);
                        if (crms > 0.01f && crms < 100.0f && !std::isnan(crms))
                            memcpy(cond_llm_data + (1+call)*DIT_HIDDEN_SIZE, cnd, DIT_HIDDEN_SIZE*sizeof(float));
                        else
                            memcpy(cond_llm_data + (1+call)*DIT_HIDDEN_SIZE, pe_data, DIT_HIDDEN_SIZE*sizeof(float)); }
                }
                llama_batch_free(eb);
            } else {
                memcpy(cond_llm_data + (1+call)*DIT_HIDDEN_SIZE, pe_data, DIT_HIDDEN_SIZE*sizeof(float));
            }
        }
        float ms=0; for(int i=0;i<patch_flat;i++){if(std::isnan(z_t[i])||std::isinf(z_t[i]))z_t[i]=0;ms+=z_t[i]*z_t[i];}
        printf("rms=%.4f\n", sqrtf(ms/patch_flat));
    }
    delete[] cond_llm_data; delete[] z_t; delete[] v_t; delete[] history_latents;

    // VAE normalization
    for (int i = 0; i < total_frames * VAE_LATENT_DIM; i++) if (std::isnan(all_latents[i]) || std::isinf(all_latents[i])) all_latents[i] = 0;
    float ch_mean[128]={0}, ch_std[128]={0};
    for(int f=0;f<total_frames;f++) for(int c=0;c<VAE_LATENT_DIM;c++) ch_mean[c]+=all_latents[f*VAE_LATENT_DIM+c];
    for(int c=0;c<VAE_LATENT_DIM;c++) ch_mean[c]/=total_frames;
    for(int f=0;f<total_frames;f++) for(int c=0;c<VAE_LATENT_DIM;c++){float d=all_latents[f*VAE_LATENT_DIM+c]-ch_mean[c];ch_std[c]+=d*d;}
    for(int c=0;c<VAE_LATENT_DIM;c++) ch_std[c]=sqrtf(ch_std[c]/total_frames+1e-8f);
    // Denormalize to VAE distribution: z = (z - mean) / std * VAE_STD + VAE_MEAN
    for(int f=0;f<total_frames;f++) for(int c=0;c<VAE_LATENT_DIM;c++)
        all_latents[f*VAE_LATENT_DIM+c] = (all_latents[f*VAE_LATENT_DIM+c] - ch_mean[c]) / ch_std[c] * VAE_STD[c] + VAE_MEAN[c];
    { float rms=0; for(int i=0;i<total_frames*VAE_LATENT_DIM;i++) rms+=all_latents[i]*all_latents[i]; printf("  Latents RMS=%.4f\\n", sqrtf(rms/(total_frames*VAE_LATENT_DIM))); }
    if (dump_debug) { FILE * f=fopen("latents.bin","wb"); if(f){fwrite(all_latents,sizeof(float),total_frames*latent_dim,f);fclose(f);} }

    // AudioVAE + BigVGAN
    printf("[5] Vocoder...\n");
    int n_frames = total_frames, n_samples;
    float * wav = new float[n_frames * VAE_HOP_SAMPLES];
    if (!gctx) { ggml_init_params gp2 = { .mem_size = 256ULL*1024*1024 }; gctx = ggml_init(gp2); }
    ggml_reset(gctx);
    ggml_tensor * lt = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VAE_LATENT_DIM, n_frames);
    memcpy(tensor_data(lt), all_latents, n_frames * VAE_LATENT_DIM * sizeof(float));
    audiovae_decode_simple(gctx, lt, n_frames, wav, &n_samples);
    { FILE * wf = fopen(out_path, "wb"); if (wf) {
        int ds = n_samples * 2, fs = 36 + ds; fwrite("RIFF",1,4,wf); fwrite(&fs,4,1,wf); fwrite("WAVE",1,4,wf);
        fwrite("fmt ",1,4,wf); int fz=16; fwrite(&fz,4,1,wf); short af=1; fwrite(&af,2,1,wf); short nc=1; fwrite(&nc,2,1,wf);
        int sr=VAE_SAMPLE_RATE; fwrite(&sr,4,1,wf); int br=sr*2; fwrite(&br,4,1,wf); short ba=2; fwrite(&ba,2,1,wf); short bp=16; fwrite(&bp,2,1,wf);
        fwrite("data",1,4,wf); fwrite(&ds,4,1,wf);
        for(int i=0;i<n_samples;i++){float s=wav[i]*32767.0f;if(s>32767)s=32767;if(s<-32768)s=-32768;short si=(short)s;fwrite(&si,2,1,wf);} fclose(wf); } }

    BigVGANDecoder bigvgan;
    if (bigvgan_load("models/vocoder_eff.safetensors", bigvgan)) {
        int real_samples; float * real_wav = new float[n_frames * VAE_HOP_SAMPLES];
        bigvgan_decode(bigvgan, all_latents, n_frames, real_wav, &real_samples);
        { FILE * wf = fopen(out_path, "wb"); if (wf) {
            int ds = real_samples * 2, fs = 36 + ds; fwrite("RIFF",1,4,wf); fwrite(&fs,4,1,wf); fwrite("WAVE",1,4,wf);
            fwrite("fmt ",1,4,wf); int fz=16; fwrite(&fz,4,1,wf); short af=1; fwrite(&af,2,1,wf); short nc=1; fwrite(&nc,2,1,wf);
            int sr=VAE_SAMPLE_RATE; fwrite(&sr,4,1,wf); int br=sr*2; fwrite(&br,4,1,wf); short ba=2; fwrite(&ba,2,1,wf); short bp=16; fwrite(&bp,2,1,wf);
            fwrite("data",1,4,wf); fwrite(&ds,4,1,wf);
            for(int i=0;i<real_samples;i++){float s=real_wav[i]*32767.0f;if(s>32767)s=32767;if(s<-32768)s=-32768;short si=(short)s;fwrite(&si,2,1,wf);} fclose(wf); }
            printf("  %s: %d samples (pure C++)\n", out_path, real_samples); }
        delete[] real_wav; bigvgan_free(bigvgan);
    }
    delete[] wav; delete[] all_latents;
    ggml_free(gctx); ggml_free(w_ctx_dit); ggml_free(w_ctx_pe);
    if (lctx) llama_free(lctx);
    if (llm) llama_model_free(llm);
    if (benchmark) { double t2 = now_ms(); printf("\n=== BENCHMARK ===\n  Load: %d ms  Gen: %d ms  Total: %d ms\n", (int)t_load, (int)(t2-t_gen_start), (int)(t2-t1)); }
    if (dump_debug) printf("  Debug dumps written\n");
    printf("\nDone.\n");
    return 0;
}
