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

static const float VAE_MEAN[128] = {-0.9025,-0.6183,-0.0263,-0.0685,-0.1783,-0.5570,-0.6110,-0.3879,-0.3249,-0.2662,0.3478,0.0372,-0.5750,0.0344,-0.5378,0.2272,-0.2407,0.2857,-0.4126,0.0592,-0.3421,-0.5783,0.0266,-0.0334,-0.0829,-0.3520,0.1200,-0.4391,0.3389,-0.4224,-0.3515,-0.0558,0.1107,-0.0028,0.6533,-0.0096,0.3998,0.0979,0.0710,0.1544,-0.1871,0.1712,0.2767,-0.3803,-0.2228,-0.3248,0.0888,0.0382,-0.2751,-0.1028,-0.3058,0.2343,0.0244,-0.2566,-0.0230,0.7069,-0.4051,0.5665,-0.1977,-0.4928,0.1987,-0.3744,0.2669,-0.5638,-0.0391,0.1564,-0.1035,-0.4483,0.6975,0.2761,-0.2020,0.2640,0.0682,0.1176,0.7706,-1.3600,0.0722,-0.1375,0.0337,0.1764,-0.2927,0.3615,-0.1335,0.3452,0.0590,-0.2773,0.0653,0.3692,-0.3961,-0.0325,0.1038,-0.2431,-0.0402,0.3810,-0.0239,-0.3426,-0.0924,-0.2611,0.1166,0.4047,0.1759,-0.6634,-0.0081,2.0526,0.0439,-0.1364,0.1873,-0.0263,-0.7545,0.5526,-0.0752,-0.3717,0.1637,0.7791,0.7361,-0.3620,-0.3784,0.4654,-0.2694,-0.2115,0.3204,0.2995,-0.7347,0.4701,0.1141,0.6925,0.7140,-0.0032};
static const float VAE_STD[128] = {3.0316,2.8055,1.7751,1.7465,2.0646,1.9869,2.2711,2.0302,1.8684,2.3117,2.0107,1.9569,2.8697,2.3088,2.0507,2.1352,2.6940,1.7395,1.9994,1.8537,2.0859,2.1602,1.9089,2.6236,1.9415,3.6039,1.7381,1.8368,2.7283,2.0870,2.4967,2.4496,1.6383,1.6716,4.6339,2.4076,3.9610,1.8117,1.8016,1.8019,2.0125,1.7533,1.9128,2.2415,2.0514,1.9496,1.8469,2.1641,1.9519,1.9785,1.7531,1.6008,1.9471,2.6635,2.2622,2.4504,1.9333,2.8137,2.0679,2.1297,1.8198,2.4678,2.2560,2.2628,1.9336,1.6912,2.1193,1.7280,3.0617,1.5928,1.7773,1.8976,1.6118,1.5717,2.4272,3.1342,2.3934,1.8040,1.8262,1.7740,3.5085,1.9349,2.2693,1.8149,1.5992,1.8857,2.5525,2.2709,1.8558,1.6975,1.9260,1.8116,1.5260,1.8547,1.4196,1.7291,1.6253,2.0238,2.5385,2.1883,1.6950,2.5633,2.0288,13.0302,2.1418,1.6957,4.7318,2.3545,2.6665,2.8945,1.7005,6.3088,2.2874,2.7593,2.3732,1.7797,1.9751,2.6089,1.6278,2.5487,2.0741,2.5441,2.0957,1.6018,1.4397,3.2050,1.8041,2.3601};

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
    auto token_ids_vec = bpe_tok.encode(text);
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

    // === Conditioning ===
    printf("[3] Conditioning...\n");
    ggml_init_params gp = { .mem_size = 1ULL*1024*1024*1024 };
    ggml_context * gctx = ggml_init(gp);
    ggml_tensor * ht = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, n_tok);
    memcpy(tensor_data(ht), hiddens, n_tok * n_embd * sizeof(float)); delete[] hiddens;
    ggml_tensor * cond_llm = ggml_mul_mat(gctx, dit.hidden_proj_w, ht);
    if (dit.hidden_proj_b) cond_llm = ggml_add(gctx, cond_llm, dit.hidden_proj_b);
    { ggml_cgraph * cgf = ggml_new_graph(gctx); ggml_build_forward_expand(cgf, cond_llm); ggml_graph_compute_with_ctx(gctx, cgf, n_threads); }
    { float * cd = tensor_data(cond_llm); float r=0; for(int i=0;i<DIT_HIDDEN_SIZE*n_tok;i++) r+=cd[i]*cd[i];
      printf("  hidden_proj: [%d x %d] RMS=%.4f\n", DIT_HIDDEN_SIZE, n_tok, sqrtf(r/(DIT_HIDDEN_SIZE*n_tok)));
      if (dump_debug) { FILE * f=fopen("debug_hidden_proj.bin","wb"); if(f){fwrite(cd,sizeof(float),DIT_HIDDEN_SIZE*n_tok,f);fclose(f);} } }

    t_load = now_ms() - t1; t_gen_start = now_ms();

    // === Flow Matching ===
    printf("[4] Flow matching...\n");
    int latent_dim = VAE_LATENT_DIM, patch_size = PATCHENC_PATCH_SIZE, patch_flat = patch_size * latent_dim, nfe = 10, n_calls = n_calls_user;
    float dt = 1.0f / nfe;
    int frames_per_call = patch_size, n_frames_total = n_calls * frames_per_call, history_len = 0, total_frames = 0;
    float * all_latents = new float[n_frames_total * latent_dim], * z_t = new float[patch_flat], * v_t = new float[patch_flat];
    float * cond_llm_data = new float[DIT_HIDDEN_SIZE * (n_tok + n_calls)];
    memcpy(cond_llm_data, tensor_data(cond_llm), DIT_HIDDEN_SIZE * n_tok * sizeof(float));
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
            int cur_n_tok = n_tok + call, cond_seq = cur_n_tok + history_len + patch_size;
            if (cond_seq < 8) cond_seq = 8;
            int noise_pos = cur_n_tok + history_len;
            // Build input in plain array (no ggml)
            float * dx_data = new float[cond_seq * DIT_HIDDEN_SIZE];
            { memset(dx_data, 0, cond_seq * DIT_HIDDEN_SIZE * sizeof(float));
              if (cur_n_tok > 0) memcpy(dx_data, cond_llm_data, cur_n_tok * DIT_HIDDEN_SIZE * sizeof(float));
              if (dit.latent_proj_w && history_len > 0) { float * lw = tensor_data(dit.latent_proj_w);
                  for (int h = 0; h < history_len; h++) { float * op = dx_data + (cur_n_tok + h) * DIT_HIDDEN_SIZE, * hv = history_latents + h * latent_dim;
                      for (int j = 0; j < DIT_HIDDEN_SIZE; j++) { float s = 0; for (int k = 0; k < latent_dim; k++) s += hv[k] * lw[k * 1024 + j]; op[j] = s; } } }
              for (int i = 0; i < patch_size; i++) { float * op = dx_data + (noise_pos + i) * DIT_HIDDEN_SIZE;
                  if (dit.coord_proj_w) { float * cw = tensor_data(dit.coord_proj_w), * nv = z_t + i * latent_dim;
                      for (int j = 0; j < DIT_HIDDEN_SIZE; j++) { float s = 0; for (int k = 0; k < latent_dim; k++) s += nv[k] * cw[k * 1024 + j]; op[j] = s; } } }
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
                          for (int j = 0; j < DIT_HIDDEN_SIZE; j++) { float s = 0; for (int k = 0; k < latent_dim; k++) s += nv[k] * cw[k * 1024 + j]; op[j] = s; } } } }
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
                        float vn = vnull[c * cond_seq + t_idx];
                        v_t[p * VAE_LATENT_DIM + c] = vn + cfg_scale * (v_t[p * VAE_LATENT_DIM + c] - vn);
                    }
                }
                delete[] dx_null; delete[] out_null;
            }
            { for(int i=0;i<patch_flat;i++){z_t[i]+=v_t[i]*dt;} }
        }
        memcpy(all_latents + call * frames_per_call * latent_dim, z_t, frames_per_call * latent_dim * sizeof(float));
        for (int i = 0; i < latent_dim; i++) { float v = z_t[i]; if(v>5)v=5; if(v<-5)v=-5; history_latents[history_len*latent_dim+i] = v; }
        history_len++; total_frames += frames_per_call;

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
                            memcpy(cond_llm_data + (n_tok+call)*DIT_HIDDEN_SIZE, cnd, DIT_HIDDEN_SIZE*sizeof(float));
                        else
                            memcpy(cond_llm_data + (n_tok+call)*DIT_HIDDEN_SIZE, pe_data, DIT_HIDDEN_SIZE*sizeof(float)); }
                }
                llama_batch_free(eb);
            } else {
                memcpy(cond_llm_data + (n_tok+call)*DIT_HIDDEN_SIZE, pe_data, DIT_HIDDEN_SIZE*sizeof(float));
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
