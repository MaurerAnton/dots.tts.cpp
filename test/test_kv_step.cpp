// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// Test: KV cache step vs full forward for same sequence
#include "llm_manual.h"
#include "safetensors.h"
#include "dit_manual.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/model.safetensors";
    
    SafeTensorsFile sf;
    if (!sf.open(model_path)) { fprintf(stderr, "FAILED open\n"); return 1; }
    
    LLMWeights w;
    if (!load_llm_weights(sf, w, 28)) { fprintf(stderr, "FAILED load\n"); return 1; }
    sf.close();
    
    const int H=1536, n_init=11;
    int init_tokens[] = {58, 108704, 60, 14990, 1879, 58, 108704, 103124, 105761, 60, 151668};
    
    // Create a fake embedding (mimics PatchEncoder output — zero-mean random)
    float * fake_emb = (float*)malloc(H * sizeof(float));
    srand(1248464540);
    for(int i=0;i<H;i++) fake_emb[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    // Normalize to unit RMS
    float rms=0; for(int i=0;i<H;i++) rms += fake_emb[i]*fake_emb[i]; rms = sqrtf(rms/H);
    for(int i=0;i<H;i++) fake_emb[i] /= rms;  // RMS = 1.0
    
    // Method 1: KV cache step
    LLMKVCache cache;
    float * init_hiddens = (float*)malloc(n_init * H * sizeof(float));
    llm_kv_cache_init(w, init_tokens, n_init, init_hiddens, cache);
    float kv_hidden[H];
    llm_kv_cache_step(w, fake_emb, cache, kv_hidden);
    
    // Method 2: Full forward with embedding injected at last position
    // Save original embedding at a valid index, overwrite with our fake embedding
    float saved_emb[H];
    int dummy_idx = 0;  // token 0 is safe to overwrite
    memcpy(saved_emb, w.embed_w + dummy_idx*H, H*sizeof(float));
    memcpy(w.embed_w + dummy_idx*H, fake_emb, H*sizeof(float));
    
    // Build token sequence: init_tokens + dummy_idx
    int full_tokens[n_init + 1];
    memcpy(full_tokens, init_tokens, n_init*sizeof(int));
    full_tokens[n_init] = dummy_idx;  // this will embed our fake embedding
    
    float * full_hiddens = (float*)malloc((n_init+1) * H * sizeof(float));
    llm_manual_forward(w, full_tokens, n_init+1, full_hiddens);
    
    // Restore
    memcpy(w.embed_w + dummy_idx*H, saved_emb, H*sizeof(float));
    
    // Compare last hidden state
    float * full_last = full_hiddens + n_init*H;
    float kv_rms=0, full_rms=0, cross=0, diff_rms=0;
    for(int i=0;i<H;i++) {
        kv_rms += kv_hidden[i]*kv_hidden[i];
        full_rms += full_last[i]*full_last[i];
        cross += kv_hidden[i] * full_last[i];
        float d = kv_hidden[i] - full_last[i];
        diff_rms += d*d;
    }
    kv_rms = sqrtf(kv_rms/H); full_rms = sqrtf(full_rms/H);
    float corr = cross/H / kv_rms / full_rms;
    diff_rms = sqrtf(diff_rms/H);
    
    printf("KV cache step: RMS=%.4f  first5=%.6f %.6f %.6f %.6f %.6f\n",
           kv_rms, kv_hidden[0],kv_hidden[1],kv_hidden[2],kv_hidden[3],kv_hidden[4]);
    printf("Full forward:  RMS=%.4f  first5=%.6f %.6f %.6f %.6f %.6f\n",
           full_rms, full_last[0],full_last[1],full_last[2],full_last[3],full_last[4]);
    printf("Corr=%.6f  Diff RMS=%.6f\n", corr, diff_rms);
    
    if (corr > 0.9999f) printf("KV CACHE STEP MATCHES FULL FORWARD!\n");
    else printf("MISMATCH (corr=%.6f)\n", corr);
    
    llm_kv_cache_free(cache);
    free_llm_weights(w);
    free(init_hiddens); free(full_hiddens); free(fake_emb);
    return 0;
}
