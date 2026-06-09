// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// AudioVAE encoder — causal strided Conv1d encoder with ResStack residual blocks
#include "audiovae_encoder.h"
#include "safetensors.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

// Use BigVGAN's LSTM
extern void lstm_forward(const float * x, int seq_len, int hidden, int n_layers,
    const float * w_ih[4], const float * w_hh[4],
    const float * b_ih[4], const float * b_hh[4], float * out, bool skip);

// Weight norm: actual_weight = g * v / ||v||
float wn_scale(float g, const float * v, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) norm += v[i] * v[i];
    norm = sqrtf(norm + 1e-12f);
    return g / norm;
}

// PyTorch-compatible causal Conv1d (forward sliding: pt = t + k)
// Different from BigVGAN's backward-sliding conv!
void conv1d_causal_pt(float * out, const float * in, int C_in, int T,
                              const float * w, const float * bias,
                              int C_out, int K, int stride, int dilation) {
    int left_pad = dilation * (K - 1);
    int padded_T = T + left_pad;
    float * padded = new float[C_in * padded_T]();
    for (int c = 0; c < C_in; c++)
        memcpy(padded + c * padded_T + left_pad, in + c * T, T * sizeof(float));
    
    int T_out = (T - 1) / stride + 1;
    for (int o = 0; o < C_out; o++) {
        float b = bias ? bias[o] : 0;
        for (int t = 0; t < T_out; t++) {
            float s = b;
            for (int k = 0; k < K; k++) {
                int pt = t * stride + k * dilation;  // FORWARD sliding with dilation
                for (int c = 0; c < C_in; c++)
                    s += padded[c * padded_T + pt] * w[(o * C_in + c) * K + k];
            }
            out[o * T_out + t] = s;
        }
    }
    delete[] padded;
}

// Causal Conv1d with weight norm (weight_g + weight_v)
// Layout: in is [C_in, T], out is [C_out, T_out]
void causal_conv1d_wn(float * out, const float * in, int C_in, int T,
                              const float * wg, const float * wv, const float * bias,
                              int C_out, int K, int stride, int dilation) {
    int left_pad = dilation * (K - 1);
    int padded_T = T + left_pad;
    
    // Allocate padded input
    float * padded = new float[C_in * padded_T]();
    // Left-pad with zeros (causal)
    for (int c = 0; c < C_in; c++)
        memcpy(padded + c * padded_T + left_pad, in + c * T, T * sizeof(float));
    
    int T_out = (padded_T - dilation * (K - 1) - 1) / stride + 1;
    
    // Compute actual weights from weight norm (or use directly if no wg)
    float * actual_w = new float[C_out * C_in * K];
    if (wg) {
        for (int o = 0; o < C_out; o++) {
            float s = wn_scale(wg[o], wv + o * C_in * K, C_in * K);
            for (int i = 0; i < C_in * K; i++)
                actual_w[o * C_in * K + i] = wv[o * C_in * K + i] * s;
        }
    } else {
        memcpy(actual_w, wv, C_out * C_in * K * sizeof(float));
    }
    
    // Conv1d
    for (int o = 0; o < C_out; o++) {
        float b = bias ? bias[o] : 0;
        for (int t = 0; t < T_out; t++) {
            float s = b;
            int base = t * stride;
            for (int c = 0; c < C_in; c++)
                for (int k = 0; k < K; k++) {
                    int idx = base + k * dilation;
                    s += padded[c * padded_T + idx] * actual_w[(o * C_in + c) * K + k];
                }
            out[o * T_out + t] = s;
        }
    }
    
    delete[] padded; delete[] actual_w;
}

// LeakyReLU
static void leaky_relu(float * x, int n, float slope = 0.2f) {
    for (int i = 0; i < n; i++)
        if (x[i] < 0) x[i] *= slope;
}

// ============ Weight loading ============

static void load_wn_conv(SafeTensorsFile & sf, const char * prefix,
                          float * & wg, float * & wv, float * & bias, int & C_out, int & C_in, int & K) {
    // Try weight_v (original safetensors with weight_norm)
    std::string wv_name = std::string(prefix) + ".weight_v";
    auto info = sf.find(wv_name.c_str());
    if (info) {
        C_out = info->shape[0]; C_in = info->shape[1]; K = info->shape[2];
        int n = C_out * C_in * K;
        wv = new float[n]; sf.load_raw(*info, wv, n);
        std::string wg_name = std::string(prefix) + ".weight_g";
        wg = new float[C_out]; sf.load_raw(*sf.find(wg_name.c_str()), wg, C_out);
        std::string b_name = std::string(prefix) + ".bias";
        auto binfo = sf.find(b_name.c_str());
        if (binfo) { bias = new float[C_out]; sf.load_raw(*binfo, bias, C_out); }
        return;
    }
    // Fallback: try .weight directly (effective weights, no weight_norm needed)
    std::string w_name = std::string(prefix) + ".weight";
    info = sf.find(w_name.c_str());
    if (!info) return;
    C_out = info->shape[0]; C_in = info->shape[1]; K = info->shape[2];
    int n = C_out * C_in * K;
    wg = nullptr;  // signal: no weight_norm
    wv = new float[n]; sf.load_raw(*info, wv, n);
    std::string b_name = std::string(prefix) + ".bias";
    auto binfo = sf.find(b_name.c_str());
    if (binfo) { bias = new float[C_out]; sf.load_raw(*binfo, bias, C_out); }
}

bool audiovae_encoder_load(const char * safetensors_path, AudioVAEEncoderWeights & w) {
    SafeTensorsFile sf;
    if (!sf.open(safetensors_path)) return false;
    
    // Pre conv: generator.0 (1→12, K=3)
    {
        int oc, ic, k;
        load_wn_conv(sf, "audio_encoder.generator.0.layer", w.pre_conv_wg, w.pre_conv_wv, w.pre_conv_bias, oc, ic, k);
        printf("  Pre-conv: %d→%d K=%d\n", ic, oc, k);
    }
    
    // 7 stages
    int stage_indices[] = {2, 5, 8, 11, 14, 17, 20};
    int in_chs[] = {12, 24, 48, 96, 192, 384};  // input channels per stage
    int out_chs[] = {24, 48, 96, 192, 384, 768, 128}; // output channels
    
    for (int s = 0; s < 7; s++) {
        int idx = stage_indices[s];
        char prefix[128];
        
        if (s < 6) {
            // Transition conv
            snprintf(prefix, sizeof(prefix), "audio_encoder.generator.%d.layer", idx);
            int oc, ic, k;
            load_wn_conv(sf, prefix, w.stage_conv_wg[s], w.stage_conv_wv[s], w.stage_conv_bias[s], oc, ic, k);
            w.stage_channels[s] = ic;
            w.stage_kernel[s] = k;
            w.stage_stride[s] = k / 2; // kernel = 2 × stride
            printf("  Stage %d conv: %d→%d K=%d stride=%d\n", s, ic, oc, k, k/2);
            
            // ResStack: 6 layers of dilated+undilated convs
            int rs_idx = idx + 1;
            for (int d = 0; d < 6; d++) {
                // First conv (dilated): layers.d.2
                snprintf(prefix, sizeof(prefix), "audio_encoder.generator.%d.layers.%d.2", rs_idx, d);
                int oc1, ic1, k1;
                load_wn_conv(sf, prefix, w.rs_w1_g[s][d], w.rs_w1_v[s][d], w.rs_b1[s][d], oc1, ic1, k1);
                
                // Second conv (undilated): layers.d.5
                snprintf(prefix, sizeof(prefix), "audio_encoder.generator.%d.layers.%d.5", rs_idx, d);
                int oc2, ic2, k2;
                load_wn_conv(sf, prefix, w.rs_w2_g[s][d], w.rs_w2_v[s][d], w.rs_b2[s][d], oc2, ic2, k2);
            }
        } else {
            // Output conv (stage 6): 768→128
            snprintf(prefix, sizeof(prefix), "audio_encoder.generator.%d.layer", idx);
            int oc, ic, k;
            load_wn_conv(sf, prefix, w.out_conv_wg, w.out_conv_wv, w.out_conv_bias, oc, ic, k);
            printf("  Output conv: %d→%d K=%d\n", ic, oc, k);
        }
    }
    
    sf.close();
    
    // Try to load enc_mi_layer + pre_proj from separate safetensors
    std::string mi_path = std::string(safetensors_path);
    size_t pos = mi_path.rfind("encoder_det");
    if (pos != std::string::npos) {
        mi_path.replace(pos, 11, "enc_mi_pre_proj");
    } else {
        mi_path = "models/enc_mi_pre_proj.safetensors";
    }
    SafeTensorsFile sf2;
    if (sf2.open(mi_path.c_str())) {
        // enc_mi_layer.0: Linear(128→512)
        auto info = sf2.find("enc_mi_layer.0.weight");
        if (info) { w.mi_w1 = new float[512*128]; sf2.load_raw(*info, w.mi_w1, 512*128); }
        info = sf2.find("enc_mi_layer.0.bias");
        if (info) { w.mi_b1 = new float[512]; sf2.load_raw(*info, w.mi_b1, 512); }
        // LSTM weights
        for (int l = 0; l < 4; l++) {
            char key[128];
            snprintf(key, sizeof(key), "enc_mi_layer.1.lstm.weight_ih_l%d", l);
            info = sf2.find(key); if (info) { w.mi_lstm_w_ih[l] = new float[2048*512]; sf2.load_raw(*info, w.mi_lstm_w_ih[l], 2048*512); }
            snprintf(key, sizeof(key), "enc_mi_layer.1.lstm.weight_hh_l%d", l);
            info = sf2.find(key); if (info) { w.mi_lstm_w_hh[l] = new float[2048*512]; sf2.load_raw(*info, w.mi_lstm_w_hh[l], 2048*512); }
            snprintf(key, sizeof(key), "enc_mi_layer.1.lstm.bias_ih_l%d", l);
            info = sf2.find(key); if (info) { w.mi_lstm_b_ih[l] = new float[2048]; sf2.load_raw(*info, w.mi_lstm_b_ih[l], 2048); }
            snprintf(key, sizeof(key), "enc_mi_layer.1.lstm.bias_hh_l%d", l);
            info = sf2.find(key); if (info) { w.mi_lstm_b_hh[l] = new float[2048]; sf2.load_raw(*info, w.mi_lstm_b_hh[l], 2048); }
        }
        // enc_mi_layer.2: Linear(512→128)
        info = sf2.find("enc_mi_layer.2.weight");
        if (info) { w.mi_w2 = new float[128*512]; sf2.load_raw(*info, w.mi_w2, 128*512); }
        info = sf2.find("enc_mi_layer.2.bias");
        if (info) { w.mi_b2 = new float[128]; sf2.load_raw(*info, w.mi_b2, 128); }
        // pre_proj: Conv1d(128→256, K=1)
        info = sf2.find("pre_proj.weight");
        if (info) { w.pre_proj_w = new float[256*128]; sf2.load_raw(*info, w.pre_proj_w, 256*128); }
        info = sf2.find("pre_proj.bias");
        if (info) { w.pre_proj_b = new float[256]; sf2.load_raw(*info, w.pre_proj_b, 256); }
        sf2.close();
        printf("  enc_mi_layer + pre_proj: loaded\n");
    }
    
    w.loaded = true;
    printf("  AudioVAE encoder: loaded\n");
    return true;
}

void audiovae_encoder_free(AudioVAEEncoderWeights & w) {
    #define SAFE_DEL(x) delete[] x; x = nullptr
    SAFE_DEL(w.pre_conv_wg); SAFE_DEL(w.pre_conv_wv); SAFE_DEL(w.pre_conv_bias);
    for (int s = 0; s < 7; s++) {
        SAFE_DEL(w.stage_conv_wg[s]); SAFE_DEL(w.stage_conv_wv[s]); SAFE_DEL(w.stage_conv_bias[s]);
        for (int d = 0; d < 6; d++) {
            SAFE_DEL(w.rs_w1_g[s][d]); SAFE_DEL(w.rs_w1_v[s][d]); SAFE_DEL(w.rs_b1[s][d]);
            SAFE_DEL(w.rs_w2_g[s][d]); SAFE_DEL(w.rs_w2_v[s][d]); SAFE_DEL(w.rs_b2[s][d]);
        }
    }
    SAFE_DEL(w.out_conv_wg); SAFE_DEL(w.out_conv_wv); SAFE_DEL(w.out_conv_bias);
}

// ============ Forward pass ============

bool audiovae_encode(const AudioVAEEncoderWeights & w, const float * audio, int n_samples,
                     float * latents_out, int * n_frames_out) {
    if (!w.loaded) return false;
    
    // Input: [1, T] audio
    int C = 1, T = n_samples;
    
    // Pre-conv: 1→12, K=3, stride=1
    int C_pre = 12;
    int T_pre = T; // stride=1
    float * pre_out = new float[C_pre * T_pre];
    conv1d_causal_pt(pre_out, audio, 1, T, w.pre_conv_wv, w.pre_conv_bias, 12, 3, 1, 1);
    { float rms=0; for(int i=0;i<12*T;i++) rms+=pre_out[i]*pre_out[i];
      printf("  enc pre-conv: rms=%.4f\n", sqrtf(rms/(12*T))); }
    leaky_relu(pre_out, C_pre * T_pre);
    
    float * cur = pre_out;
    int cur_C = C_pre, cur_T = T_pre;
    
    // 7 stages
    for (int s = 0; s < 6; s++) {
        int in_C = cur_C;
        int stride = w.stage_stride[s];
        int K = w.stage_kernel[s];
        int out_C = (s == 0) ? 24 : (s == 1) ? 48 : (s == 2) ? 96 : (s == 3) ? 192 : (s == 4) ? 384 : 768;
        
        // Transition conv
        int new_T = (cur_T + K - 1 - (K - 1)) / stride + 1; // causal: pad_left = K-1
        // Actually: conv with left_pad = dil*(K-1) = K-1, stride
        new_T = (cur_T + (K-1) - (K-1)) / stride + 1; // simplified: causal padding doesn't add length
        new_T = (cur_T - 1) / stride + 1;
        
        float * conv_out = new float[out_C * new_T];
        conv1d_causal_pt(conv_out, cur, in_C, cur_T,
                         w.stage_conv_wv[s], w.stage_conv_bias[s],
                         out_C, K, stride, 1);
        { float rms=0; for(int i=0;i<out_C*new_T;i++) rms+=conv_out[i]*conv_out[i];
          printf("  stage%d conv: rms=%.4f len=%d\n", s, sqrtf(rms/(out_C*new_T)), new_T); }
        
        // ResStack: 6 dilated residual blocks
        for (int d = 0; d < 6; d++) {
            int dil = 1 << d; // 1, 2, 4, 8, 16, 32
            
            // Save ORIGINAL for residual connection (don't modify!)
            float * orig = new float[out_C * new_T];
            memcpy(orig, conv_out, out_C * new_T * sizeof(float));
            
            // Block input: copy of current (will be mutated by LeakyReLU)
            float * x_block = new float[out_C * new_T];
            memcpy(x_block, conv_out, out_C * new_T * sizeof(float));
            
            // LeakyReLU → dilated conv → LeakyReLU → undilated conv
            leaky_relu(x_block, out_C * new_T, 0.01f);
            
            float * tmp = new float[out_C * new_T];
            conv1d_causal_pt(tmp, x_block, out_C, new_T,
                            w.rs_w1_v[s][d], w.rs_b1[s][d],
                            out_C, 3, 1, dil);
            
            leaky_relu(tmp, out_C * new_T, 0.01f);
            conv1d_causal_pt(tmp, tmp, out_C, new_T,
                            w.rs_w2_v[s][d], w.rs_b2[s][d],
                            out_C, 3, 1, 1);
            
            // Residual: conv_out = ORIGINAL + block_output
            for (int i = 0; i < out_C * new_T; i++)
                conv_out[i] = orig[i] + tmp[i];
            
            delete[] orig; delete[] x_block; delete[] tmp;
        }
        
        // LeakyReLU after ResStack
        leaky_relu(conv_out, out_C * new_T);
        { float rms=0; for(int i=0;i<out_C*new_T;i++) rms+=conv_out[i]*conv_out[i];
          printf("  stage%d after RS+ReLU: rms=%.4f\n", s, sqrtf(rms/(out_C*new_T))); }
        
        delete[] cur; // free previous buffer (except first pre_out which will be freed)
        cur = conv_out;
        cur_C = out_C; cur_T = new_T;
    }
    
    // Output conv: 768→128, K=5, stride=1, NON-CAUSAL with pad=2
    int final_T = cur_T;
    float * final_out = new float[128 * final_T];
    {
        int ic=cur_C, oc=128, K=5, pad=2;
        int padded_T = cur_T + 2*pad;
        float *padded = new float[ic * padded_T]();
        for(int c=0;c<ic;c++) {
            memset(padded + c*padded_T, 0, pad*sizeof(float));
            memcpy(padded + c*padded_T + pad, cur + c*cur_T, cur_T*sizeof(float));
            memset(padded + c*padded_T + pad + cur_T, 0, pad*sizeof(float));
        }
        for(int o=0;o<oc;o++) {
            float b = w.out_conv_bias ? w.out_conv_bias[o] : 0;
            for(int t=0;t<cur_T;t++) {
                float s = b;
                for(int k=0;k<K;k++) {
                    int pt = t + k;
                    for(int c=0;c<ic;c++)
                        s += padded[c*padded_T + pt] * w.out_conv_wv[(o*ic + c)*K + k];
                }
                final_out[o*cur_T + t] = s;
            }
        }
        delete[] padded;
    }
    
    delete[] cur;
    
    // === enc_mi_layer: Linear(128→512) → LSTM(4×512,skip) → Linear(512→128) ===
    float * mi_latents = final_out; // reuse buffer: 128 * final_T
    if (w.mi_w1) {
        int T = final_T;
        // Linear 128→512
        float * mi_buf = new float[T * 512];
        for (int t = 0; t < T; t++)
            for (int o = 0; o < 512; o++) {
                float s = w.mi_b1[o];
                for (int i = 0; i < 128; i++)
                    s += final_out[t * 128 + i] * w.mi_w1[o * 128 + i];
                mi_buf[t * 512 + o] = s;
            }
        // LSTM
        const float *w_ih[4]={w.mi_lstm_w_ih[0],w.mi_lstm_w_ih[1],w.mi_lstm_w_ih[2],w.mi_lstm_w_ih[3]};
        const float *w_hh[4]={w.mi_lstm_w_hh[0],w.mi_lstm_w_hh[1],w.mi_lstm_w_hh[2],w.mi_lstm_w_hh[3]};
        const float *b_ih[4]={w.mi_lstm_b_ih[0],w.mi_lstm_b_ih[1],w.mi_lstm_b_ih[2],w.mi_lstm_b_ih[3]};
        const float *b_hh[4]={w.mi_lstm_b_hh[0],w.mi_lstm_b_hh[1],w.mi_lstm_b_hh[2],w.mi_lstm_b_hh[3]};
        float * lstm_out = new float[T * 512];
        lstm_forward(mi_buf, T, 512, 4, w_ih, w_hh, b_ih, b_hh, lstm_out, true);
        delete[] mi_buf;
        // Linear 512→128
        for (int t = 0; t < T; t++)
            for (int o = 0; o < 128; o++) {
                float s = w.mi_b2[o];
                for (int i = 0; i < 512; i++)
                    s += lstm_out[t * 512 + i] * w.mi_w2[o * 512 + i];
                mi_latents[t * 128 + o] = s;
            }
        delete[] lstm_out;
        printf("  enc_mi_layer: done\\n");
    }
    
    // === pre_proj: Conv1d(128→256, K=1) ===
    if (w.pre_proj_w) {
        float * proj_out = new float[256 * final_T];
        for (int t = 0; t < final_T; t++)
            for (int o = 0; o < 256; o++) {
                float s = w.pre_proj_b ? w.pre_proj_b[o] : 0;
                for (int i = 0; i < 128; i++)
                    s += mi_latents[t * 128 + i] * w.pre_proj_w[o * 128 + i];
                proj_out[o * final_T + t] = s;  // channel-major for Conv1d output
            }
        // Extract mean (first 128 of 256 channels)
        for (int t = 0; t < final_T; t++)
            memcpy(mi_latents + t * 128, proj_out + t, 128 * sizeof(float)); // wait, need proper transposition
        // Actually proj_out is [C,T] channel-major, output is [T,C] time-major
        for (int t = 0; t < final_T; t++)
            for (int c = 0; c < 128; c++)
                mi_latents[t * 128 + c] = proj_out[c * final_T + t];
        delete[] proj_out;
        printf("  pre_proj: done\\n");
    }

    // Copy to output
    *n_frames_out = final_T;
    memcpy(latents_out, final_out, 128 * final_T * sizeof(float));
    delete[] final_out;
    
    // Debug
    { float rms=0; for(int i=0;i<128*final_T;i++) rms+=latents_out[i]*latents_out[i];
      printf("  Encoder output: %d frames, RMS=%.4f\n", final_T, sqrtf(rms/(128*final_T))); }
    
    return true;
}
