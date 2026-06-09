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

// Weight norm: actual_weight = g * v / ||v||
static float wn_scale(float g, const float * v, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) norm += v[i] * v[i];
    norm = sqrtf(norm + 1e-12f);
    return g / norm;
}

// Causal Conv1d with weight norm (weight_g + weight_v)
// Layout: in is [C_in, T], out is [C_out, T_out]
static void causal_conv1d_wn(float * out, const float * in, int C_in, int T,
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
    causal_conv1d_wn(pre_out, audio, 1, T, w.pre_conv_wg, w.pre_conv_wv, w.pre_conv_bias, 12, 3, 1, 1);
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
        causal_conv1d_wn(conv_out, cur, in_C, cur_T,
                         w.stage_conv_wg[s], w.stage_conv_wv[s], w.stage_conv_bias[s],
                         out_C, K, stride, 1);
        
        // ResStack: 6 dilated residual blocks
        for (int d = 0; d < 6; d++) {
            int dil = 1 << d; // 1, 2, 4, 8, 16, 32
            
            // Save input for residual
            float * residual = new float[out_C * new_T];
            memcpy(residual, conv_out, out_C * new_T * sizeof(float));
            
            // LeakyReLU → dilated conv → LeakyReLU → undilated conv
            // NOTE: The Python code applies LeakyReLU BEFORE each conv in ResStack
            // But the weight layout has layers.d.2 (first conv) and layers.d.5 (second conv)
            // with LeakyReLU in between
            
            float * tmp = new float[out_C * new_T];
            
            // First part: LeakyReLU + dilated conv
            leaky_relu(residual, out_C * new_T);
            causal_conv1d_wn(tmp, residual, out_C, new_T,
                            w.rs_w1_g[s][d], w.rs_w1_v[s][d], w.rs_b1[s][d],
                            out_C, 3, 1, dil);
            
            // LeakyReLU + undilated conv
            leaky_relu(tmp, out_C * new_T);
            causal_conv1d_wn(tmp, tmp, out_C, new_T,
                            w.rs_w2_g[s][d], w.rs_w2_v[s][d], w.rs_b2[s][d],
                            out_C, 3, 1, 1);
            
            // Residual: conv_out = residual + block_output
            for (int i = 0; i < out_C * new_T; i++)
                conv_out[i] = residual[i] + tmp[i];
            
            delete[] tmp;
        }
        
        // LeakyReLU after ResStack
        leaky_relu(conv_out, out_C * new_T);
        
        delete[] cur; // free previous buffer (except first pre_out which will be freed)
        cur = conv_out;
        cur_C = out_C; cur_T = new_T;
    }
    
    // Output conv: cur_C→128, K=3, stride=1 (using stage 6 = output layer)
    // Actually the output is at index 20 with K=5 in the safetensors
    // Let me use a separate output conv
    int final_T = cur_T;
    float * final_out = new float[128 * final_T];
    
    // Output projection: look at generator.20
    // Use out_conv from the weights
    {
        // The output weight is [128, 768, 5] — K=5, stride=1
        causal_conv1d_wn(final_out, cur, cur_C, cur_T,
                         w.out_conv_wg, w.out_conv_wv, w.out_conv_bias,
                         128, 5, 1, 1);
    }
    
    delete[] cur;
    
    // Copy to output
    *n_frames_out = final_T;
    memcpy(latents_out, final_out, 128 * final_T * sizeof(float));
    delete[] final_out;
    
    // Debug
    { float rms=0; for(int i=0;i<128*final_T;i++) rms+=latents_out[i]*latents_out[i];
      printf("  Encoder output: %d frames, RMS=%.4f\n", final_T, sqrtf(rms/(128*final_T))); }
    
    return true;
}
