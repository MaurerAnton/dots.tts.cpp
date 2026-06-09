// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// BigVGAN calibration — stage-by-stage comparison with Python reference
#include "bigvgan_cpp.h"
#include "dots_tts.h"
#include "safetensors.h"
#include <cstdio>
#include <cmath>
#include <cstring>

// Forward declarations from bigvgan_cpp.cpp (must match exact signatures)
void lstm_forward(const float * x, int seq_len, int hidden, int n_layers,
    const float * w_ih[4], const float * w_hh[4],
    const float * b_ih[4], const float * b_hh[4], float * out, bool skip);
void conv1d_causal(float * out, const float * in, int ic, int ilen,
    const float * w, const float * bias, int oc, int K, int dil=1);
void convT1d_causal(float * out, const float * in, int ic, int ilen,
    const float * w, const float * bias, int oc, int stride, int K, int * olen);
static const float AA_FILTER[12] = {0}; // dummy, actual from bigvgan_cpp.cpp
void aa_snakebeta(float * x, int cur_len, int ch,
    const float * alpha, const float * beta, bool logscale,
    float * buf_up, float * buf_down, int max_up_len,
    const float * filter_up = nullptr, const float * filter_down = nullptr, int filter_ch = 1);

// Load .npy file (simple: 1D or 2D float32, no header parsing — we know shapes)
static float * load_npy(const char * path, int expected_n) {
    // .npy files have a 128-byte header — skip to raw data
    FILE * f = fopen(path, "rb");
    if (!f) { printf("  MISSING: %s\n", path); return nullptr; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    // Find data start: .npy magic is \x93NUMPY, then header len
    fseek(f, 0, SEEK_SET);
    char magic[6]; fread(magic, 1, 6, f);
    if (magic[0] != (char)0x93 || strncmp(magic+1, "NUMPY", 5) != 0) {
        // Maybe no header — just raw floats
        fseek(f, 0, SEEK_SET);
    } else {
        unsigned char ver_major, ver_minor;
        fread(&ver_major, 1, 1, f);
        fread(&ver_minor, 1, 1, f);
        unsigned short header_len;
        fread(&header_len, 2, 1, f);
        fseek(f, 6 + 2 + header_len, SEEK_SET); // skip header
    }
    int remaining = (sz - ftell(f)) / sizeof(float);
    float * data = new float[remaining];
    fread(data, sizeof(float), remaining, f);
    fclose(f);
    return data;
}

static float compute_rms(const float * x, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    return (float)sqrt(s / n);
}

static float compute_corr(const float * a, const float * b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return (float)(dot / sqrt(na * nb));
}

int main() {
    const char * sf_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/vocoder.safetensors";
    const char * ref_dir = "debug";
    
    // Load weights
    BigVGANDecoder dec;
    if (!bigvgan_load(sf_path, dec)) return 1;
    
    // Load Python reference latents
    float * latents = new float[16 * 128];
    FILE * f = fopen("/tmp/ref_latents.bin", "rb");
    fread(latents, sizeof(float), 16*128, f); fclose(f);
    int n_frames = 16;
    
    int hop = VAE_HOP_SAMPLES;
    int max_ch = 1536, max_len = n_frames * 1920 + 1024;
    dec.buf1.resize(max_len * max_ch);
    dec.buf2.resize(max_len * max_ch);
    dec.buf3.resize(max_len * max_ch);
    dec.buf_act_up.resize(max_len * max_ch * 2);
    dec.buf_act_down.resize(max_len * max_ch * 2);
    dec.buf_amp_save.resize(max_len * max_ch);
    
    float * x = dec.buf1.data(), * tmp = dec.buf2.data(), * tmp2 = dec.buf3.data();
    float * amp_save = dec.buf_amp_save.data();
    float * act_up = dec.buf_act_up.data(), * act_down = dec.buf_act_down.data();
    
    printf("=== BigVGAN Stage-by-Stage Calibration ===\n\n");
    
    // --- Step 1: post_proj ---
    conv1d_causal(tmp, latents, 128, n_frames, dec.post_proj_w.ptr(), dec.post_proj_b.ptr(), 128, 1);
    {
        float * ref = load_npy("debug/ref_postproj.npy", n_frames * 128);
        if (ref) {
            float cpp_rms = compute_rms(tmp, n_frames*128);
            float py_rms = compute_rms(ref, n_frames*128);
            float corr = compute_corr(tmp, ref, n_frames*128);
            printf("post_proj      : RMS C++=%.4f Py=%.4f  corr=%+.4f %s\n",
                   cpp_rms, py_rms, corr, fabsf(corr) > 0.999 ? "✓" : "✗");
            delete[] ref;
        }
    }
    
    // --- Step 2: dec_mi_layer ---
    float * mi_buf = (float*)malloc(n_frames * 512 * sizeof(float));
    for (int t = 0; t < n_frames; t++)
        for (int o = 0; o < 512; o++) {
            float s = dec.mi_b1.ptr()[o];
            for (int i = 0; i < 128; i++)
                s += tmp[t * 128 + i] * dec.mi_w1.ptr()[o * 128 + i];
            mi_buf[t * 512 + o] = s;
        }
    const float * w_ih[4] = {dec.mi_lstm_w_ih[0].ptr(), dec.mi_lstm_w_ih[1].ptr(),
                              dec.mi_lstm_w_ih[2].ptr(), dec.mi_lstm_w_ih[3].ptr()};
    const float * w_hh[4] = {dec.mi_lstm_w_hh[0].ptr(), dec.mi_lstm_w_hh[1].ptr(),
                              dec.mi_lstm_w_hh[2].ptr(), dec.mi_lstm_w_hh[3].ptr()};
    const float * b_ih[4] = {dec.mi_lstm_b_ih[0].ptr(), dec.mi_lstm_b_ih[1].ptr(),
                              dec.mi_lstm_b_ih[2].ptr(), dec.mi_lstm_b_ih[3].ptr()};
    const float * b_hh[4] = {dec.mi_lstm_b_hh[0].ptr(), dec.mi_lstm_b_hh[1].ptr(),
                              dec.mi_lstm_b_hh[2].ptr(), dec.mi_lstm_b_hh[3].ptr()};
    float * lstm_out = (float*)malloc(n_frames * 512 * sizeof(float));
    lstm_forward(mi_buf, n_frames, 512, 4, w_ih, w_hh, b_ih, b_hh, lstm_out, true);
    for (int t = 0; t < n_frames; t++)
        for (int o = 0; o < 128; o++) {
            float s = dec.mi_b2.ptr()[o];
            for (int i = 0; i < 512; i++)
                s += lstm_out[t * 512 + i] * dec.mi_w2.ptr()[o * 512 + i];
            tmp[t * 128 + o] = s;
        }
    free(mi_buf); free(lstm_out);
    {
        float * ref = load_npy("debug/ref_milayer.npy", n_frames * 128);
        if (ref) {
            float cpp_rms = compute_rms(tmp, n_frames*128);
            float py_rms = compute_rms(ref, n_frames*128);
            float corr = compute_corr(tmp, ref, n_frames*128);
            printf("dec_mi_layer   : RMS C++=%.4f Py=%.4f  corr=%+.4f %s\n",
                   cpp_rms, py_rms, corr, fabsf(corr) > 0.999 ? "✓" : "✗");
            delete[] ref;
        }
    }
    
    // --- Step 3: conv_pre ---
    int conv_pre_ch = 1536;
    {
        int ic = 128, oc = conv_pre_ch, K = 5, pad = 2;
        int padded_len = n_frames + 2 * pad;
        float * padded = (float*)malloc(ic * padded_len * sizeof(float));
        memset(padded, 0, ic * pad * sizeof(float));
        memcpy(padded + ic * pad, tmp, ic * n_frames * sizeof(float));
        memset(padded + ic * (pad + n_frames), 0, ic * pad * sizeof(float));
        for (int o = 0; o < oc; o++) {
            float b = dec.conv_pre_b.ptr() ? dec.conv_pre_b.ptr()[o] : 0;
            for (int t = 0; t < n_frames; t++) {
                float s = b;
                for (int k = 0; k < K; k++) {
                    int pt = t + k;
                    if (pt >= 0 && pt < padded_len)
                        for (int c = 0; c < ic; c++)
                            s += padded[pt * ic + c] * dec.conv_pre_w.ptr()[(o * ic + c) * K + k];
                }
                x[t * oc + o] = s;
            }
        }
        free(padded);
    }
    {
        float * ref = load_npy("debug/ref_convpre.npy", n_frames * conv_pre_ch);
        if (ref) {
            float cpp_rms = compute_rms(x, n_frames*conv_pre_ch);
            float py_rms = compute_rms(ref, n_frames*conv_pre_ch);
            float corr = compute_corr(x, ref, n_frames*conv_pre_ch);
            printf("conv_pre       : RMS C++=%.4f Py=%.4f  corr=%+.4f %s\n",
                   cpp_rms, py_rms, corr, fabsf(corr) > 0.999 ? "✓" : "✗");
            if (fabsf(corr) < 0.999) {
                // Check per-channel
                float ch_corrs[10];
                for (int c = 0; c < 10; c++) {
                    float * cpp_ch = x + c * n_frames;
                    float * py_ch = ref + c * n_frames;
                    ch_corrs[c] = compute_corr(cpp_ch, py_ch, n_frames);
                }
                printf("  First 10 channels corr:");
                for (int c = 0; c < 10; c++) printf(" %.3f", ch_corrs[c]);
                printf("\n");
            }
            delete[] ref;
        }
    }
    
    int cur_ch = conv_pre_ch, cur_len = n_frames;
    
    // --- Steps 4-9: 6 upsampling stages ---
    int rb_dilations[3] = {1, 3, 5};
    for (int s = 0; s < 6; s++) {
        char path[256];
        
        // Upsample
        int stride = dec.ups_stride[s], K = dec.ups_kernel[s];
        int out_ch = dec.ups_out[s];
        int up_len;
        convT1d_causal(tmp, x, cur_ch, cur_len, dec.ups_w[s].ptr(), dec.ups_b[s].ptr(), out_ch, stride, K, &up_len);
        cur_ch = out_ch; cur_len = up_len;
        
        snprintf(path, sizeof(path), "%s/ref_stage%d_ups.npy", ref_dir, s);
        {
            float * ref = load_npy(path, cur_ch * cur_len);
            if (ref) {
                float cpp_rms = compute_rms(tmp, cur_ch*cur_len);
                float py_rms = compute_rms(ref, cur_ch*cur_len);
                float corr = compute_corr(tmp, ref, cur_ch*cur_len);
                printf("stage%d ups     : RMS C++=%.4f Py=%.4f  corr=%+.4f %s\n",
                       s, cpp_rms, py_rms, corr, fabsf(corr) > 0.99 ? "✓" : "✗");
                delete[] ref;
            }
        }
        
        // AMP blocks
        memset(x, 0, cur_len * cur_ch * sizeof(float));
        for (int j = 0; j < 3; j++) {
            int rb = s * 3 + j;
            int kernel = (j == 0) ? 3 : (j == 1) ? 7 : 11;
            float * residual = tmp2;
            memcpy(residual, tmp, cur_len * cur_ch * sizeof(float));
            for (int pair = 0; pair < 3; pair++) {
                int dil = rb_dilations[pair];
                memcpy(act_down, residual, cur_len * cur_ch * sizeof(float));
                memcpy(act_up, residual, cur_len * cur_ch * sizeof(float));
                aa_snakebeta(act_up, cur_len, cur_ch,
                             dec.rb_alpha[rb][pair*2].ptr(), dec.rb_beta[rb][pair*2].ptr(),
                             true, residual, amp_save, max_len * 2);
                conv1d_causal(residual, act_up, cur_ch, cur_len,
                             dec.rb_conv1_w[rb][pair].ptr(), dec.rb_conv1_b[rb][pair].ptr(),
                             cur_ch, kernel, dil);
                aa_snakebeta(residual, cur_len, cur_ch,
                             dec.rb_alpha[rb][pair*2+1].ptr(), dec.rb_beta[rb][pair*2+1].ptr(),
                             true, act_up, amp_save, max_len * 2);
                conv1d_causal(act_up, residual, cur_ch, cur_len,
                             dec.rb_conv2_w[rb][pair].ptr(), dec.rb_conv2_b[rb][pair].ptr(),
                             cur_ch, kernel, 1);
                for (int i = 0; i < cur_len * cur_ch; i++)
                    residual[i] = act_down[i] + act_up[i];
            }
            for (int i = 0; i < cur_len * cur_ch; i++) x[i] += residual[i];
        }
        for (int i = 0; i < cur_len * cur_ch; i++) x[i] /= 3.0f;
        
        snprintf(path, sizeof(path), "%s/ref_stage%d_amp.npy", ref_dir, s);
        {
            float * ref = load_npy(path, cur_ch * cur_len);
            if (ref) {
                float cpp_rms = compute_rms(x, cur_ch*cur_len);
                float py_rms = compute_rms(ref, cur_ch*cur_len);
                float corr = compute_corr(x, ref, cur_ch*cur_len);
                printf("stage%d amp     : RMS C++=%.4f Py=%.4f  corr=%+.4f %s\n",
                       s, cpp_rms, py_rms, corr, fabsf(corr) > 0.99 ? "✓" : "✗");
                delete[] ref;
            }
        }
    }
    
    // --- activation_post ---
    aa_snakebeta(x, cur_len, cur_ch, dec.act_post_alpha.ptr(), dec.act_post_beta.ptr(),
                 true, act_up, act_down, max_len * 2,
                 dec.act_post_filter_up.ptr(), dec.act_post_filter_down.ptr(), cur_ch);
    {
        float * ref = load_npy("debug/ref_actpost.npy", cur_ch * cur_len);
        if (ref) {
            float cpp_rms = compute_rms(x, cur_ch*cur_len);
            float py_rms = compute_rms(ref, cur_ch*cur_len);
            float corr = compute_corr(x, ref, cur_ch*cur_len);
            printf("act_post       : RMS C++=%.4f Py=%.4f  corr=%+.4f %s\n",
                   cpp_rms, py_rms, corr, fabsf(corr) > 0.99 ? "✓" : "✗");
            delete[] ref;
        }
    }
    
    // --- conv_post ---
    conv1d_causal(tmp, x, cur_ch, cur_len, dec.conv_post_w.ptr(), nullptr, 1, 7);
    {
        float * ref = load_npy("debug/ref_convpost.npy", cur_len);
        if (ref) {
            float cpp_rms = compute_rms(tmp, cur_len);
            float py_rms = compute_rms(ref, cur_len);
            float corr = compute_corr(tmp, ref, cur_len);
            printf("conv_post      : RMS C++=%.4f Py=%.4f  corr=%+.4f %s\n",
                   cpp_rms, py_rms, corr, fabsf(corr) > 0.99 ? "✓" : "✗");
            delete[] ref;
        }
    }
    
    // --- Final (after clamp) ---
    const float output_gain = 1.0f; // no gain for calibration
    for (int i = 0; i < cur_len; i++) {
        tmp[i] *= output_gain;
        if (tmp[i] > 1.0f) tmp[i] = 1.0f;
        if (tmp[i] < -1.0f) tmp[i] = -1.0f;
    }
    {
        float * ref = load_npy("debug/ref_final.npy", cur_len);
        if (ref) {
            float cpp_rms = compute_rms(tmp, cur_len);
            float py_rms = compute_rms(ref, cur_len);
            float corr = compute_corr(tmp, ref, cur_len);
            printf("FINAL          : RMS C++=%.4f Py=%.4f  corr=%+.4f %s\n",
                   cpp_rms, py_rms, corr, fabsf(corr) > 0.99 ? "✓ MATCH!" : "✗");
            delete[] ref;
        }
    }
    
    printf("\nDone.\n");
    bigvgan_free(dec);
    delete[] latents;
    return 0;
}
