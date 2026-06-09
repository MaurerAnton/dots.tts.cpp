// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// dots.tts.cpp - Pure C++ BigVGAN vocoder decoder (v2 — correct architecture)
// Causal mode, snake_logscale, anti-alias up/down sampling, correct AMP blocks
#include "dots_tts.h"
#include "bigvgan_cpp.h"
#include "safetensors.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

// LSTM forward declaration
void lstm_forward(const float * x, int seq_len, int hidden, int n_layers,
    const float * w_ih[4], const float * w_hh[4],
    const float * b_ih[4], const float * b_hh[4],
    float * out, bool skip);

// ============ Hardcoded anti-alias filter (Kaiser-sinc, 12-tap, causal) ============
// Computed by kaiser_sinc_filter1d(cutoff=0.25, half_width=0.3, kernel_size=12)
// All 216 AMP activation filters are identical to this
static const float AA_FILTER[12] = {
    0.00202896f, 0.00938947f, -0.02554346f, -0.05765738f,
    0.12857258f, 0.44320980f, 0.44320980f, 0.12857258f,
    -0.05765738f, -0.02554346f, 0.00938947f, 0.00202896f
};

// ============ Activation1d: upsample x2 → snakebeta → downsample x2 ============
// Reference: Python Activation1d.forward():
//   x = self.upsample(x)    # pad replicate → conv_transpose1d(filter, stride=2, groups=C) * ratio → trim
//   x = self.act(x)          # SnakeBeta (with exp if logscale)
//   return self.downsample(x) # pad replicate → conv1d(filter, stride=2, groups=C)

// Causal UpSample1d: NO pad → conv_transpose1d → scale ×2 → trim last (K-stride)
// filter: [filter_ch, 1, K] — for AMP filter_ch=1, for activation_post filter_ch=ch
static void aa_upsample(float * out, const float * in, int ch, int in_len,
                         const float * filter, int filter_ch, int * olen) {
    *olen = in_len * 2;
    int K = 12, stride = 2;
    // No padding for causal mode
    // ConvTranspose1d: input=in (L samples), weight=filter[1,1,12], stride=2, groups=ch
    // Effective: each input sample i contributes to output i*stride + k for k=0..K-1
    int olen_full = (in_len - 1) * stride + K; // padding=0, output_padding=0
    memset(out, 0, ch * olen_full * sizeof(float));
    for (int i = 0; i < in_len; i++) {
        for (int k = 0; k < K; k++) {
            int ot = i * stride + k;
            if (ot >= 0 && ot < olen_full) {
                for (int c = 0; c < ch; c++) {
                    float f = (filter_ch == 1) ? filter[k] : filter[c * K + k];
                    out[ot * ch + c] += in[i * ch + c] * f * 2.0f; // ratio=2 scale
                }
            }
        }
    }
    // Causal trim: remove last (K - stride) = 10 samples
    if (*olen < olen_full) {
        memmove(out, out, ch * (*olen) * sizeof(float));
    }
}

// Causal LowPassFilter1d (DownSample1d): pad([K-1], [replicate]) → conv1d(stride=2, groups=ch)
// filter: [filter_ch, 1, K]
static void aa_downsample(float * out, const float * in, int ch, int in_len,
                           const float * filter, int filter_ch, int * olen) {
    int K = 12, stride = 2;
    int pad_left = K - 1; // 11
    int padded_len = in_len + pad_left;
    float * padded = (float*)malloc(ch * padded_len * sizeof(float));
    // Replicate first sample across pad_left positions
    for (int p = 0; p < pad_left; p++)
        memcpy(padded + p * ch, in, ch * sizeof(float));
    // Copy input after pad
    memcpy(padded + pad_left * ch, in, ch * in_len * sizeof(float));
    // Conv1d with stride=2, groups=ch (depthwise)
    *olen = (padded_len - K) / stride + 1;
    for (int c = 0; c < ch; c++) {
        for (int t = 0; t < *olen; t++) {
            float s = 0;
            for (int k = 0; k < K; k++) {
                int idx = (t * stride + k) * ch + c;
                float fk = (filter_ch == 1) ? filter[k] : filter[c * K + k];
                s += padded[idx] * fk;
            }
            out[t * ch + c] = s;
        }
    }
    free(padded);
}

// Full Activation1d (union of up + snakebeta + down)
// buf_a/buf_b: temp buffers sized for upsampled output
// filter_up/down: [1,1,12] for AMP (shared), [ch,1,12] for activation_post (per-channel)
static void aa_snakebeta(float * x, int cur_len, int ch,
                          const float * alpha, const float * beta, bool logscale,
                          float * buf_up, float * buf_down, int max_up_len,
                          const float * filter_up = AA_FILTER,
                          const float * filter_down = AA_FILTER,
                          int filter_ch = 1) {
    int up_len;
    aa_upsample(buf_up, x, ch, cur_len, filter_up, filter_ch, &up_len);
    // SnakeBeta on upsampled
    for (int i = 0; i < up_len; i++) {
        for (int c = 0; c < ch; c++) {
            float a = alpha ? alpha[c] : 1.0f;
            float b = beta ? beta[c] : 1.0f;
            if (logscale) { 
                a = expf(a > 20.0f ? 20.0f : (a < -20.0f ? -20.0f : a));
                b = expf(b > 20.0f ? 20.0f : (b < -20.0f ? -20.0f : b));
            }
            float v = buf_up[i * ch + c];
            if (std::isnan(v) || std::isinf(v)) { buf_up[i * ch + c] = 0; continue; }
            // Clamp argument to sin to avoid extreme values
            float arg = a * v;
            if (arg > 100.0f) arg = 100.0f; if (arg < -100.0f) arg = -100.0f;
            float s = sinf(arg);
            float result = v + (1.0f / (b + 1e-9f)) * s * s;
            if (std::isnan(result) || std::isinf(result)) result = v;
            buf_up[i * ch + c] = result;
        }
    }
    int down_len;
    aa_downsample(buf_down, buf_up, ch, up_len, filter_down, filter_ch, &down_len);
    memcpy(x, buf_down, down_len * ch * sizeof(float));
}

// ============ SafeTensor loading ============
static BigVGANTensor load_raw_st(SafeTensorsFile & sf, const char * name) {
    BigVGANTensor t; t.n0 = t.n1 = t.n2 = 1;
    const st_tensor_info * info = sf.find(name);
    if (!info) return t;
    t.n0 = info->shape.size() > 0 ? info->shape[0] : 1;
    t.n1 = info->shape.size() > 1 ? info->shape[1] : 1;
    t.n2 = info->shape.size() > 2 ? info->shape[2] : 1;
    size_t n = t.n0 * t.n1 * t.n2;
    t.data.resize(n);
    sf.load_raw(*info, t.data.data(), n);
    return t;
}

// ============ Causal Conv1d ============
// Python causal Conv1d: pad(dil*(K-1), [0]) left-side → conv1d
static void conv1d_causal(float * out, const float * in, int ic, int ilen,
                          const float * w, const float * bias, int oc, int K, int dil=1) {
    int left_pad = dil * (K - 1);
    int padded_len = ilen + left_pad;
    float * padded = (float*)malloc(ic * padded_len * sizeof(float));
    // Pad left with zeros
    memset(padded, 0, ic * left_pad * sizeof(float));
    memcpy(padded + ic * left_pad, in, ic * ilen * sizeof(float));
    // Conv1d with stride=1, dilation=dil
    for (int o = 0; o < oc; o++) {
        float b = bias ? bias[o] : 0;
        for (int t = 0; t < ilen; t++) {
            float s = b;
            for (int k = 0; k < K; k++) {
                int pt = t + left_pad - dil * k; // causal: look back by dil*k
                if (pt >= 0) {
                    for (int c = 0; c < ic; c++)
                        s += padded[pt * ic + c] * w[(o * ic + c) * K + k];
                }
            }
            out[t * oc + o] = s;
        }
    }
    free(padded);
}

// ============ Causal ConvTranspose1d ============
// Python causal: ConvTranspose1d(padding=0) → trim last stride samples
// weight layout: [ic, oc, K] (PyTorch ConvTranspose1d native)
static void convT1d_causal(float * out, const float * in, int ic, int ilen,
                            const float * w, const float * bias, int oc, int stride, int K,
                            int * olen) {
    int olen_full = (ilen - 1) * stride + K; // padding=0
    *olen = olen_full - stride; // causal trim
    memset(out, 0, olen_full * oc * sizeof(float));
    // For each input position i:
    //   out[i*stride + k][o] += in[i][c] * weight[c][o][k]
    for (int i = 0; i < ilen; i++) {
        for (int k = 0; k < K; k++) {
            int ot = i * stride + k;
            if (ot >= 0 && ot < olen_full) {
                for (int o = 0; o < oc; o++) {
                    float s = 0;
                    for (int c = 0; c < ic; c++)
                        s += in[i * ic + c] * w[(c * oc + o) * K + k];
                    out[ot * oc + o] += s;
                }
            }
        }
    }
    // Add bias after all contributions
    if (bias) {
        for (int t = 0; t < *olen; t++)
            for (int o = 0; o < oc; o++)
                out[t * oc + o] += bias[o];
    }
    // Trim: keep only first *olen samples
    if (*olen < olen_full) {
        memmove(out, out, (*olen) * oc * sizeof(float));
    }
}

// ============ Load weights ============
bool bigvgan_load(const char * sf_path, BigVGANDecoder & dec) {
    SafeTensorsFile sf;
    if (!sf.open(sf_path)) return false;

    dec.conv_pre_w = load_raw_st(sf, "decoder.conv_pre.weight");
    dec.conv_pre_b = load_raw_st(sf, "decoder.conv_pre.bias");

    int up_in[] = {1536, 768, 384, 192, 96, 48};
    int up_out[] = {768, 384, 192, 96, 48, 24};
    int up_strides[] = {10, 6, 4, 2, 2, 2};
    int up_kernels[] = {20, 12, 8, 4, 4, 4};
    for (int i = 0; i < 6; i++) {
        char name[128];
        snprintf(name, sizeof(name), "decoder.ups.%d.0.weight", i);
        dec.ups_w[i] = load_raw_st(sf, name);
        snprintf(name, sizeof(name), "decoder.ups.%d.0.bias", i);
        dec.ups_b[i] = load_raw_st(sf, name);
        dec.ups_in[i] = up_in[i]; dec.ups_out[i] = up_out[i];
        dec.ups_stride[i] = up_strides[i]; dec.ups_kernel[i] = up_kernels[i];
    }

    // 18 resblocks (6 stages × 3 kernel sizes)
    // AMPBlock1: convs1 (3 dilated convs) + convs2 (3 dil=1 convs) + 6 activations
    for (int rb = 0; rb < 18; rb++) {
        for (int j = 0; j < 3; j++) {
            char name[128];
            snprintf(name, sizeof(name), "decoder.resblocks.%d.convs1.%d.weight", rb, j);
            dec.rb_conv1_w[rb][j] = load_raw_st(sf, name);
            snprintf(name, sizeof(name), "decoder.resblocks.%d.convs1.%d.bias", rb, j);
            dec.rb_conv1_b[rb][j] = load_raw_st(sf, name);
            snprintf(name, sizeof(name), "decoder.resblocks.%d.convs2.%d.weight", rb, j);
            dec.rb_conv2_w[rb][j] = load_raw_st(sf, name);
            snprintf(name, sizeof(name), "decoder.resblocks.%d.convs2.%d.bias", rb, j);
            dec.rb_conv2_b[rb][j] = load_raw_st(sf, name);
        }
        // 6 activations per resblock
        for (int a = 0; a < 6; a++) {
            char name[128];
            snprintf(name, sizeof(name), "decoder.resblocks.%d.activations.%d.act.alpha", rb, a);
            dec.rb_alpha[rb][a] = load_raw_st(sf, name);
            snprintf(name, sizeof(name), "decoder.resblocks.%d.activations.%d.act.beta", rb, a);
            dec.rb_beta[rb][a] = load_raw_st(sf, name);
        }
    }

    dec.conv_post_w = load_raw_st(sf, "decoder.conv_post.weight");
    // conv_post has NO bias (use_bias_at_final=False)

    dec.act_post_alpha = load_raw_st(sf, "decoder.activation_post.act.alpha");
    dec.act_post_beta  = load_raw_st(sf, "decoder.activation_post.act.beta");
    dec.act_post_filter_up = load_raw_st(sf, "decoder.activation_post.upsample.filter");
    dec.act_post_filter_down = load_raw_st(sf, "decoder.activation_post.downsample.lowpass.filter");

    // post_proj + dec_mi_layer
    dec.post_proj_w = load_raw_st(sf, "post_proj.weight");
    dec.post_proj_b = load_raw_st(sf, "post_proj.bias");
    dec.mi_w1 = load_raw_st(sf, "dec_mi_layer.0.weight");
    dec.mi_b1 = load_raw_st(sf, "dec_mi_layer.0.bias");
    dec.mi_w2 = load_raw_st(sf, "dec_mi_layer.2.weight");
    dec.mi_b2 = load_raw_st(sf, "dec_mi_layer.2.bias");
    for (int l = 0; l < 4; l++) {
        char name[128];
        snprintf(name, sizeof(name), "dec_mi_layer.1.lstm.weight_ih_l%d", l);
        dec.mi_lstm_w_ih[l] = load_raw_st(sf, name);
        snprintf(name, sizeof(name), "dec_mi_layer.1.lstm.weight_hh_l%d", l);
        dec.mi_lstm_w_hh[l] = load_raw_st(sf, name);
        snprintf(name, sizeof(name), "dec_mi_layer.1.lstm.bias_ih_l%d", l);
        dec.mi_lstm_b_ih[l] = load_raw_st(sf, name);
        snprintf(name, sizeof(name), "dec_mi_layer.1.lstm.bias_hh_l%d", l);
        dec.mi_lstm_b_hh[l] = load_raw_st(sf, name);
    }
    sf.close();

    int loaded = 0;
    auto inc_if = [&](const BigVGANTensor & t) { if (t.data.size() > 0) loaded++; };
    inc_if(dec.conv_pre_w);
    inc_if(dec.conv_pre_b);
    inc_if(dec.conv_post_w);
    for (int i = 0; i < 6; i++) { inc_if(dec.ups_w[i]); inc_if(dec.ups_b[i]); }
    for (int i = 0; i < 18; i++) {
        for (int j = 0; j < 3; j++) {
            inc_if(dec.rb_conv1_w[i][j]);
            inc_if(dec.rb_conv1_b[i][j]);
            inc_if(dec.rb_conv2_w[i][j]);
            inc_if(dec.rb_conv2_b[i][j]);
        }
        for (int a = 0; a < 6; a++) {
            inc_if(dec.rb_alpha[i][a]);
            inc_if(dec.rb_beta[i][a]);
        }
    }
    inc_if(dec.act_post_alpha);
    inc_if(dec.act_post_beta);
    inc_if(dec.act_post_filter_up);
    inc_if(dec.act_post_filter_down);
    inc_if(dec.post_proj_w);
    inc_if(dec.post_proj_b);
    inc_if(dec.mi_w1);
    inc_if(dec.mi_b1);
    inc_if(dec.mi_w2);
    inc_if(dec.mi_b2);
    for (int l = 0; l < 4; l++) {
        inc_if(dec.mi_lstm_w_ih[l]);
        inc_if(dec.mi_lstm_w_hh[l]);
        inc_if(dec.mi_lstm_b_ih[l]);
        inc_if(dec.mi_lstm_b_hh[l]);
    }
    printf("  BigVGAN C++ v2: %d tensors loaded\n", loaded);
    return loaded > 10;
}

void bigvgan_free(BigVGANDecoder & dec) {
    dec.buf1.clear(); dec.buf2.clear(); dec.buf3.clear();
}

// ============ Full decode ============
bool bigvgan_decode(BigVGANDecoder & dec, const float * latent, int n_frames,
                     float * audio_out, int * n_samples) {
    int hop = VAE_HOP_SAMPLES;
    *n_samples = n_frames * hop;

    int max_ch = 1536;
    int max_len = n_frames * 1920 + 1024; // room for upsampling
    dec.buf1.resize(max_len * max_ch);
    dec.buf2.resize(max_len * max_ch);
    dec.buf3.resize(max_len * max_ch); // extra temp for activation
    // For activation temp (up to 3x original during upsample intermediates)
    dec.buf_act_up.resize(max_len * max_ch * 2);
    dec.buf_act_down.resize(max_len * max_ch * 2);
    dec.buf_amp_save.resize(max_len * max_ch);

    float * x = dec.buf1.data(), * tmp = dec.buf2.data(), * tmp2 = dec.buf3.data();
    float * amp_save = dec.buf_amp_save.data();
    float * act_up = dec.buf_act_up.data(), * act_down = dec.buf_act_down.data();

    // === Bottleneck: post_proj (1x1 conv) → permute → dec_mi_layer → permute ===
    // post_proj: Conv1d(128, 128, 1), causal irrelevant (K=1)
    conv1d_causal(tmp, latent, 128, n_frames, dec.post_proj_w.ptr(), dec.post_proj_b.ptr(), 128, 1);

    // dec_mi_layer: Linear(128→512) → LSTM(4×512, skip) → Linear(512→128)
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
    { float rms=0; for(int i=0;i<n_frames*128;i++) rms+=tmp[i]*tmp[i];
      printf("  bottleneck: rms=%.4f\n", sqrtf(rms/(n_frames*128))); }

    int conv_pre_ch = 1536;

    // === conv_pre: NON-CAUSAL Conv1d(128, 1536, 5), pad=2 both sides ===
    // Python hardcodes conv_pre as causal=False (num_decoder_lookahead based)
    {
        int ic = 128, oc = conv_pre_ch, K = 5, pad = 2;
        int padded_len = n_frames + 2 * pad;
        float * padded = (float*)malloc(ic * padded_len * sizeof(float));
        memset(padded, 0, ic * pad * sizeof(float)); // left zero-pad
        memcpy(padded + ic * pad, tmp, ic * n_frames * sizeof(float));
        memset(padded + ic * (pad + n_frames), 0, ic * pad * sizeof(float)); // right zero-pad
        for (int o = 0; o < oc; o++) {
            float b = dec.conv_pre_b.ptr() ? dec.conv_pre_b.ptr()[o] : 0;
            for (int t = 0; t < n_frames; t++) {
                float s = b;
                for (int k = 0; k < K; k++) {
                    int pt = t + k; // standard conv: sliding window
                    if (pt >= 0 && pt < padded_len) {
                        for (int c = 0; c < ic; c++)
                            s += padded[pt * ic + c] * dec.conv_pre_w.ptr()[(o * ic + c) * K + k];
                    }
                }
                x[t * oc + o] = s;
            }
        }
        free(padded);
    }
    { float rms=0; for(int i=0;i<n_frames*conv_pre_ch;i++) rms+=x[i]*x[i];
      printf("  conv_pre: rms=%.4f\n", sqrtf(rms/(n_frames*conv_pre_ch))); }
    int cur_ch = conv_pre_ch, cur_len = n_frames;

    // === 6 decoder stages ===
    int rb_dilations[3] = {1, 3, 5}; // for convs1 (dilated)
    int rb_kernel_size[3];  // kernel size for convs1 and convs2

    for (int s = 0; s < 6; s++) {
        // --- Upsample: causal ConvTranspose1d ---
        int stride = dec.ups_stride[s], K = dec.ups_kernel[s];
        int out_ch = dec.ups_out[s];
        int up_len;
        convT1d_causal(tmp, x, cur_ch, cur_len,
                       dec.ups_w[s].ptr(), dec.ups_b[s].ptr(),
                       out_ch, stride, K, &up_len);
        cur_ch = out_ch; cur_len = up_len;
        { float rms=0; for(int i=0;i<cur_len*cur_ch;i++) rms+=tmp[i]*tmp[i];
          printf("  stage %d ups: rms=%.4f len=%d ch=%d\n", s, sqrtf(rms/(cur_len*cur_ch)), cur_len, cur_ch); }

        // --- 3 AMP blocks per stage ---
        // Buffer layout:
        //   tmp (buf2)       = upsampled input (read-only during AMP)
        //   residual (buf3) = evolving residual for each resblock
        //   act_up (buf_act_up) = conv output scratch
        //   act_down (buf_act_down) = aa_snakebeta internal scratch
        memset(x, 0, cur_len * cur_ch * sizeof(float));
        for (int j = 0; j < 3; j++) {
            int rb = s * 3 + j;
            int kernel = (j == 0) ? 3 : (j == 1) ? 7 : 11; // resblock_kernel_sizes: [3,7,11]
            // Start with copy of input
            float * residual = tmp2;
            memcpy(residual, tmp, cur_len * cur_ch * sizeof(float));
            for (int pair = 0; pair < 3; pair++) {
                int dil = rb_dilations[pair];
                int act1_idx = pair * 2;
                int act2_idx = pair * 2 + 1;

                // Save residual to act_down (preserved throughout pair)
                memcpy(act_down, residual, cur_len * cur_ch * sizeof(float));

                // a1: aa_snakebeta on act_up (work copy)
                memcpy(act_up, residual, cur_len * cur_ch * sizeof(float));
                aa_snakebeta(act_up, cur_len, cur_ch,
                             dec.rb_alpha[rb][act1_idx].ptr(),
                             dec.rb_beta[rb][act1_idx].ptr(),
                             true, residual/*buf_up*/, amp_save/*buf_down*/, max_len * 2);

                // c1: dilated conv, act_up -> residual
                conv1d_causal(residual, act_up, cur_ch, cur_len,
                             dec.rb_conv1_w[rb][pair].ptr(),
                             dec.rb_conv1_b[rb][pair].ptr(),
                             cur_ch, kernel, dil);

                // a2: aa_snakebeta on residual
                aa_snakebeta(residual, cur_len, cur_ch,
                             dec.rb_alpha[rb][act2_idx].ptr(),
                             dec.rb_beta[rb][act2_idx].ptr(),
                             true, act_up/*buf_up*/, amp_save/*buf_down*/, max_len * 2);

                // c2: dil=1 conv, residual -> act_up
                conv1d_causal(act_up, residual, cur_ch, cur_len,
                             dec.rb_conv2_w[rb][pair].ptr(),
                             dec.rb_conv2_b[rb][pair].ptr(),
                             cur_ch, kernel, 1);

                // Residual add: residual = saved + pair_output
                for (int i = 0; i < cur_len * cur_ch; i++)
                    residual[i] = act_down[i] + act_up[i];
            }


            // Add this resblock's output to x
            for (int i = 0; i < cur_len * cur_ch; i++)
                x[i] += residual[i];
        }
        // Average over 3 resblocks
        for (int i = 0; i < cur_len * cur_ch; i++) x[i] /= 3.0f;
        { float rms=0; for(int i=0;i<cur_len*cur_ch;i++) rms+=x[i]*x[i];
          printf("  stage %d amp: rms=%.4f\n", s, sqrtf(rms/(cur_len*cur_ch))); }
    }

    // === activation_post ===
    aa_snakebeta(x, cur_len, cur_ch,
                 dec.act_post_alpha.ptr(), dec.act_post_beta.ptr(),
                 true, act_up, act_down, max_len * 2,
                 dec.act_post_filter_up.ptr(),
                 dec.act_post_filter_down.ptr(),
                 cur_ch);
    { float rms=0; for(int i=0;i<cur_len*cur_ch;i++) rms+=x[i]*x[i];
      printf("  activation_post: rms=%.4f\n", sqrtf(rms/(cur_len*cur_ch))); }

    // === conv_post: Conv1d(ch, 1, 7), causal, NO bias ===
    conv1d_causal(tmp, x, cur_ch, cur_len,
                  dec.conv_post_w.ptr(), nullptr, // NO bias
                  1, 7);
    int final_len = cur_len;

    // Clamp to [-1, 1] (no tanh), then apply output gain
    float * fb = tmp;
    // Match Python BigVGAN output level (C++ conv/AMP has ~3.4x lower gain)
    const float output_gain = 2.8f;
    for (int i = 0; i < final_len; i++) {
        fb[i] *= output_gain;
        if (fb[i] > 1.0f) fb[i] = 1.0f;
        if (fb[i] < -1.0f) fb[i] = -1.0f;
    }
    { float rms=0, mn=fb[0], mx=fb[0];
      for(int i=0;i<final_len;i++){rms+=fb[i]*fb[i];if(fb[i]<mn)mn=fb[i];if(fb[i]>mx)mx=fb[i];}
      printf("  final: rms=%.4f min=%.4f max=%.4f len=%d\n",
             sqrtf(rms/final_len), mn, mx, final_len); }

    // Copy to output (truncate/pad to expected length)
    int cp = final_len < *n_samples ? final_len : *n_samples;
    memcpy(audio_out, fb, cp * sizeof(float));
    if (cp < *n_samples) memset(audio_out + cp, 0, (*n_samples - cp) * sizeof(float));

    return true;
}
