// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

#include "campp.h"
#include "safetensors.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>

// Weight structures now in campp.h

// ============ Helper: 2D convolution
static void conv2d(float * out, const float * in, int H, int W, int ic,
                   const Conv2dW & cw, int stride_h, int stride_w, int pad_h, int pad_w) {
    int out_h = (H + 2*pad_h - cw.kh) / stride_h + 1;
    int out_w = (W + 2*pad_w - cw.kw) / stride_w + 1;
    int oc = cw.oc;
    memset(out, 0, oc * out_h * out_w * sizeof(float));
    // Pad input
    int pad_H = H + 2*pad_h, pad_W = W + 2*pad_w;
    float * padded = new float[ic * pad_H * pad_W]();
    for (int c = 0; c < ic; c++)
        for (int h = 0; h < H; h++)
            memcpy(padded + c*pad_H*pad_W + (h+pad_h)*pad_W + pad_w,
                   in + c*H*W + h*W, W * sizeof(float));
    for (int o = 0; o < oc; o++) {
        float b = cw.has_bias ? cw.bias[o] : 0;
        for (int oh = 0; oh < out_h; oh++)
            for (int ow = 0; ow < out_w; ow++) {
                float s = b;
                for (int c = 0; c < ic; c++)
                    for (int kh = 0; kh < cw.kh; kh++)
                        for (int kw = 0; kw < cw.kw; kw++)
                            s += padded[c*pad_H*pad_W + (oh*stride_h+kh)*pad_W + (ow*stride_w+kw)]
                               * cw.w[(o*ic + c)*cw.kh*cw.kw + kh*cw.kw + kw];
                out[(o*out_h + oh)*out_w + ow] = s;
            }
    }
    delete[] padded;
}

// ============ BatchNorm2d (eval mode) ============
static void batchnorm2d(float * x, int c, int H, int W, const BatchNorm2d & bn) {
    float eps = 1e-5f;
    for (int ch = 0; ch < c; ch++) {
        float gamma = bn.w.size() ? bn.w[ch] : 1.0f;
        float beta = bn.b.size() ? bn.b[ch] : 0;
        float inv_std = 1.0f / sqrtf(bn.rv[ch] + eps);
        for (int i = 0; i < H*W; i++)
            x[ch*H*W + i] = gamma * (x[ch*H*W + i] - bn.rm[ch]) * inv_std + beta;
    }
}

// ============ ReLU in-place ============
static void relu(float * x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0;
}

// ============ Conv1d ============
static void conv1d(float * out, const float * in, int T, int ic,
                   const Conv1dW & cw, int stride, int pad, int dilation) {
    int K = cw.k, oc = cw.oc;
    int out_T = (T + 2*pad - dilation*(K-1) - 1) / stride + 1;
    int pad_T = T + 2*pad;
    float * padded = new float[ic * pad_T]();
    for (int c = 0; c < ic; c++)
        memcpy(padded + c*pad_T + pad, in + c*T, T * sizeof(float));
    for (int o = 0; o < oc; o++) {
        float b = cw.has_bias ? cw.bias[o] : 0;
        for (int t = 0; t < out_T; t++) {
            float s = b;
            int base = t * stride;
            for (int c = 0; c < ic; c++)
                for (int k = 0; k < K; k++) {
                    int idx = base + k * dilation;
                    s += padded[c*pad_T + idx] * cw.w[(o*ic + c)*K + k];
                }
            out[o*out_T + t] = s;
        }
    }
    delete[] padded;
}

// ============ BatchNorm1d (eval mode) ============
static void batchnorm1d(float * x, int c, int T, const BatchNorm1d & bn) {
    if (!bn.affine && bn.w.empty()) {
        // affine=False: just normalize
        float eps = 1e-5f;
        for (int ch = 0; ch < c; ch++) {
            float inv_std = 1.0f / sqrtf(bn.rv[ch] + eps);
            for (int t = 0; t < T; t++)
                x[ch*T + t] = (x[ch*T + t] - bn.rm[ch]) * inv_std;
        }
    } else {
        float eps = 1e-5f;
        for (int ch = 0; ch < c; ch++) {
            float gamma = bn.w.empty() ? 1.0f : bn.w[ch];
            float beta = bn.b.empty() ? 0 : bn.b[ch];
            float inv_std = 1.0f / sqrtf(bn.rv[ch] + eps);
            for (int t = 0; t < T; t++)
                x[ch*T + t] = gamma * (x[ch*T + t] - bn.rm[ch]) * inv_std + beta;
        }
    }
}

// ============ FCM head forward ============
static void fcm_forward(float * out, int * out_T, const float * fbank, int T_fbank,
                        const CAMWeights & w) {
    int F = CAMPP_FBANK_N_MELS; // 80
    int m_channels = 32;
    
    // x shape: (1, T, F) -> unsqueeze(1) -> (1, 1, F, T) -> permute to (1, 1, T, F)
    // Actually: input is (T, F), we work in (C, H, W) = (1, T, F)
    int H = T_fbank, W = F;
    int n = H * W;
    float * x = new float[n];
    memcpy(x, fbank, n * sizeof(float));
    
    // conv1: Conv2d(1→32, K=3, pad=1, stride=1)
    float * tmp = new float[32 * H * W];
    conv2d(tmp, x, H, W, 1, w.head_conv1, 1, 1, 1, 1);
    batchnorm2d(tmp, 32, H, W, w.head_bn1);
    relu(tmp, 32*H*W);
    delete[] x;
    
    // layer1: 2 BasicResBlocks, stride=2
    H = (H + 1) / 2; // stride 2
    for (int b = 0; b < 2; b++) {
        const BasicResBlock & rb = w.head_layer1[b];
        float * residual = new float[32 * H * W];
        memcpy(residual, tmp, 32 * H * W * sizeof(float));
        
        conv2d(tmp, tmp, H, W, 32, rb.conv1, 1, 1, 1, 1);
        batchnorm2d(tmp, 32, H, W, rb.bn1);
        relu(tmp, 32*H*W);
        conv2d(tmp, tmp, H, W, 32, rb.conv2, 1, 1, 1, 1);
        batchnorm2d(tmp, 32, H, W, rb.bn2);
        
        for (int i = 0; i < 32*H*W; i++) tmp[i] += residual[i];
        relu(tmp, 32*H*W);
        delete[] residual;
    }
    
    // layer2: 2 BasicResBlocks, stride=2
    H = (H + 1) / 2;
    for (int b = 0; b < 2; b++) {
        const BasicResBlock & rb = w.head_layer2[b];
        float * residual = new float[32 * H * W];
        memcpy(residual, tmp, 32 * H * W * sizeof(float));
        
        conv2d(tmp, tmp, H, W, 32, rb.conv1, 1, 1, 1, 1);
        batchnorm2d(tmp, 32, H, W, rb.bn1);
        relu(tmp, 32*H*W);
        conv2d(tmp, tmp, H, W, 32, rb.conv2, 1, 1, 1, 1);
        batchnorm2d(tmp, 32, H, W, rb.bn2);
        
        for (int i = 0; i < 32*H*W; i++) tmp[i] += residual[i];
        relu(tmp, 32*H*W);
        delete[] residual;
    }
    
    // conv2: Conv2d(32→32, K=3, stride=(2,1), pad=1)
    int new_H = (H + 2 - 3) / 2 + 1; // stride_h=2
    int new_W = (W + 2 - 3) / 1 + 1; // stride_w=1, pad_w=1
    new_W = W; // with pad=1, stride=1: same dim
    float * tmp2 = new float[32 * new_H * new_W];
    conv2d(tmp2, tmp, H, W, 32, w.head_conv2, 2, 1, 1, 1);
    batchnorm2d(tmp2, 32, new_H, new_W, w.head_bn2);
    relu(tmp2, 32*new_H*new_W);
    delete[] tmp;
    
    // Flatten: (C, H, W) -> (C*H, W) = (channels, time)
    int channels = 32 * new_H;
    *out_T = new_W;
    for (int c = 0; c < 32; c++)
        for (int h = 0; h < new_H; h++)
            memcpy(out + (c*new_H + h) * new_W, tmp2 + c*new_H*new_W + h*new_W, new_W*sizeof(float));
    delete[] tmp2;
}

// ============ CAM layer forward ============
static void cam_layer_forward(float * out, const float * in, int T, int channels,
                              const CAMDenseTDNNLayer & layer) {
    // nonlinear2 (bn+relu) applied before CAM in DenseTDNNLayer
    // Actually: in CAMDenseTDNNLayer.forward:
    //   x = bn_function(x)  // nonlinear1 + linear1 (1x1 conv)
    //   return cam_layer(nonlinear2(x))
    // We receive x after bn_function, apply nonlinear2 then CAM
    
    // First: nonlinear2 (BatchNorm1d + ReLU)
    float * xn = new float[channels * T];
    memcpy(xn, in, channels * T * sizeof(float));
    batchnorm1d(xn, channels, T, layer.nonlinear2);
    relu(xn, channels * T);
    
    // CAM: 
    //   y = linear_local(x)  (dilated conv)
    //   context = mean(x) + seg_pooling(x)
    //   context = relu(linear1(context))
    //   m = sigmoid(linear2(context))
    //   return y * m
    
    int cam_out = layer.cam_linear_local.oc;
    float * y = new float[cam_out * T];
    conv1d(y, xn, T, channels, layer.cam_linear_local, 1, (layer.cam_linear_local.k-1)/2, 1);
    
    // Context: global mean + segment pooling
    float * context = new float[channels];
    for (int c = 0; c < channels; c++) {
        float s = 0;
        for (int t = 0; t < T; t++) s += xn[c*T + t];
        context[c] = s / T;
    }
    
    // Segment pooling (avg, seg_len=100)
    int seg_len = 100;
    if (T < seg_len) seg_len = T;
    float * seg = new float[channels * T];
    for (int c = 0; c < channels; c++) {
        for (int pos = 0; pos < T; pos += seg_len) {
            int end = pos + seg_len < T ? pos + seg_len : T;
            float avg = 0;
            for (int t = pos; t < end; t++) avg += xn[c*T + t];
            avg /= (end - pos);
            for (int t = pos; t < end && t < T; t++)
                seg[c*T + t] = avg;
        }
    }
    
    // context = mean + seg (broadcast mean)
    for (int c = 0; c < channels; c++) {
        float m = context[c];
        for (int t = 0; t < T; t++)
            seg[c*T + t] += m;
    }
    delete[] context;
    
    // context shape: (channels, 1) -> unsqueeze
    float * ctx_1d = new float[channels];
    for (int c = 0; c < channels; c++) ctx_1d[c] = seg[c*T + T/2]; // take middle time step
    
    // linear1: 1x1 conv on context
    int ctx_mid = layer.cam_linear1.oc;
    float * ctx_mid_data = new float[ctx_mid];
    for (int o = 0; o < ctx_mid; o++) {
        float s = layer.cam_linear1.has_bias ? layer.cam_linear1.bias[o] : 0;
        for (int i = 0; i < channels; i++)
            s += ctx_1d[i] * layer.cam_linear1.w[o*channels + i];
        ctx_mid_data[o] = s;
    }
    delete[] ctx_1d;
    
    // ReLU
    for (int i = 0; i < ctx_mid; i++) if (ctx_mid_data[i] < 0) ctx_mid_data[i] = 0;
    
    // linear2: 1x1 conv
    float * mask = new float[cam_out];
    for (int o = 0; o < cam_out; o++) {
        float s = layer.cam_linear2.has_bias ? layer.cam_linear2.bias[o] : 0;
        for (int i = 0; i < ctx_mid; i++)
            s += ctx_mid_data[i] * layer.cam_linear2.w[o*ctx_mid + i];
        // Sigmoid
        mask[o] = 1.0f / (1.0f + expf(-s));
    }
    delete[] ctx_mid_data;
    
    // y * mask (broadcast mask over time)
    for (int o = 0; o < cam_out; o++) {
        float m = mask[o];
        for (int t = 0; t < T; t++)
            out[o*T + t] = y[o*T + t] * m;
    }
    delete[] mask; delete[] y; delete[] xn; delete[] seg;
}

// ============ CAMDenseTDNNLayer forward ============
// Returns: output appended to input (dense connection)
static void cam_dense_tdnn_forward(float * out, const float * in, int T, int in_channels,
                                    const CAMDenseTDNNLayer & layer, int growth_rate) {
    // bn_function: nonlinear1 + linear1
    float * x = new float[in_channels * T];
    memcpy(x, in, in_channels * T * sizeof(float));
    batchnorm1d(x, in_channels, T, layer.nonlinear1);
    relu(x, in_channels * T);
    
    int bn_ch = layer.linear1.oc;
    float * bn_out = new float[bn_ch * T];
    for (int t = 0; t < T; t++)
        for (int o = 0; o < bn_ch; o++) {
            float s = 0;
            for (int i = 0; i < in_channels; i++)
                s += x[i*T + t] * layer.linear1.w[o*in_channels + i];
            bn_out[o*T + t] = s;
        }
    delete[] x;
    
    // Then nonlinear2 + CAM — but wait, the CAM layer does its own nonlinear2 first.
    // Actually looking at Python: cam_dense_tdnn_forward receives the bn_out, then
    // CAMDenseTDNNLayer.forward applies nonlinear2 internally before CAM.
    // The CAM layer receives bn_out and applies nonlinear2 then CAM.
    
    // Apply cam_layer (which internally applies nonlinear2)
    cam_layer_forward(out, bn_out, T, bn_ch, layer);
    delete[] bn_out;
}

// ============ Simple DFT (real input → complex output, power spectrum) ============
static void dft_magnitudes(float * mag, const float * frame, int frame_len, int n_fft) {
    // Simple O(N²) DFT — fine for ~100 frames of speaker embedding
    float scale = 1.0f / sqrtf((float)n_fft);
    for (int k = 0; k <= n_fft/2; k++) {
        float re = 0, im = 0;
        for (int n = 0; n < frame_len && n < n_fft; n++) {
            float angle = -2.0f * M_PI * k * n / n_fft;
            re += frame[n] * cosf(angle);
            im += frame[n] * sinf(angle);
        }
        mag[k] = sqrtf(re*re + im*im) * scale;
    }
}

// ============ Mel filterbank ============
static float hz_to_mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

static void create_mel_filterbank(std::vector<float> & mel_basis, int n_mels, int n_fft, int sample_rate,
                                   float f_min, float f_max) {
    int n_freqs = n_fft / 2 + 1;
    mel_basis.assign(n_mels * n_freqs, 0);
    
    float mel_min = hz_to_mel(f_min), mel_max = hz_to_mel(f_max);
    float mel_step = (mel_max - mel_min) / (n_mels + 1);
    
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++)
        mel_points[i] = mel_to_hz(mel_min + i * mel_step);
    
    std::vector<float> freq_centers(n_freqs);
    for (int i = 0; i < n_freqs; i++)
        freq_centers[i] = (float)i * sample_rate / n_fft;
    
    for (int m = 0; m < n_mels; m++) {
        float f_left = mel_points[m], f_center = mel_points[m+1], f_right = mel_points[m+2];
        for (int k = 0; k < n_freqs; k++) {
            float f = freq_centers[k];
            if (f >= f_left && f <= f_center)
                mel_basis[m * n_freqs + k] = (f - f_left) / (f_center - f_left);
            else if (f > f_center && f <= f_right)
                mel_basis[m * n_freqs + k] = (f_right - f) / (f_right - f_center);
        }
    }
}

// ============ FBank extraction (STFT → mel → log) ============
static int extract_fbank(float * fbank_out, const float * audio, int n_samples,
                          int sample_rate, int n_mels, int n_fft,
                          float frame_len_ms, float frame_shift_ms) {
    int frame_len = (int)(frame_len_ms * sample_rate / 1000);
    int frame_shift = (int)(frame_shift_ms * sample_rate / 1000);
    int n_frames = (n_samples - frame_len) / frame_shift + 1;
    if (n_frames < 5) return 0;
    
    int n_freqs = n_fft / 2 + 1;
    
    // Hamming window
    std::vector<float> window(frame_len);
    for (int i = 0; i < frame_len; i++)
        window[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (frame_len - 1));
    
    // Create mel filterbank (cached)
    static std::vector<float> mel_basis_cache;
    static int mel_basis_n_mels = 0, mel_basis_n_fft = 0;
    if (mel_basis_cache.empty() || mel_basis_n_mels != n_mels || mel_basis_n_fft != n_fft) {
        create_mel_filterbank(mel_basis_cache, n_mels, n_fft, sample_rate, 20.0f, 7600.0f);
        mel_basis_n_mels = n_mels; mel_basis_n_fft = n_fft;
    }
    
    std::vector<float> frame(frame_len), mag(n_freqs), mel_energies(n_mels);
    
    for (int f = 0; f < n_frames; f++) {
        int start = f * frame_shift;
        // Apply window
        for (int i = 0; i < frame_len; i++)
            frame[i] = audio[start + i] * window[i];
        
        dft_magnitudes(mag.data(), frame.data(), frame_len, n_fft);
        
        // Apply mel filterbank
        for (int m = 0; m < n_mels; m++) {
            float energy = 0;
            for (int k = 0; k < n_freqs; k++)
                energy += mag[k] * mag[k] * mel_basis_cache[m * n_freqs + k];
            mel_energies[m] = energy;
        }
        
        // Log (with floor)
        for (int m = 0; m < n_mels; m++) {
            float v = mel_energies[m];
            if (v < 1e-10f) v = 1e-10f;
            fbank_out[f * n_mels + m] = logf(v);
        }
    }
    return n_frames;
}

// ============ Full CAM++ forward ============
bool campp_extract(const CAMPPWeights & wp, const float * audio, int n_samples, float * embedding) {
    const CAMWeights & w = wp.w;
    int n_mels = CAMPP_FBANK_N_MELS;
    int n_fft = 512;
    float frame_len_ms = CAMPP_FRAME_LEN_MS;
    float frame_shift_ms = CAMPP_FRAME_SHIFT_MS;
    
    // Step 1: Extract FBank features
    std::vector<float> fbank(n_samples * n_mels); // overallocate
    int T = extract_fbank(fbank.data(), audio, n_samples, CAMPP_SAMPLE_RATE,
                          n_mels, n_fft, frame_len_ms, frame_shift_ms);
    if (T < 10) {
        memset(embedding, 0, CAMPP_EMBEDDING_SIZE * sizeof(float));
        return false;
    }
    // DEBUG
    { float rms=0; for(int i=0;i<T*n_mels;i++) rms+=fbank[i]*fbank[i];
      printf("  FBank: T=%d rms=%.4f nan=%d\n", T, sqrtf(rms/(T*n_mels)),
             (int)std::isnan(sqrtf(rms))); }
    
    // Step 2: FCM head (conv2d feature extraction)
    // FCM expects input as (F, T): frequency=80, time=n_frames
    // After head: channels=160 (32*5), time=n_frames
    int head_T;
    int head_channels;
    {
        int F = n_mels;  // frequency bins = 80
        int W_time = T;  // time dimension
        int m_channels = 32;
        
        // Transpose input from (T,F) to (F,T) for FCM
        float * x = new float[F * W_time];
        for (int t = 0; t < W_time; t++)
            for (int f = 0; f < F; f++)
                x[f * W_time + t] = fbank[t * F + f];
        
        int H_freq = F;
        float * tmp = new float[m_channels * H_freq * W_time];
        
        // conv1: Conv2d(1→32, K=3, pad=1, stride=1)
        conv2d(tmp, x, H_freq, W_time, 1, w.head_conv1, 1, 1, 1, 1);
        batchnorm2d(tmp, m_channels, H_freq, W_time, w.head_bn1);
        relu(tmp, m_channels * H_freq * W_time);
        delete[] x;
        
        // layer1: 2 BasicResBlocks, first with stride=2, second stride=1
        H_freq = (H_freq + 2 - 3) / 2 + 1; // stride=2 on first block
        for (int b = 0; b < 2; b++) {
            int stride = (b == 0) ? 2 : 1;
            int blk_H = (b == 0) ? H_freq : H_freq; // H_freq already adjusted
            const BasicResBlock & rb = w.head_layer1[b];
            float * residual = new float[m_channels * blk_H * W_time];
            memcpy(residual, tmp, m_channels * blk_H * W_time * sizeof(float));
            
            conv2d(tmp, tmp, blk_H, W_time, m_channels, rb.conv1, 1, 1, 1, 1);
            batchnorm2d(tmp, m_channels, blk_H, W_time, rb.bn1);
            relu(tmp, m_channels * blk_H * W_time);
            conv2d(tmp, tmp, blk_H, W_time, m_channels, rb.conv2, 1, 1, 1, 1);
            batchnorm2d(tmp, m_channels, blk_H, W_time, rb.bn2);
            for (int i = 0; i < m_channels * blk_H * W_time; i++) tmp[i] += residual[i];
            relu(tmp, m_channels * blk_H * W_time);
            delete[] residual;
        }
        
        // layer2: 2 BasicResBlocks, first stride=2, second stride=1
        H_freq = (H_freq + 2 - 3) / 2 + 1; // stride=2 on first block of layer2
        for (int b = 0; b < 2; b++) {
            int blk_H = H_freq; // same H for both blocks in layer2 (second has stride=1)
            if (b == 1) blk_H = H_freq; // actually stride=1 keeps same H
            const BasicResBlock & rb = w.head_layer2[b];
            float * residual = new float[m_channels * blk_H * W_time];
            memcpy(residual, tmp, m_channels * blk_H * W_time * sizeof(float));
            
            conv2d(tmp, tmp, blk_H, W_time, m_channels, rb.conv1, 1, 1, 1, 1);
            batchnorm2d(tmp, m_channels, blk_H, W_time, rb.bn1);
            relu(tmp, m_channels * blk_H * W_time);
            conv2d(tmp, tmp, blk_H, W_time, m_channels, rb.conv2, 1, 1, 1, 1);
            batchnorm2d(tmp, m_channels, blk_H, W_time, rb.bn2);
            for (int i = 0; i < m_channels * blk_H * W_time; i++) tmp[i] += residual[i];
            relu(tmp, m_channels * blk_H * W_time);
            delete[] residual;
        }
        
        // conv2: Conv2d(32→32, stride=(2,1), pad=1)
        // stride_h=2 on frequency, stride_w=1 on time
        int new_H = (H_freq + 2 - 3) / 2 + 1;
        float * tmp2 = new float[m_channels * new_H * W_time];
        conv2d(tmp2, tmp, H_freq, W_time, m_channels, w.head_conv2, 2, 1, 1, 1);
        batchnorm2d(tmp2, m_channels, new_H, W_time, w.head_bn2);
        relu(tmp2, m_channels * new_H * W_time);
        delete[] tmp;
        
        head_channels = m_channels * new_H;
        head_T = W_time;
        { float rms=0; int n=head_channels*head_T; for(int i=0;i<n;i++){float v=tmp2[i% (m_channels*new_H*W_time)]; rms+=v*v;}
          (void)rms; } // just to avoid unused warning, real debug below
        { float rms=0; int n=m_channels*new_H*W_time; for(int i=0;i<n;i++) rms+=tmp2[i]*tmp2[i];
          printf("  FCM head output: ch=%d T=%d rms=%.4f\\n", head_channels, head_T, sqrtf(rms/n)); }
        
        // Flatten: (C, H, W) → (C*H, W)
        float * feats = new float[head_channels * head_T];
        for (int c = 0; c < m_channels; c++)
            for (int h = 0; h < new_H; h++)
                memcpy(feats + (c * new_H + h) * head_T,
                       tmp2 + c * new_H * W_time + h * W_time,
                       head_T * sizeof(float));
        delete[] tmp2;
        
        // Step 3: TDNN backbone
        // Reallocate larger buffer for dense connections
        {
            int max_ch = head_channels + 12*32 + 24*32 + 16*32;
            float * big_feats = new float[max_ch * head_T];
            memcpy(big_feats, feats, head_channels * head_T * sizeof(float));
            delete[] feats;
            feats = big_feats;
        }
        
        int cur_ch = head_channels;
        int cur_T = head_T;
        
        // === TDNN: Conv1d(cur_ch→128, K=5, stride=2, dil=1, pad=2) ===
        {
            int out_ch = 128;
            float * tdnn_out = new float[out_ch * head_T]; // overallocate
            conv1d(tdnn_out, feats, cur_T, cur_ch, w.tdnn, 2, 2, 1);
            int out_T = (cur_T + 4 - 4) / 2 + 1; // stride=2, pad=2, K=5, dil=1
            batchnorm1d(tdnn_out, out_ch, out_T, w.tdnn_bn);
            relu(tdnn_out, out_ch * out_T);
            memcpy(feats, tdnn_out, out_ch * out_T * sizeof(float));
            delete[] tdnn_out;
            cur_ch = out_ch; cur_T = out_T;
        }
        
        // === Block1: 12 CAMDenseTDNN layers, growth=32, dil=1, K=3 ===
        {
            int growth = 32;
            for (int i = 0; i < w.BLOCK1_LAYERS; i++) {
                float * cam_out = new float[growth * cur_T];
                cam_dense_tdnn_forward(cam_out, feats, cur_T, cur_ch, w.block1[i], growth);
                // Concatenate: feats = [feats; cam_out]
                memcpy(feats + cur_ch * cur_T, cam_out, growth * cur_T * sizeof(float));
                delete[] cam_out;
                cur_ch += growth;
            }
        }
        
        // Transit1: Conv1d(cur_ch→cur_ch/2, K=1)
        {
            int out_ch = cur_ch / 2;
            float * t_out = new float[out_ch * cur_T];
            conv1d(t_out, feats, cur_T, cur_ch, w.transit1, 1, 0, 1);
            batchnorm1d(t_out, out_ch, cur_T, w.transit1_bn);
            relu(t_out, out_ch * cur_T);
            memcpy(feats, t_out, out_ch * cur_T * sizeof(float));
            delete[] t_out;
            cur_ch = out_ch;
        }
        
        // === Block2: 24 layers, growth=32, dil=2, K=3 ===
        {
            int growth = 32;
            for (int i = 0; i < w.BLOCK2_LAYERS; i++) {
                float * cam_out = new float[growth * cur_T];
                cam_dense_tdnn_forward(cam_out, feats, cur_T, cur_ch, w.block2[i], growth);
                memcpy(feats + cur_ch * cur_T, cam_out, growth * cur_T * sizeof(float));
                delete[] cam_out;
                cur_ch += growth;
            }
        }
        
        // Transit2
        {
            int out_ch = cur_ch / 2;
            float * t_out = new float[out_ch * cur_T];
            conv1d(t_out, feats, cur_T, cur_ch, w.transit2, 1, 0, 1);
            batchnorm1d(t_out, out_ch, cur_T, w.transit2_bn);
            relu(t_out, out_ch * cur_T);
            memcpy(feats, t_out, out_ch * cur_T * sizeof(float));
            delete[] t_out;
            cur_ch = out_ch;
        }
        
        // === Block3: 16 layers, growth=32, dil=2, K=3 ===
        {
            int growth = 32;
            for (int i = 0; i < w.BLOCK3_LAYERS; i++) {
                float * cam_out = new float[growth * cur_T];
                cam_dense_tdnn_forward(cam_out, feats, cur_T, cur_ch, w.block3[i], growth);
                memcpy(feats + cur_ch * cur_T, cam_out, growth * cur_T * sizeof(float));
                delete[] cam_out;
                cur_ch += growth;
            }
        }
        
        // Transit3
        {
            int out_ch = cur_ch / 2;
            float * t_out = new float[out_ch * cur_T];
            conv1d(t_out, feats, cur_T, cur_ch, w.transit3, 1, 0, 1);
            batchnorm1d(t_out, out_ch, cur_T, w.transit3_bn);
            relu(t_out, out_ch * cur_T);
            memcpy(feats, t_out, out_ch * cur_T * sizeof(float));
            delete[] t_out;
            cur_ch = out_ch;
        }
        
        // out_nonlinear: BatchNorm1d
        batchnorm1d(feats, cur_ch, cur_T, w.out_nonlinear);
        
        // StatsPool: mean + std → 2*cur_ch
        float * stats = new float[cur_ch * 2];
        for (int c = 0; c < cur_ch; c++) {
            float sum = 0, sum_sq = 0;
            for (int t = 0; t < cur_T; t++) {
                float v = feats[c * cur_T + t];
                sum += v; sum_sq += v * v;
            }
            float mean = sum / cur_T;
            float var = sum_sq / cur_T - mean * mean;
            if (var < 1e-4f) var = 1e-4f;
            stats[c] = mean;
            stats[cur_ch + c] = sqrtf(var);
        }
        
        // Dense: Linear(2*cur_ch→512) → BN(affine=False)
        int dense_in = cur_ch * 2;
        float * dense_out = new float[CAMPP_EMBEDDING_SIZE];
        for (int o = 0; o < CAMPP_EMBEDDING_SIZE; o++) {
            float s = 0;
            for (int i = 0; i < dense_in; i++)
                s += stats[i] * w.dense.w[o * dense_in + i];
            dense_out[o] = s;
        }
        
        // BN affine=False: just normalize
        {
            float eps = 1e-5f;
            for (int c = 0; c < CAMPP_EMBEDDING_SIZE; c++) {
                float inv_std = 1.0f / sqrtf(w.dense_bn.rv[c] + eps);
                embedding[c] = (dense_out[c] - w.dense_bn.rm[c]) * inv_std;
            }
        }
        
        delete[] stats; delete[] dense_out; delete[] feats;
    }
    
    return true;
}

// ============ Weight loading from safetensors ============
static void load_conv2d(SafeTensorsFile & sf, const std::string & prefix, Conv2dW & cw, bool bias) {
    auto w = sf.find((prefix + ".weight").c_str());
    if (!w) return;
    cw.oc = w->shape[0]; cw.ic = w->shape[1]; cw.kh = w->shape[2]; cw.kw = w->shape[3];
    cw.w.resize(cw.oc * cw.ic * cw.kh * cw.kw);
    sf.load_raw(*w, cw.w.data(), cw.w.size());
    cw.has_bias = bias;
    if (bias) {
        auto b = sf.find((prefix + ".bias").c_str());
        if (b) { cw.bias.resize(cw.oc); sf.load_raw(*b, cw.bias.data(), cw.oc); cw.has_bias = true; }
        else cw.has_bias = false;
    }
}

static void load_conv1d(SafeTensorsFile & sf, const std::string & prefix, Conv1dW & cw) {
    auto w = sf.find((prefix + ".weight").c_str());
    if (!w) return;
    cw.oc = w->shape[0]; cw.ic = w->shape[1]; cw.k = w->shape[2];
    cw.w.resize(cw.oc * cw.ic * cw.k);
    sf.load_raw(*w, cw.w.data(), cw.w.size());
    auto b = sf.find((prefix + ".bias").c_str());
    if (b) { cw.bias.resize(cw.oc); sf.load_raw(*b, cw.bias.data(), cw.oc); cw.has_bias = true; }
    else cw.has_bias = false;
}

static void load_bn2d(SafeTensorsFile & sf, const std::string & prefix, BatchNorm2d & bn) {
    auto w = sf.find((prefix + ".weight").c_str());
    if (!w) return;
    bn.c = w->shape[0];
    bn.w.resize(bn.c); sf.load_raw(*w, bn.w.data(), bn.c);
    bn.b.resize(bn.c);
    auto bb = sf.find((prefix + ".bias").c_str());
    if (bb) sf.load_raw(*bb, bn.b.data(), bn.c);
    bn.rm.resize(bn.c); bn.rv.resize(bn.c);
    sf.load_raw(*sf.find((prefix + ".running_mean").c_str()), bn.rm.data(), bn.c);
    sf.load_raw(*sf.find((prefix + ".running_var").c_str()), bn.rv.data(), bn.c);
}

static void load_bn1d(SafeTensorsFile & sf, const std::string & prefix, BatchNorm1d & bn, bool affine=true) {
    bn.affine = affine;
    auto w = sf.find((prefix + ".weight").c_str());
    if (!w && !affine) {
        // affine=False: no weight/bias, still need running stats
        auto rm = sf.find((prefix + ".running_mean").c_str());
        if (!rm) return;
        bn.c = rm->shape[0];
        bn.rm.resize(bn.c); sf.load_raw(*rm, bn.rm.data(), bn.c);
        bn.rv.resize(bn.c);
        sf.load_raw(*sf.find((prefix + ".running_var").c_str()), bn.rv.data(), bn.c);
        return;
    }
    if (!w) return;
    bn.c = w->shape[0];
    if (affine) {
        bn.w.resize(bn.c); sf.load_raw(*w, bn.w.data(), bn.c);
        auto bb = sf.find((prefix + ".bias").c_str());
        if (bb) { bn.b.resize(bn.c); sf.load_raw(*bb, bn.b.data(), bn.c); }
    }
    bn.rm.resize(bn.c); bn.rv.resize(bn.c);
    sf.load_raw(*sf.find((prefix + ".running_mean").c_str()), bn.rm.data(), bn.c);
    sf.load_raw(*sf.find((prefix + ".running_var").c_str()), bn.rv.data(), bn.c);
}

static void load_bn1d_nonaffine(SafeTensorsFile & sf, const std::string & prefix, BatchNorm1d & bn) {
    load_bn1d(sf, prefix, bn, false);
}

static void load_resblock(SafeTensorsFile & sf, const std::string & prefix, BasicResBlock & rb) {
    load_conv2d(sf, prefix + ".conv1", rb.conv1, false);
    load_conv2d(sf, prefix + ".conv2", rb.conv2, false);
    load_bn2d(sf, prefix + ".bn1", rb.bn1);
    load_bn2d(sf, prefix + ".bn2", rb.bn2);
    // shortcut: Conv2d 1x1 + BatchNorm2d
    load_conv2d(sf, prefix + ".shortcut.0", rb.shortcut_conv, false);
    load_bn2d(sf, prefix + ".shortcut.1", rb.shortcut_bn);
}

static void load_cam_tdnn_layer(SafeTensorsFile & sf, const std::string & prefix, CAMDenseTDNNLayer & l) {
    load_bn1d(sf, prefix + ".nonlinear1.batchnorm", l.nonlinear1);
    load_conv1d(sf, prefix + ".linear1", l.linear1);
    load_bn1d(sf, prefix + ".nonlinear2.batchnorm", l.nonlinear2);
    load_conv1d(sf, prefix + ".cam_layer.linear_local", l.cam_linear_local);
    load_conv1d(sf, prefix + ".cam_layer.linear1", l.cam_linear1);
    load_conv1d(sf, prefix + ".cam_layer.linear2", l.cam_linear2);
}

bool campp_load(const char * safetensors_path, CAMPPWeights & wp) {
    SafeTensorsFile sf;
    if (!sf.open(safetensors_path)) return false;
    
    CAMWeights & w = wp.w;
    
    // FCM head
    load_conv2d(sf, "model.head.conv1", w.head_conv1, false);
    load_bn2d(sf, "model.head.bn1", w.head_bn1);
    load_conv2d(sf, "model.head.conv2", w.head_conv2, false);
    load_bn2d(sf, "model.head.bn2", w.head_bn2);
    
    for (int i = 0; i < 2; i++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "model.head.layer1.%d", i);
        load_resblock(sf, prefix, w.head_layer1[i]);
        snprintf(prefix, sizeof(prefix), "model.head.layer2.%d", i);
        load_resblock(sf, prefix, w.head_layer2[i]);
    }
    
    // TDNN
    load_conv1d(sf, "model.xvector.tdnn.linear", w.tdnn);
    load_bn1d(sf, "model.xvector.tdnn.nonlinear.batchnorm", w.tdnn_bn);
    
    // Block 1: 12 layers
    for (int i = 0; i < 12; i++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "model.xvector.block1.tdnnd%d", i+1);
        load_cam_tdnn_layer(sf, prefix, w.block1[i]);
    }
    load_conv1d(sf, "model.xvector.transit1.linear", w.transit1);
    load_bn1d(sf, "model.xvector.transit1.nonlinear.batchnorm", w.transit1_bn);
    
    // Block 2: 24 layers
    for (int i = 0; i < 24; i++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "model.xvector.block2.tdnnd%d", i+1);
        load_cam_tdnn_layer(sf, prefix, w.block2[i]);
    }
    load_conv1d(sf, "model.xvector.transit2.linear", w.transit2);
    load_bn1d(sf, "model.xvector.transit2.nonlinear.batchnorm", w.transit2_bn);
    
    // Block 3: 16 layers
    for (int i = 0; i < 16; i++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "model.xvector.block3.tdnnd%d", i+1);
        load_cam_tdnn_layer(sf, prefix, w.block3[i]);
    }
    load_conv1d(sf, "model.xvector.transit3.linear", w.transit3);
    load_bn1d(sf, "model.xvector.transit3.nonlinear.batchnorm", w.transit3_bn);
    
    // out_nonlinear
    load_bn1d(sf, "model.xvector.out_nonlinear.batchnorm", w.out_nonlinear);
    
    // Dense + BN (affine=False)
    load_conv1d(sf, "model.xvector.dense.linear", w.dense);
    load_bn1d_nonaffine(sf, "model.xvector.dense.nonlinear.batchnorm", w.dense_bn);
    
    // Resample kernel
    auto rk = sf.find("resample.kernel");
    if (rk) sf.load_raw(*rk, wp.resample_kernel, 41);
    
    sf.close();
    printf("  CAM++: loaded speaker encoder weights\n");
    return true;
}

void campp_free(CAMPPWeights & w) {
    // Vectors auto-free
}
