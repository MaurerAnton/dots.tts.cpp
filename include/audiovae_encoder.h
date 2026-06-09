// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// AudioVAE encoder — causal Conv1d encoder (HoliTok-style)
// Input: 48kHz mono PCM, output: 128-dim latents at 25 Hz
#pragma once
#include <cstring>

struct AudioVAEEncoderWeights {
    // input conv: 1→12, K=3, stride=1
    float *pre_conv_wg, *pre_conv_wv, *pre_conv_bias;
    
    // 7 stages: transition conv + ResStack (6 dilated conv pairs each)
    static const int NUM_STAGES = 7;
    int stage_channels[7], stage_kernel[7], stage_stride[7];
    float *stage_conv_wg[7], *stage_conv_wv[7], *stage_conv_bias[7];
    static const int RESSTACK_DEPTH = 6;
    float *rs_w1_g[7][6], *rs_w1_v[7][6], *rs_b1[7][6];
    float *rs_w2_g[7][6], *rs_w2_v[7][6], *rs_b2[7][6];
    float *out_conv_wg, *out_conv_wv, *out_conv_bias;
    
    // enc_mi_layer: Linear(128→512) → LSTM(4×512, skip) → Linear(512→128)
    float *mi_w1, *mi_b1;
    float *mi_lstm_w_ih[4], *mi_lstm_w_hh[4], *mi_lstm_b_ih[4], *mi_lstm_b_hh[4];
    float *mi_w2, *mi_b2;
    
    // pre_proj: Conv1d(128→256, K=1)
    float *pre_proj_w, *pre_proj_b;
    
    bool loaded;
    AudioVAEEncoderWeights() : loaded(false) { memset(this, 0, sizeof(*this)); }
};

bool audiovae_encoder_load(const char * safetensors_path, AudioVAEEncoderWeights & w);
void audiovae_encoder_free(AudioVAEEncoderWeights & w);

// Encode 48kHz PCM audio → 256-dim latents at 25Hz (mean + logvar)
// audio: float samples (48kHz mono), n_samples: sample count
// latents_out: output [n_frames * 128], caller allocates (mean only)
// n_frames_out: number of output frames (25Hz)
bool audiovae_encode(const AudioVAEEncoderWeights & w, const float * audio, int n_samples,
                     float * latents_out, int * n_frames_out);
