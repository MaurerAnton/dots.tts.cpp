// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// AudioVAE encoder — causal Conv1d encoder (HoliTok-style)
// Input: 48kHz mono PCM, output: 128-dim latents at 25 Hz
#pragma once
#include <cstring>

struct AudioVAEEncoderWeights {
    // input conv: 1→12, K=3, stride=1
    float *pre_conv_wg, *pre_conv_wv, *pre_conv_bias;
    
    // 7 stages: each has transition conv + ResStack (6 dilated conv pairs)
    static const int NUM_STAGES = 7;
    int stage_channels[7];     // input channels per stage
    int stage_kernel[7];       // transition conv kernel size
    int stage_stride[7];       // transition conv stride
    
    // Transition convs per stage: weight_g, weight_v, bias
    float *stage_conv_wg[7], *stage_conv_wv[7], *stage_conv_bias[7];
    
    // ResStack: 6 pairs of dilated convs per stage
    // Each pair: dilated conv (w1_g, w1_v, b1) + undilated conv (w2_g, w2_v, b2)
    static const int RESSTACK_DEPTH = 6;
    float *rs_w1_g[7][6], *rs_w1_v[7][6], *rs_b1[7][6];
    float *rs_w2_g[7][6], *rs_w2_v[7][6], *rs_b2[7][6];
    
    // Output conv: 768→128, K=3, stride=1
    float *out_conv_wg, *out_conv_wv, *out_conv_bias;
    
    bool loaded;
    AudioVAEEncoderWeights() : loaded(false) {
        memset(this, 0, sizeof(*this));
    }
};

bool audiovae_encoder_load(const char * safetensors_path, AudioVAEEncoderWeights & w);
void audiovae_encoder_free(AudioVAEEncoderWeights & w);

// Encode 48kHz PCM audio → 128-dim latents at 25Hz
// audio: float samples (48kHz mono), n_samples: sample count
// latents_out: output [n_frames * 128], caller allocates
// n_frames_out: number of output frames (25Hz)
bool audiovae_encode(const AudioVAEEncoderWeights & w, const float * audio, int n_samples,
                     float * latents_out, int * n_frames_out);
