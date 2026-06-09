// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

#pragma once
// dots.tts.cpp - Pure C++ BigVGAN vocoder v2 (causal, snake_logscale, anti-alias)
#include "safetensors.h"
#include <vector>

struct BigVGANTensor {
    std::vector<float> data;
    int n0, n1, n2;
    float * ptr() { return data.data(); }
};

struct BigVGANDecoder {
    BigVGANTensor conv_pre_w, conv_pre_b;
    BigVGANTensor ups_w[6], ups_b[6];
    int ups_stride[6], ups_kernel[6], ups_in[6], ups_out[6];
    
    // 18 resblocks, each with 3 conv pairs and 6 activations
    BigVGANTensor rb_conv1_w[18][3], rb_conv1_b[18][3];
    BigVGANTensor rb_conv2_w[18][3], rb_conv2_b[18][3];
    BigVGANTensor rb_alpha[18][6], rb_beta[18][6]; // 6 activations per resblock
    
    BigVGANTensor conv_post_w; // NO bias (use_bias_at_final=False)
    BigVGANTensor act_post_alpha, act_post_beta;
    BigVGANTensor act_post_filter_up, act_post_filter_down; // per-channel [ch,1,12]
    
    // post_proj + dec_mi_layer (bottleneck)
    BigVGANTensor post_proj_w, post_proj_b;
    BigVGANTensor mi_w1, mi_b1;  // Linear 128->512
    BigVGANTensor mi_w2, mi_b2;  // Linear 512->128
    BigVGANTensor mi_lstm_w_ih[4], mi_lstm_w_hh[4];
    BigVGANTensor mi_lstm_b_ih[4], mi_lstm_b_hh[4];
    
    // Working buffers
    std::vector<float> buf1, buf2, buf3;
    std::vector<float> buf_act_up, buf_act_down;
    std::vector<float> buf_amp_save; // residual save during AMP pairs
};

bool bigvgan_load(const char * sf_path, BigVGANDecoder & dec);
bool bigvgan_decode(BigVGANDecoder & dec, const float * latent, int n_frames,
                     float * audio_out, int * n_samples);
void bigvgan_free(BigVGANDecoder & dec);
