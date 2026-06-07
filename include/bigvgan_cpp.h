#pragma once
// dots.tts.cpp - Pure C++ BigVGAN vocoder
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
    int ups_stride[6], ups_kernel[6], ups_in_ch[6], ups_out_ch[6];
    
    BigVGANTensor rb_conv1_w[18], rb_conv1_b[18], rb_conv2_w[18], rb_conv2_b[18];
    BigVGANTensor rb_alpha[18][2], rb_beta[18][2]; // 2 groups of 3 activations each
    int rb_kernel[18][3], rb_dilation[18][3];
    
    BigVGANTensor conv_post_w, conv_post_b;
    BigVGANTensor act_post_alpha, act_post_beta;
    
    // post_proj: Conv1d 128->128, kernel=1
    BigVGANTensor post_proj_w, post_proj_b;
    // dec_mi_layer: Linear -> LSTM(4 layers) -> Linear
    BigVGANTensor mi_w1, mi_b1;  // Linear 128->512
    BigVGANTensor mi_w2, mi_b2;  // Linear 512->128
    BigVGANTensor mi_lstm_w_ih[4], mi_lstm_w_hh[4];
    BigVGANTensor mi_lstm_b_ih[4], mi_lstm_b_hh[4];
    
    std::vector<float> buf1, buf2;
};

bool bigvgan_load(const char * sf_path, BigVGANDecoder & dec);
bool bigvgan_decode(BigVGANDecoder & dec, const float * latent, int n_frames,
                     float * audio_out, int * n_samples);
void bigvgan_free(BigVGANDecoder & dec);
