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
    BigVGANTensor rb_alpha[18], rb_beta[18];
    int rb_kernel1[18], rb_dilation1[18], rb_kernel2[18], rb_dilation2[18], rb_channels[18];
    
    BigVGANTensor conv_post_w, conv_post_b;
    std::vector<float> buf1, buf2;
};

bool bigvgan_load(const char * sf_path, BigVGANDecoder & dec);
bool bigvgan_decode(BigVGANDecoder & dec, const float * latent, int n_frames,
                     float * audio_out, int * n_samples);
void bigvgan_free(BigVGANDecoder & dec);
