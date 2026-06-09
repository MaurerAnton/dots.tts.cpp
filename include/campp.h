// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// CAM++ speaker encoder — pure C++ inference (based on 3D-Speaker / Alibaba DAMO Academy)
// Input: raw PCM audio (16kHz mono), output: 512-dim speaker embedding (x-vector)
// Architecture: FCM head → TDNN → 3× CAMDenseTDNN blocks → StatsPool → Dense projection
#pragma once
#include <vector>
#include <string>

#define CAMPP_SAMPLE_RATE  16000
#define CAMPP_FBANK_N_MELS 80
#define CAMPP_EMBEDDING_SIZE 512
#define CAMPP_FRAME_LEN_MS 25
#define CAMPP_FRAME_SHIFT_MS 10

// ============ Internal weight structures ============

struct Conv2dW { int oc, ic, kh, kw; std::vector<float> w, bias; bool has_bias; };
struct Conv1dW { int oc, ic, k; std::vector<float> w, bias; bool has_bias; };
struct BatchNorm2d { int c; std::vector<float> w, b, rm, rv; };
struct BatchNorm1d { int c; std::vector<float> w, b, rm, rv; bool affine; };

struct BasicResBlock {
    Conv2dW conv1, conv2;
    BatchNorm2d bn1, bn2;
    Conv2dW shortcut_conv;
    BatchNorm2d shortcut_bn;
};

struct CAMDenseTDNNLayer {
    BatchNorm1d nonlinear1, nonlinear2;
    Conv1dW linear1;
    Conv1dW cam_linear_local, cam_linear1, cam_linear2;
};

struct CAMWeights {
    Conv2dW head_conv1, head_conv2;
    BatchNorm2d head_bn1, head_bn2;
    BasicResBlock head_layer1[2], head_layer2[2];
    
    Conv1dW tdnn;
    BatchNorm1d tdnn_bn;
    
    static const int BLOCK1_LAYERS = 12, BLOCK2_LAYERS = 24, BLOCK3_LAYERS = 16;
    CAMDenseTDNNLayer block1[12], block2[24], block3[16];
    Conv1dW transit1, transit2, transit3;
    BatchNorm1d transit1_bn, transit2_bn, transit3_bn;
    BatchNorm1d out_nonlinear;
    Conv1dW dense;
    BatchNorm1d dense_bn;
};

struct CAMPPWeights {
    CAMWeights w;
    float resample_kernel[41];
};

// Load weights from speaker_encoder.safetensors
bool campp_load(const char * safetensors_path, CAMPPWeights & wp);
void campp_free(CAMPPWeights & w);

// Extract speaker embedding from raw PCM float audio (16kHz mono)
// audio: float samples, n_samples: number of samples
// embedding: output 512-dim float array (caller allocates)
bool campp_extract(const CAMPPWeights & wp, const float * audio, int n_samples, float * embedding);
