// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

#pragma once

// dots.tts.cpp - AudioVAE decoder (BigVGAN-style vocoder, simplified)
// Decodes VAE latent patches [n_frames, latent_dim] -> 48kHz waveform [n_samples]
//
// Simplified for MVP: Linear projection + ConvTranspose1d upsampling.
// No alias-free resampling, no SnakeBeta — uses SiLU instead.
// Full quality requires 6 AMPBlock stages; this is a functional stub.

#include "dots_tts.h"
#include "ggml.h"

struct audiovae_decoder {
    // Input projection: [latent_dim -> hidden]
    ggml_tensor * in_proj_w;   // [latent_dim, hidden]
    ggml_tensor * in_proj_b;   // [hidden]

    // Upsampling stages: total factor = HOP_SAMPLES = 960
    // Divided into stages: 10, 6, 4, 2, 2 = 960
    // Each stage: ConvTranspose1d(kernel=stage*2, stride=stage)
    int n_stages;
    int * stage_factors;        // [n_stages] e.g. {10, 6, 4, 2, 2}
    ggml_tensor ** stage_weight; // [n_stages] each [in_ch, out_ch, kernel]
    ggml_tensor ** stage_bias;   // [n_stages] each [out_ch]

    // Final conv: [hidden -> 1]
    ggml_tensor * out_conv_w;  // [hidden, 1, kernel=7]
    ggml_tensor * out_conv_b;  // [1]

    // Channel dimensions
    int latent_dim;
    int hidden_dim;             // intermediate channel count
};

// Decode latent frames to waveform
// latent: [n_frames, latent_dim] — flattened VAE latent
// returns: [n_samples] — 48kHz mono float waveform (-1..1)
// n_samples = n_frames * VAE_HOP_SAMPLES
ggml_tensor * audiovae_decode(
    audiovae_decoder & dec,
    ggml_context * ctx,
    ggml_tensor * latent,
    int n_frames);

// Simplified MVP decoder: nearest-neighbor upsample, no learned params
// Always produces some output for pipeline validation
ggml_tensor * audiovae_decode_simple(
    ggml_context * ctx,
    ggml_tensor * latent,      // [n_frames, latent_dim]
    int n_frames,
    float * waveform_out,      // pre-allocated [n_frames * VAE_HOP_SAMPLES]
    int * n_samples_out);
