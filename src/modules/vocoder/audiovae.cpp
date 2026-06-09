// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// dots.tts.cpp - AudioVAE decoder (simplified MVP)
// For pipeline validation: manual upsampling, no learned conv layers

#include "audiovae.h"
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <cstdio>

// ===========================================================================
// Simplified decoder: nearest-neighbor upsampling
// Each latent frame (128-dim) is repeated hop_samples times, then projected
// to mono via learned linear layer.
// ===========================================================================

ggml_tensor * audiovae_decode_simple(
    ggml_context * ctx,
    ggml_tensor * latent,      // [n_frames, latent_dim] in ggml layout: [latent_dim, n_frames]
    int n_frames,
    float * waveform_out,      // pre-allocated [n_frames * VAE_HOP_SAMPLES]
    int * n_samples_out)
{
    int latent_dim = VAE_LATENT_DIM;
    int hop = VAE_HOP_SAMPLES;
    int n_samples = n_frames * hop;
    *n_samples_out = n_samples;

    // latent is [latent_dim, n_frames] in ggml memory
    float * lat_data = tensor_data(latent);

    // Simple approach: for each frame, repeat latent vector hop times
    // Then apply a simple random projection to 1 channel
    // This produces a "scratchy" but non-zero waveform for validation

    // Use deterministic "weights" derived from position
    for (int f = 0; f < n_frames; f++) {
        for (int s = 0; s < hop; s++) {
            float sample = 0.0f;
            // Weighted sum of latent dims with sinusoidal weights
            for (int d = 0; d < latent_dim; d++) {
                float weight = sinf((float)(f * hop + s) * 0.001f * (float)(d + 1)) * 0.05f;
                sample += lat_data[f * latent_dim + d] * weight;
            }
            // Clamp to reasonable range
            if (sample > 1.0f) sample = 1.0f;
            if (sample < -1.0f) sample = -1.0f;
            waveform_out[f * hop + s] = sample;
        }
    }

    // Return a dummy ggml tensor for the waveform (for graph consistency)
    ggml_tensor * wav = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_samples);
    memcpy(tensor_data(wav), waveform_out, n_samples * sizeof(float));

    return wav;
}

// ===========================================================================
// Full decoder (stub — actual conv_transpose not yet implemented in ggml graph)
// ===========================================================================

ggml_tensor * audiovae_decode(
    audiovae_decoder & dec,
    ggml_context * ctx,
    ggml_tensor * latent,
    int n_frames)
{
    // Stub: delegates to simple decoder for now
    int n_samples;
    float * wav_buf = new float[n_frames * VAE_HOP_SAMPLES];
    ggml_tensor * out = audiovae_decode_simple(ctx, latent, n_frames, wav_buf, &n_samples);
    delete[] wav_buf;
    return out;
}
