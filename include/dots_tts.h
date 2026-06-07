#pragma once

// dots.tts.cpp - First C++ implementation of dots.tts
// Based on: rednote-hilab/dots.tts (Apache 2.0)
// Architecture: 2B params, 24 languages, 48kHz, BPE-LLM backbone

// =============================================================================
// DiT (Diffusion Transformer) - flow-matching head
// =============================================================================
// 18 DiTBlocks, hidden=1024, 16 heads, FFN=4096
// adaLN modulation: timestep + speaker x-vector -> shift/scale/gate per block
// Predicts velocity field v_t for flow-matching ODE

#define DIT_NUM_LAYERS      18
#define DIT_HIDDEN_SIZE     1024
#define DIT_NUM_HEADS       16
#define DIT_HEAD_SIZE       64
#define DIT_FFN_SIZE        4096
#define DIT_ADA_DIM         1024
#define DIT_MAX_SEQ_LEN     2048
#define DIT_ROPE_THETA      10000.0f
#define DIT_SPEAKER_DIM     512

// =============================================================================
// PatchEncoder - 24-layer streaming transformer encoder
// =============================================================================
// Encodes audio VAE patches (4 x 128) into LLM-compatible embeddings (1 x 1536)
// Causal, qk_norm, streaming via decode_step

#define PATCHENC_NUM_LAYERS 24
#define PATCHENC_HIDDEN     1024
#define PATCHENC_NUM_HEADS  16
#define PATCHENC_HEAD_SIZE  64
#define PATCHENC_FFN_SIZE   4096
#define PATCHENC_LATENT_DIM 128
#define PATCHENC_PATCH_SIZE 4
#define PATCHENC_ROPE_THETA 10000.0f

// =============================================================================
// AudioVAE - BigVGAN-style vocoder
// =============================================================================
// Encoder: 7 Conv1d stages -> 128d latent
// Decoder: 6 upsampling stages, 3 AMPBlock1 (SnakeBeta) per stage
// Hop: 960 samples @ 48kHz, SLSTM bottleneck

#define VAE_LATENT_DIM      128
#define VAE_HOP_SAMPLES     960
#define VAE_SAMPLE_RATE     48000
#define VAE_SLSTM_HIDDEN    512
#define VAE_SLSTM_LAYERS    4
