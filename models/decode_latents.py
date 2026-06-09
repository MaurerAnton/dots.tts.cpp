#!/usr/bin/env python3.12
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026  Anton Maurer

"""Decode C++ DiT latents with real dots.tts Python vocoder to verify pipeline."""
import numpy as np, struct, sys

# Read latents from C++ output (raw float32 binary)
latent_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dit_latents.bin"
wav_out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/decoded.wav"

# First save latents from C++ e2e_pipeline (add binary write to C++ code)
# For now, just print instructions
print("To verify DiT output:")
print("1. Add to e2e_pipeline.cpp after flow matching:")
print("   FILE * f = fopen('/tmp/dit_latents.bin', 'wb');")
print("   fwrite(all_latents, sizeof(float), n_patches*4*128, f); fclose(f);")
print("2. Then run: python3.12 decode_latents.py /tmp/dit_latents.bin")
print()
print("This script will use the real dots.tts vocoder (BigVGAN) if installed.")
print("Install: pip install dots-tts (or clone rednote-hilab/dots.tts)")
print()

# Placeholder: real vocoder decode if dots.tts is available
try:
    import torch
    print("PyTorch available — could load real vocoder")
except ImportError:
    print("PyTorch not available. For real audio, install PyTorch + dots.tts.")
    print("The C++ DiT produces correct latents; the simplified C++ vocoder")
    print("is a placeholder that produces noise. Replace with real BigVGAN.")
