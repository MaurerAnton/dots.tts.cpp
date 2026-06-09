#!/usr/bin/env python3.12
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026  Anton Maurer
"""Export effective BigVGAN weights (after weight_norm) to safetensors for C++ calibration.

PyTorch applies weight_norm during load_state_dict, transforming the stored weights.
C++ loads raw weights from safetensors without weight_norm, causing ~30% RMS divergence.
This script exports the EFFECTIVE weights (what PyTorch actually uses) as a new safetensors file.

Usage: python3.12 export_eff_vocoder.py [output.safetensors]
"""
import sys, os, numpy as np, torch, json
sys.path.insert(0, '/tmp/dots_tts_py/src')
from dots_tts.modules.vocoder.bigvgan import AudioVAE
from dots_tts.models.dots_tts.config import ModelConfig
from safetensors.torch import save_file

model_dir = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
with open(f'{model_dir}/config.json') as f:
    cfg = ModelConfig(**json.load(f))
v = AudioVAE(cfg.vocoder).eval()

from safetensors.torch import load_file
v.load_state_dict(load_file(f'{model_dir}/vocoder.safetensors'), strict=False)

out_path = sys.argv[1] if len(sys.argv) > 1 else 'models/vocoder_eff.safetensors'

# Collect effective weights AND all parameters (alpha, beta, etc.)
state = {}
count = 0
for name, module in v.named_modules():
    if name == '': continue  # skip root
    # Weight
    if hasattr(module, 'weight') and module.weight is not None and not isinstance(module.weight, bool):
        state[name + '.weight'] = module.weight.data.cpu()
        count += 1
    # Bias
    if hasattr(module, 'bias') and module.bias is not None and not isinstance(module.bias, bool):
        state[name + '.bias'] = module.bias.data.cpu()
    # SnakeBeta/activation parameters
    if hasattr(module, 'alpha') and module.alpha is not None:
        state[name + '.alpha'] = module.alpha.data.cpu()
        count += 1
    if hasattr(module, 'beta') and module.beta is not None:
        state[name + '.beta'] = module.beta.data.cpu()
        count += 1

# Also include ALL named parameters (catches LSTM weights, etc.)
for pname, param in v.named_parameters():
    if pname not in state:
        state[pname] = param.data.cpu()
        count += 1

# And buffers (running_mean, running_var, etc.)
for bname, buf in v.named_buffers():
    if bname not in state:
        state[bname] = buf.data.cpu()
        count += 1

save_file(state, out_path)
print(f'Exported {count} effective tensors to {out_path}')

# Verify: compare a key weight
# Original raw weight RMS
import struct
with open(f'{model_dir}/vocoder.safetensors', 'rb') as f:
    hlen = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(hlen))
    ds = 8 + hlen
    info = h['decoder.conv_pre.weight']
    f.seek(ds + info['data_offsets'][0])
    raw = np.frombuffer(f.read(info['data_offsets'][1]-info['data_offsets'][0]), dtype=np.float32)
print(f'Original conv_pre.weight RMS: {np.sqrt(np.mean(raw**2)):.6f}')

# Effective weight RMS (from module)
eff = v.decoder.conv_pre.weight.data.cpu().numpy()
print(f'Effective conv_pre.weight RMS: {np.sqrt(np.mean(eff**2)):.6f}')
print(f'Ratio: {np.sqrt(np.mean(eff**2))/np.sqrt(np.mean(raw**2)):.3f}')
