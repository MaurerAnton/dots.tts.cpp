#!/usr/bin/env python3.12
"""Export BigVGAN weights AFTER remove_weight_norm (effective weights)."""
import sys, numpy as np, torch
sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime
from safetensors.torch import save_file

model_dir = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
out_path = sys.argv[1] if len(sys.argv) > 1 else 'models/vocoder_eff.safetensors'

# DotsTtsRuntime.from_pretrained calls remove_weight_norm internally
rt = DotsTtsRuntime.from_pretrained(model_dir, precision='float32')
model = rt.model.vocoder

# Get state dict WITHOUT weight_g/weight_v (those are stale after remove_weight_norm)
sd = model.state_dict()
eff = {}
for k, v in sd.items():
    if k.endswith('.weight_g') or k.endswith('.weight_v'):
        continue
    eff[k] = v.cpu()

save_file(eff, out_path)
print(f'Exported {len(eff)} tensors to {out_path}')

# Verify
pt_w = model.decoder.ups[0][0].weight
print(f'Exported ups.0.0.weight RMS: {pt_w.norm().item() / pt_w.numel()**0.5:.6f}')
