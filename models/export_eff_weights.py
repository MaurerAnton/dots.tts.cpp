#!/usr/bin/env python3.12
"""Export effective BigVGAN weights (after weight_norm) for C++ calibration."""
import sys, numpy as np, torch, json, os
sys.path.insert(0, '/tmp/dots_tts_py/src')
from dots_tts.modules.vocoder.bigvgan import AudioVAE
from dots_tts.models.dots_tts.config import ModelConfig

model_dir = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
with open(f'{model_dir}/config.json') as f:
    cfg = ModelConfig(**json.load(f))
v = AudioVAE(cfg.vocoder).eval()
from safetensors.torch import load_file
v.load_state_dict(load_file(f'{model_dir}/vocoder.safetensors'), strict=False)

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'debug/eff_weights'
os.makedirs(out_dir, exist_ok=True)

# For each module, get the effective weight (after weight_norm)
count = 0
for name, module in v.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        w = module.weight.data.cpu().numpy()
        # Save as raw float32
        safe_name = name.replace('.', '_').replace('/', '_')
        fname = f'{out_dir}/{safe_name}.bin'
        w.astype(np.float32).tofile(fname)
        count += 1
        if count <= 5:
            print(f'{name}: shape={w.shape} RMS={np.sqrt(np.mean(w**2)):.6f} -> {fname}')
    if hasattr(module, 'bias') and module.bias is not None and not isinstance(module.bias, bool):
        b = module.bias.data.cpu().numpy()
        safe_name = name.replace('.', '_').replace('/', '_')
        fname = f'{out_dir}/{safe_name}_bias.bin'
        b.astype(np.float32).tofile(fname)

print(f'\nExported {count} weight tensors to {out_dir}/')
