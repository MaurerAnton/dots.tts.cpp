#!/usr/bin/env python3.12
"""Export effective BigVGAN weights (weight_norm applied) for C++ calibration."""
import sys, numpy as np, torch
sys.path.insert(0, '/tmp/dots_tts_py/src')
from dots_tts.modules.vocoder.bigvgan import AudioVAE
from dots_tts.models.dots_tts.config import ModelConfig
import json
from safetensors.torch import save_file, load_file

model_dir = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
with open(f'{model_dir}/config.json') as f: cfg = ModelConfig(**json.load(f))
model = AudioVAE(cfg.vocoder).eval()
model.load_state_dict(load_file(f'{model_dir}/vocoder.safetensors'), strict=False)

out_path = sys.argv[1] if len(sys.argv) > 1 else 'models/vocoder_eff.safetensors'

sd = model.state_dict()
weight_g_keys = {k for k in sd if k.endswith('.weight_g')}

eff_state = {}

# Process weight_norm modules: compute effective weight
for gk in sorted(weight_g_keys):
    base = gk[:-len('.weight_g')]
    vk = base + '.weight_v'
    bk = base + '.bias'
    
    w_g = sd[gk].cpu().numpy()
    w_v = sd[vk].cpu().numpy()
    
    # norm along all dims except dim 0
    axes = tuple(range(1, w_v.ndim))
    norm_v = np.sqrt(np.sum(w_v**2, axis=axes, keepdims=True))
    w_g_r = w_g.reshape(norm_v.shape)
    eff_w = w_v * (w_g_r / (norm_v + 1e-12))
    
    eff_state[base + '.weight'] = torch.from_numpy(eff_w.astype(np.float32))
    if bk in sd:
        eff_state[bk] = sd[bk].cpu()

# Copy non-weight_norm parameters
for k, val in sd.items():
    if k in eff_state: continue
    if k.endswith('.weight_g') or k.endswith('.weight_v'): continue
    eff_state[k] = val.cpu()

# Copy buffers
for k, buf in model.named_buffers():
    if k not in eff_state:
        eff_state[k] = buf.cpu()

save_file(eff_state, out_path)
print(f'Exported {len(eff_state)} tensors to {out_path}')

# Verify
eff_w = eff_state['decoder.ups.0.0.weight'].cpu().numpy()
pt_eff = model.decoder.ups[0][0].weight.data.cpu().numpy()
print(f'Export ups.0.0 RMS: {np.sqrt(np.mean(eff_w**2)):.6f}')
print(f'PyTorch ups.0.0 RMS: {np.sqrt(np.mean(pt_eff**2)):.6f}')
print(f'Match: {np.allclose(eff_w, pt_eff)}')
