#!/usr/bin/env python3.12
"""Export deterministic encoder weights (with weight_norm cache fix)."""
import sys, torch, numpy as np, json
sys.path.insert(0, '/tmp/dots_tts_py/src')
from dots_tts.modules.vocoder.bigvgan import AudioVAE
from dots_tts.models.dots_tts.config import ModelConfig
from safetensors.torch import save_file, load_file

model_dir = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
with open(f'{model_dir}/config.json') as f:
    cfg = ModelConfig(**json.load(f))
v = AudioVAE(cfg.vocoder).eval()
v.load_state_dict(load_file(f'{model_dir}/vocoder.safetensors'), strict=False)

# CRITICAL: run dummy forward to update weight_norm caches
dummy = torch.zeros(1, 1, 1920)
with torch.no_grad():
    _ = v.audio_encoder(dummy)
print('Cache updated via forward pass')

# Export
state = {}
enc = v.audio_encoder
for i, layer in enumerate(enc.generator):
    prefix = f'audio_encoder.generator.{i}'
    if hasattr(layer, 'layer'):
        mod = layer.layer
        if hasattr(mod, 'weight') and mod.weight is not None:
            state[prefix + '.layer.weight'] = mod.weight.data.cpu()
        if hasattr(mod, 'bias') and mod.bias is not None:
            state[prefix + '.layer.bias'] = mod.bias.data.cpu()
    if hasattr(layer, 'layers'):
        for j, reslayer in enumerate(layer.layers):
            for k, submod in enumerate(reslayer):
                if hasattr(submod, 'weight') and submod.weight is not None:
                    state[f'{prefix}.layers.{j}.{k}.weight'] = submod.weight.data.cpu()
                if hasattr(submod, 'bias') and submod.bias is not None:
                    state[f'{prefix}.layers.{j}.{k}.bias'] = submod.bias.data.cpu()

out_path = sys.argv[1] if len(sys.argv) > 1 else 'models/encoder_det.safetensors'
save_file(state, out_path)
w0 = state['audio_encoder.generator.0.layer.weight']
print(f'Exported {len(state)} tensors to {out_path}')
print(f'pre-conv ch0: {w0[0,0,:].numpy()}')
