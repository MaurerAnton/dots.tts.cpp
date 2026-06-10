#!/usr/bin/env python3.12
"""Compare BigVGAN stage outputs: C++ vs Python for same latents."""
import sys, torch, numpy as np, math
sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime

MODEL_DIR = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
dec = rt.model.vocoder.decoder

# Load C++ latents
latents = np.fromfile('/home/bym/dots.tts.cpp/build/latents.bin', dtype=np.float32).reshape(1, 128, -1)
print(f'Latents: {latents.shape}, RMS={math.sqrt((latents**2).mean()):.4f}')

# Python bottleneck
x = torch.from_numpy(latents.copy())
x = rt.model.vocoder.post_proj(x)
print(f'post_proj: RMS={x.norm().item()/math.sqrt(x.numel()):.4f}')
x = x.permute(0, 2, 1)
x = rt.model.vocoder.dec_mi_layer(x)
x = x.permute(0, 2, 1)
print(f'bottleneck: RMS={x.norm().item()/math.sqrt(x.numel()):.4f}')

# Hook each stage
stage_data = {}
def hook_conv_pre(m,i,o):
    stage_data['conv_pre'] = o.detach().cpu().numpy()
    print(f'Py conv_pre: RMS={o.norm().item()/math.sqrt(o.numel()):.4f}')

# Hook upsample + each AMP
for stage_idx in range(6):
    def make_ups_hook(s):
        def h(m,i,o):
            stage_data[f'stage{s}_ups'] = o.detach().cpu().numpy()
            print(f'Py stage{s} ups: RMS={o.norm().item()/math.sqrt(o.numel()):.4f}')
        return h
    # Hook the upsample conv
    if hasattr(dec.ups[stage_idx], '__getitem__'):
        dec.ups[stage_idx][0].register_forward_hook(make_ups_hook(stage_idx))
    
    # Hook AMP blocks (they're in resblocks, 3 per stage)
    for j in range(3):
        def make_amp_hook(s, jj):
            def h(m,i,o):
                stage_data[f'stage{s}_amp{jj}'] = o.detach().cpu().numpy()
                rms = o.norm().item()/math.sqrt(o.numel())
                print(f'Py stage{s} amp{jj}: RMS={rms:.4f}')
            return h
        dec.resblocks[stage_idx*3 + j].register_forward_hook(make_amp_hook(stage_idx, j))

def hook_act_post(m,i,o):
    stage_data['act_post'] = o.detach().cpu().numpy()
    print(f'Py act_post: RMS={o.norm().item()/math.sqrt(o.numel()):.4f}')

def hook_conv_post(m,i,o):
    stage_data['conv_post'] = o.detach().cpu().numpy()
    print(f'Py conv_post: RMS={o.norm().item()/math.sqrt(o.numel()):.4f}')

dec.conv_pre.register_forward_hook(hook_conv_pre)
dec.activation_post.register_forward_hook(hook_act_post)
dec.conv_post.register_forward_hook(hook_conv_post)

# Run decoder
with torch.no_grad():
    out = dec(x)
    out_tanh = torch.tanh(out)
    print(f'Py post_tanh: RMS={out_tanh.norm().item()/math.sqrt(out_tanh.numel()):.4f}')

# Spectral correlation at each stage
import os
for name in sorted(stage_data.keys()):
    arr = stage_data[name]
    arr.ravel().astype(np.float32).tofile(f'/tmp/py_bv_{name}.bin')
    # Compute spectral centroid
    if arr.ndim >= 2 and arr.shape[-1] > 4:
        spec = np.abs(np.fft.rfft(arr.ravel()[:min(4096, arr.size)]))
        centroid = np.sum(np.fft.rfftfreq(min(4096, arr.size))*spec)/np.sum(spec)
        print(f'  {name}: centroid={centroid*48000:.0f} Hz')
