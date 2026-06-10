#!/usr/bin/env python3.12
"""Dump BigVGAN decoder intermediates per stage for C++ comparison."""
import sys, torch, numpy as np, math
sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime

MODEL_DIR = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
torch.manual_seed(42)

# Get latents from C++ file
latents = np.fromfile('/home/bym/dots.tts.cpp/latents.bin', dtype=np.float32).reshape(1, 128, -1)
print(f'Latents: shape={latents.shape}, RMS={math.sqrt((latents**2).mean()):.4f}')

model = rt.model.vocoder
dec = model.decoder

# Hook to capture intermediates
stage_outs = {}
def hook_conv_pre(module, input, output):
    stage_outs['conv_pre'] = output.detach().cpu().numpy()
    
def hook_stage(s_idx, name):
    def _hook(module, input, output):
        stage_outs[f'stage{s_idx}_{name}'] = output.detach().cpu().numpy()
    return _hook

# Register hooks
dec.conv_pre.register_forward_hook(hook_conv_pre)
for s_idx, stage in enumerate(dec.resblocks):
    stage[0].register_forward_hook(hook_stage(s_idx, 'ups'))
    for a_idx, amp in enumerate(stage[1:]):
        amp.register_forward_hook(hook_stage(s_idx, f'amp{a_idx}'))
dec.activation_post.register_forward_hook(lambda m,i,o: stage_outs.__setitem__('act_post', o.detach().cpu().numpy()))
dec.conv_post.register_forward_hook(lambda m,i,o: stage_outs.__setitem__('conv_post', o.detach().cpu().numpy()))

# Run decoder
with torch.no_grad():
    x = torch.from_numpy(latents.copy())
    x = model.post_proj(x)
    x = x.permute(0, 2, 1)
    x = model.dec_mi_layer(x)
    x = x.permute(0, 2, 1)
    _ = dec(x)

for name, arr in sorted(stage_outs.items()):
    rms = math.sqrt((arr.astype(float)**2).mean())
    print(f'  py_{name}: shape={arr.shape}, RMS={rms:.4f}')
    arr.ravel().astype(np.float32).tofile(f'/tmp/py_bv_{name}.bin')
print('Dumps saved to /tmp/py_bv_*.bin')
