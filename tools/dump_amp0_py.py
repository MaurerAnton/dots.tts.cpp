#!/usr/bin/env python3.12
"""Dump Python BigVGAN first AMP block input/output for comparison with C++."""
import sys, torch, numpy as np, math
sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime

rt = DotsTtsRuntime.from_pretrained('/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35', precision='float32')
dec = rt.model.vocoder.decoder

lat = np.fromfile('/home/bym/dots.tts.cpp/build/latents.bin', dtype=np.float32).reshape(1,128,-1)
x = torch.from_numpy(lat.copy())
x = rt.model.vocoder.post_proj(x)
x = x.permute(0,2,1)
x = rt.model.vocoder.dec_mi_layer(x)
x = x.permute(0,2,1)

# Hook all intermediate stages of first AMP block
stage0_input = {}
def hook_inp(m, inp, out):
    stage0_input['conv_pre'] = inp[0].detach().cpu().numpy()

dec.conv_pre.register_forward_hook(hook_inp)

# Hook stage 0 upsample
stage0_ups = {}
def hook_ups(m, inp, out):
    stage0_ups['ups'] = out.detach().cpu().numpy()

dec.ups[0][0].register_forward_hook(hook_ups)

# Hook first AMP block
stage0_amp0 = {}
def hook_amp0(m, inp, out):
    stage0_amp0['amp0'] = out.detach().cpu().numpy()

dec.resblocks[0].register_forward_hook(hook_amp0)

with torch.no_grad():
    _ = dec(x)

for name, arr in {**stage0_input, **stage0_ups, **stage0_amp0}.items():
    rms = math.sqrt((arr.astype(float)**2).mean())
    print(f'Py {name}: shape={arr.shape}, RMS={rms:.4f}')
    arr.ravel().astype(np.float32).tofile(f'/tmp/py_bv_{name}.bin')
print('Dumped to /tmp/py_bv_*.bin')
