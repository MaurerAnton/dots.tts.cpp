#!/usr/bin/env python3.12
"""Dump Python BigVGAN stage outputs for frequency spectrum comparison."""
import sys, torch, numpy as np, math
sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime
MODEL_DIR = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
torch.manual_seed(42)

result = rt.generate(text='hello world', num_steps=10, guidance_scale=1.0)
audio = result['audio'].numpy().ravel()
rms = math.sqrt((audio.astype(float)**2).mean())
print(f'Py audio: {len(audio)} samp, RMS={rms:.4f}')

# Save raw float audio for spectrum comparison
audio.astype(np.float32).tofile('/tmp/py_audio_float.bin')
print('Saved /tmp/py_audio_float.bin')

# Dump BigVGAN intermediates using hooks
model = rt.model.vocoder
dec = model.decoder

# Monkey-patch to capture stage outputs
outputs = {}
orig_forward = dec.forward
def hooked_forward(x):
    outputs['decoder_input'] = x.detach().cpu().numpy()
    # Run original
    return orig_forward(x)

dec.forward = hooked_forward

# Re-run with hooks
lat_t = torch.from_numpy(result.get('latents', result.get('audio_latents')).numpy().copy())
with torch.no_grad():
    _ = rt.model.vocoder.inference_from_latents(lat_t, do_sample=False)

for name, arr in outputs.items():
    print(f'{name}: shape={arr.shape}, RMS={math.sqrt((arr.astype(float)**2).mean()):.4f}')
    arr.ravel().astype(np.float32).tofile(f'/tmp/py_bv_{name}.bin')
