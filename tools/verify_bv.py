#!/usr/bin/env python3.12
"""Verify Python BigVGAN decoder from C++ intermediate values."""
import sys, torch, numpy as np, math
sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime

MODEL_DIR = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
dec = rt.model.vocoder.decoder

# Load C++ intermediates: post_proj output, conv_pre output
# These should be dumped by C++ BigVGAN — let's use the printed RMS values to verify
# Actually, we need the raw data. Let's modify C++ to dump them.

# For now, let's verify: feed C++ latents to Python bottleneck, then use C++ conv_pre output
# to see if conv_pre matches

latents = np.fromfile('/home/bym/dots.tts.cpp/latents.bin', dtype=np.float32).reshape(1, 128, -1)
print(f'C++ latents: shape={latents.shape}, RMS={math.sqrt((latents**2).mean()):.4f}')

# Python bottleneck (post_proj + LSTM)
x = torch.from_numpy(latents.copy())
x = rt.model.vocoder.post_proj(x)
print(f'Py post_proj: RMS={x.norm().item()/math.sqrt(x.numel()):.4f}')
x = x.permute(0, 2, 1)
x = rt.model.vocoder.dec_mi_layer(x)
x = x.permute(0, 2, 1)
print(f'Py bottleneck: RMS={x.norm().item()/math.sqrt(x.numel()):.4f}')

# Python conv_pre
x = dec.conv_pre(x)
print(f'Py conv_pre: RMS={x.norm().item()/math.sqrt(x.numel()):.4f}')

# Now run full decoder
x = dec(x)
audio = x.squeeze().numpy()
rms = math.sqrt((audio.astype(float)**2).mean())
print(f'Py decoder audio RMS: {rms:.4f}')

# Compare with C++ output
import wave
with wave.open('/home/bym/dots.tts.cpp/build/output_llm_full.wav', 'rb') as wf:
    cpp_audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(float)/32768
print(f'C++ decoder audio RMS: {math.sqrt((cpp_audio**2).mean()):.4f}')
print(f'C++ decoder audio max: {cpp_audio.max():.4f}')

# Spectral comparison
n = min(len(audio), len(cpp_audio))
py_spec = np.abs(np.fft.rfft(audio[:n]))
cpp_spec = np.abs(np.fft.rfft(cpp_audio[:n]))
print(f'Spectral corr: {np.corrcoef(py_spec, cpp_spec)[0,1]:.4f}')
