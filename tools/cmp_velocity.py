#!/usr/bin/env python3.12
"""Compare C++ DiT velocity field output vs Python on identical input.
Debug mode - step-by-step to find issues.
"""
import sys, os, math
import torch
import torch.nn as nn
import numpy as np

MODEL_DIR = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
DEBUG_DIR = '/home/bym/dots.tts.cpp/debug'

sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.modules.backbone.dit import DiTBlock

# Monkey-patch DiTBlock to debug
_orig_forward = DiTBlock.forward
def debug_forward(self, x, c, mask=None, **kwargs):
    adaln_out = self.adaLN_modulation(c)
    print(f"  [DiTBlock] x={list(x.shape)}, c={list(c.shape)}, adaLN_out={list(adaln_out.shape)}")
    chunks = adaln_out.chunk(6, dim=1)
    print(f"    chunks: {len(chunks)}, each: {list(chunks[0].shape)}")
    return _orig_forward(self, x, c, mask=mask, **kwargs)
DiTBlock.forward = debug_forward

rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
vfp = rt.model.core.velocity_field_predictor
core = rt.model.core
vfp.eval()

print(f"vfp mode: {vfp.mode}")

cpp_input = np.fromfile(f'{DEBUG_DIR}/cpp_dit_input.bin', dtype=np.float32)
seq_len = cpp_input.size // 1024
x_input = torch.from_numpy(cpp_input.reshape(1, seq_len, 1024).copy()).float()

t = torch.zeros(1)  # shape [1] for time_embedder (scalar-like)
spk_emb = torch.zeros(1, 512)
g_cond = core.xvec_proj(spk_emb)

print(f"\nInput: x={list(x_input.shape)}, t={list(t.shape)}, g_cond={list(g_cond.shape)}")

# Run step by step
with torch.no_grad():
    t_emb = vfp.time_embedder(t)
    print(f"time_embedder: {list(t_emb.shape)}, RMS={t_emb.norm().item()/math.sqrt(1024):.4f}")
    c = t_emb + g_cond
    print(f"cond c: {list(c.shape)}, RMS={c.norm().item()/math.sqrt(1024):.4f}")
    
    x2 = vfp.input_layer(x_input)
    print(f"after input_layer: {list(x2.shape)}")
    
    for i, block in enumerate(vfp.blocks):
        x2 = block(x2, c)
        if i <= 1:
            print(f"  block[{i}] out RMS: {x2.norm().item()/math.sqrt(x2.numel()):.4f}")
    
    py_out = vfp.output_layer(x2, c)
    print(f"output_layer: {list(py_out.shape)}")

py_out = py_out.detach().cpu().float()
print(f"\nPython output RMS: {py_out.norm().item()/math.sqrt(py_out.numel()):.4f}")

# Compare with C++
cpp_vel = np.fromfile(f'{DEBUG_DIR}/cpp_velocity.bin', dtype=np.float32)
cpp_rms = math.sqrt(np.mean(cpp_vel**2))
print(f"C++ output RMS: {cpp_rms:.4f}")

py_flat = py_out.reshape(-1).numpy()
cpp_flat = cpp_vel
corr = np.corrcoef(py_flat, cpp_flat)[0, 1]
max_diff = np.abs(py_flat - cpp_flat).max()
rms_diff = math.sqrt(np.mean((py_flat - cpp_flat)**2))
print(f"\nCorrelation: {corr:.6f}")
print(f"Max |diff|: {max_diff:.6f}")
print(f"RMS diff: {rms_diff:.6f}")

# Save Python output
py_flat.astype(np.float32).tofile(f'{DEBUG_DIR}/py_velocity.bin')
print(f"\nSaved py_velocity.bin")
print("Done.")
