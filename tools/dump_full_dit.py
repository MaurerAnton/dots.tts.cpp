#!/usr/bin/env python3.12
"""Dump Python DiT blocks + final velocity using C++ input (real text)."""
import sys, os, math
import torch
import numpy as np

MODEL_DIR = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
DEBUG_DIR = '/home/bym/dots.tts.cpp/debug'

sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime

def write_bin(name, arr):
    data = arr.detach().cpu().numpy().astype(np.float32).ravel()
    path = os.path.join(DEBUG_DIR, f'py_{name}.bin')
    with open(path, 'wb') as f: f.write(data.tobytes())
    rms = math.sqrt(np.mean(data**2))
    print(f'  py_{name}: rms={rms:.4f}')

print("Loading...")
rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
vfp = rt.model.core.velocity_field_predictor; vfp.eval()
core = rt.model.core

# Load C++ input
cpp_input = np.fromfile(f'{DEBUG_DIR}/cpp_dit_input.bin', dtype=np.float32)
seq_len = cpp_input.size // 1024
x_input = torch.from_numpy(cpp_input.reshape(1, seq_len, 1024).copy()).float()

t = torch.zeros(1)
spk_emb = torch.zeros(1, 512)
g_cond = core.xvec_proj(spk_emb)

with torch.no_grad():
    # Run Python DiT forward, dumping per-block outputs
    t_emb = vfp.time_embedder(t)
    c = t_emb + g_cond
    
    x = vfp.input_layer(x_input)
    write_bin('dit_input_layer', x)
    
    for i, block in enumerate(vfp.blocks):
        x = block(x, c)
        write_bin(f'dit_block_{i}', x)
    
    # Output layer
    out = vfp.output_layer(x, c)
    write_bin('dit_velocity', out)

print("\nDone. Comparing...\n")

def compare(name, cpp_path=None):
    if cpp_path is None:
        cpp_path = os.path.join(DEBUG_DIR, f'cpp_{name}.bin')
    py_path = os.path.join(DEBUG_DIR, f'py_{name}.bin')
    if not os.path.exists(cpp_path) or not os.path.exists(py_path):
        print(f"  {name}: MISSING")
        return
    cpp = np.fromfile(cpp_path, dtype=np.float32)
    py_data = np.fromfile(py_path, dtype=np.float32)
    if len(cpp) != len(py_data):
        print(f"  {name}: SIZE cpp={len(cpp)} py={len(py_data)}")
        return
    max_diff = np.abs(cpp - py_data).max()
    corr = np.corrcoef(cpp, py_data)[0, 1]
    cpp_rms = math.sqrt(np.mean(cpp**2))
    py_rms = math.sqrt(np.mean(py_data**2))
    status = "✓" if max_diff < 0.001 else ("~" if max_diff < 0.1 else "✗")
    print(f"  {name}: max_diff={max_diff:.6f} corr={corr:.6f} cpp={cpp_rms:.4f} py={py_rms:.4f} {status}")

compare('dit_input_layer')
for i in range(18):
    compare(f'dit_block_{i}')
compare('dit_velocity')
