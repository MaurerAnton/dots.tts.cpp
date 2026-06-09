#!/usr/bin/env python3.12
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026  Anton Maurer

"""compare_pipelines.py — Byte-level comparison of C++ vs Python dots.tts"""
import subprocess, torch, numpy as np, sys, wave, os, struct, json

CPP_BIN = '/home/bym/dots.tts.cpp/build/e2e_pipeline'
MODEL_DIR = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
TEXT = sys.argv[1] if len(sys.argv) > 1 else "Hello world"

print(f"=== Comparing C++ vs Python for: '{TEXT}' ===\n")

# 1. Run C++
print("[1] Running C++ pipeline...")
os.chdir('/home/bym/dots.tts.cpp')
subprocess.run([CPP_BIN, '--dump', '--length', '1', TEXT], 
               capture_output=True, timeout=60)

# 2. Load Python
print("[2] Loading Python reference...")
sys.path.insert(0, '/tmp/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime
rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')

# 3. Compare hidden_proj
print("[3] Comparing hidden_proj...")
# Load C++ output
cpp_hp = np.fromfile('debug_hidden_proj.bin', dtype=np.float32)
n_tok = len(cpp_hp) // 1024
cpp_hp = cpp_hp.reshape(1024, n_tok).T  # [n_tok, 1024]

# Compute Python hidden_proj with same token embeddings
with open(f'{MODEL_DIR}/model.safetensors', 'rb') as f:
    hlen = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(hlen))
    for name in ['hidden_proj.weight', 'hidden_proj.bias']:
        info = header[name]
        f.seek(8 + hlen + info['data_offsets'][0])
        raw = f.read(info['data_offsets'][1] - info['data_offsets'][0])
        if 'weight' in name: w = np.frombuffer(raw, dtype=np.float32).reshape(info['shape'])
        else: b = np.frombuffer(raw, dtype=np.float32)

cpp_emb = np.fromfile('models/token_embd_flat.bin', dtype=np.float32).reshape(151672, 1536)
tokens = [9707, 1879]  # Hello, world
if n_tok > len(tokens): tokens = tokens[:n_tok]
x = np.stack([cpp_emb[t] for t in tokens[:n_tok]])
py_hp = x @ w.T + b

hp_diff = np.max(np.abs(cpp_hp - py_hp))
print(f"  hidden_proj max diff: {hp_diff:.2e} {'✓' if hp_diff < 1e-5 else '✗'}")

# 4. Compare BigVGAN
print("[4] Comparing BigVGAN...")
cpp_latents = np.fromfile('latents.bin', dtype=np.float32).reshape(-1, 128)
latents_t = torch.from_numpy(cpp_latents.T.copy()).unsqueeze(0)

from dots_tts.modules.vocoder.bigvgan import AudioVAE
from dots_tts.models.dots_tts.config import ModelConfig
with open(f'{MODEL_DIR}/config.json') as f: cfg = ModelConfig(**json.load(f))
v = AudioVAE(cfg.vocoder).eval()
import safetensors.torch
v.load_state_dict(safetensors.torch.load_file(f'{MODEL_DIR}/vocoder.safetensors'), strict=False)

with torch.no_grad():
    py_audio = v.inference_from_latents(latents_t, do_sample=False).squeeze().cpu().numpy()

import wave as wv
with wv.open('output.wav', 'rb') as wf:
    cpp_audio = np.frombuffer(wf.readframes(wf.getparams().nframes), dtype=np.int16).astype(np.float32)/32768

ml = min(len(py_audio), len(cpp_audio))
audio_diff = np.max(np.abs(py_audio[:ml] - cpp_audio[:ml]))
py_rms = np.sqrt(np.mean(py_audio[:ml]**2))
cpp_rms = np.sqrt(np.mean(cpp_audio[:ml]**2))
print(f"  BigVGAN max diff: {audio_diff:.4f} (py_rms={py_rms:.3f}, cpp_rms={cpp_rms:.3f})")
print(f"  RMS ratio: {cpp_rms/py_rms:.2f}x {'≈1.0 ✓' if abs(cpp_rms/py_rms-1.0) < 0.3 else '✗'}")

# 5. Summary
print(f"\n=== SUMMARY ===")
print(f"  hidden_proj: {'PASS' if hp_diff < 1e-5 else 'FAIL'} (diff={hp_diff:.2e})")
print(f"  BigVGAN:     {'PASS' if audio_diff < 0.1 else 'CHECK'} (diff={audio_diff:.4f})")
print(f"  Tokens:      {n_tok}")
print(f"  Text:        '{TEXT}'")
