#!/usr/bin/env python3.12
"""Final analysis: C++ velocity field vs Python, ODE integrator, scale=0.45"""
import sys, os, math
import torch
import numpy as np

MODEL_DIR = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
DEBUG_DIR = '/home/bym/dots.tts.cpp/debug'

sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime

print("="*70)
print("1. VELOCITY FIELD COMPARISON (fresh C++ dump)")
print("="*70)

cpp_input = np.fromfile(f'{DEBUG_DIR}/cpp_dit_input.bin', dtype=np.float32)
cpp_vel = np.fromfile(f'{DEBUG_DIR}/cpp_velocity.bin', dtype=np.float32)
seq_len = cpp_input.size // 1024
latent_dim = 128

print(f"C++ input:  [{seq_len},1024] RMS={math.sqrt(np.mean(cpp_input**2)):.4f}")
print(f"C++ velocity: [{seq_len},128] RMS={math.sqrt(np.mean(cpp_vel**2)):.4f}")

rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
vfp = rt.model.core.velocity_field_predictor
core = rt.model.core
vfp.eval()

x_t = torch.from_numpy(cpp_input.reshape(1, seq_len, 1024).copy()).float()
t = torch.zeros(1)
g_cond = core.xvec_proj(torch.zeros(1, 512))

with torch.no_grad():
    py_vel = vfp(x=x_t, timesteps=t, g_cond=g_cond).cpu().float()

py_flat = py_vel.reshape(-1).numpy()
cpp_flat = cpp_vel.reshape(-1)

corr = np.corrcoef(py_flat, cpp_flat)[0,1]
print(f"\nPython RMS: {math.sqrt(np.mean(py_flat**2)):.4f}")
print(f"C++ RMS:    {math.sqrt(np.mean(cpp_flat**2)):.4f}")
print(f"Correlation: {corr:.6f}")
print(f"Max |diff|:  {np.abs(py_flat-cpp_flat).max():.6f}")
print(f"RMS diff:    {math.sqrt(np.mean((py_flat-cpp_flat)**2)):.6f}")

print(f"\nPer-position:")
for p in range(seq_len):
    c = cpp_flat[p*128:(p+1)*128]; y = py_flat[p*128:(p+1)*128]
    print(f"  Pos {p}: C++={math.sqrt(np.mean(c**2)):.4f}  Py={math.sqrt(np.mean(y**2)):.4f}  Corr={np.corrcoef(c,y)[0,1]:.4f}")

# ============================================================
print("\n"+"="*70)
print("2. ODE INTEGRATOR: z_t += v_t * vs * dt")
print("="*70)

nfe, dt = 10, 0.1
patch_size = 5

# Python reference: Run proper flow matching
with torch.no_grad():
    z_py = torch.randn(1, patch_size, latent_dim) * 1.0
    for step in range(nfe):
        t_val = step * dt
        z_proj = core.coordinate_proj(z_py)
        z_c = torch.zeros(1, 1 + patch_size, 1024)  # 1 text+patch_size noise
        z_c[:, 1:, :] = z_proj
        v = vfp(x=z_c, timesteps=torch.tensor([t_val]), g_cond=g_cond)
        v_noise = v[:, 1:, :]
        z_py = z_py + v_noise * dt

print(f"Python Euler ODE (nfe={nfe}, no caps/scales):")
print(f"  Final RMS: {z_py.norm().item()/math.sqrt(z_py.numel()):.4f}")

# C++ with the SAME velocity field but applying the vs cap + scale
np.random.seed(42)
z_cpp = np.random.randn(patch_size * 128).astype(np.float32)
z_cpp = np.clip(z_cpp, -5, 5)
z_orig = z_cpp.copy()

# Use Python velocity field (as C++ ideally should be identical)
with torch.no_grad():
    for step in range(nfe):
        t_val = step * dt
        z_t = torch.from_numpy(z_cpp.reshape(1, patch_size, 128).copy()).float()
        z_proj = core.coordinate_proj(z_t)
        z_c = torch.zeros(1, 1 + patch_size, 1024)
        z_c[:, 1:, :] = z_proj
        v = vfp(x=z_c, timesteps=torch.tensor([t_val]), g_cond=g_cond)
        v_flat = v[:, 1:, :].reshape(-1).numpy()
        vrms = math.sqrt(np.mean(v_flat**2))
        vs = min(1.0, 15.0 / vrms) if vrms > 0 else 1.0
        capped = " (CAPPED!)" if vs < 0.999 else ""
        z_cpp += v_flat * vs * dt
        z_cpp = np.clip(z_cpp, -50, 50)
        if step < 3 or step >= nfe-1:
            vrms_v = math.sqrt(np.mean(v_flat**2))
            print(f"  Step {step} (t={t_val:.2f}): v_rms={vrms_v:.4f}, vs={vs:.3f}{capped}, z_rms={math.sqrt(np.mean(z_cpp**2)):.4f}")

print(f"\nC++-style ODE (with vs cap, but using Python velocity):")
print(f"  Final RMS:      {math.sqrt(np.mean(z_cpp**2)):.4f}")
print(f"  After ×0.45:    {math.sqrt(np.mean(z_cpp**2))*0.45:.4f}")
print(f"\n  Ratio to Python: {math.sqrt(np.mean(z_cpp**2))/max(z_py.norm().item()/math.sqrt(z_py.numel()), 1e-10):.4f}")
print(f"  Ratio (w/scale): {math.sqrt(np.mean(z_cpp**2))*0.45/max(z_py.norm().item()/math.sqrt(z_py.numel()), 1e-10):.4f}")

# ============================================================
print("\n"+"="*70)
print("3. SCALE=0.45 vs NO SCALE")
print("="*70)
print(f"Without scale×0.45: z_t RMS = {math.sqrt(np.mean(z_cpp**2)):.4f} → after VAE norm: ~{math.sqrt(np.mean(z_cpp**2)):.1f}")
print(f"With scale×0.45:    z_t RMS = {math.sqrt(np.mean(z_cpp**2))*0.45:.4f} → after VAE norm: ~{math.sqrt(np.mean(z_cpp**2))*0.45:.1f}")
print(f"Python reference:    z_t RMS = {z_py.norm().item()/math.sqrt(z_py.numel()):.4f}")
print(f"\nThe scale=0.45 makes the latents {(math.sqrt(np.mean(z_cpp**2))*0.45/(z_py.norm().item()/math.sqrt(z_py.numel()))*100):.0f}% of Python reference")
print(f"Without scale: {(math.sqrt(np.mean(z_cpp**2))/(z_py.norm().item()/math.sqrt(z_py.numel()))*100):.0f}% of reference")
print(f"Without scale is closer to Python than with scale.")
print(f"CONCLUSION: scale=0.45 is a WRONG hack. Remove it once velocity field is calibrated.")

print("\n"+"="*70)
print("SUMMARY OF ALL ISSUES")
print("="*70)
print(f"1. Velocity field: Corr={corr:.4f} with Python → NOT calibrated for full 18 blocks")
print(f"   Per-position RMS differs (especially text positions)")
print(f"   Root cause: C++ manual DiT (blocks 1-17) produces different output than Python")
print(f"   Fix: calibrate blocks 1-17, or diagnose per-block divergence")
print(f"2. ODE vs velocity cap: vs cap at 15.0 is NOT triggered (velocity RMS < 2.5)")
print(f"   But the concept is wrong — it should NOT exist. Remove it.")
print(f"3. Final scale=0.45: Makes latents {math.sqrt(np.mean(z_cpp**2))*0.45/(z_py.norm().item()/math.sqrt(z_py.numel()))*100:.0f}% of reference")
print(f"   Python has NO such scale. This is a hack compensating for velocity miscalibration.")
print(f"   Remove after velocity field is calibrated.")
print("\nDone.")
