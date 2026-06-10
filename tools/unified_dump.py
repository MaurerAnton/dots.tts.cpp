#!/usr/bin/env python3.12
"""Unified dump: verify block 0 via BOTH DiTBlock.forward AND manual reconstruction."""
import sys, os, math, torch, numpy as np

MODEL_DIR = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
DEBUG_DIR = '/home/bym/dots.tts.cpp/debug'
sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.modules.backbone.layers import apply_rotary_pos_emb

rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
vfp = rt.model.core.velocity_field_predictor; vfp.eval()
core = rt.model.core

cpp_input = np.fromfile(f'{DEBUG_DIR}/cpp_dit_input.bin', dtype=np.float32)
x_in = torch.from_numpy(cpp_input.reshape(1, 8, 1024).copy()).float()
t = torch.zeros(1); g_cond = core.xvec_proj(torch.zeros(1,512))
c = vfp.time_embedder(t) + g_cond
x = vfp.input_layer(x_in)

b0 = vfp.blocks[0]
adaln = b0.adaLN_modulation(c)
sa, sca, ga, sf, scf, gf = [t.squeeze(0) for t in adaln.chunk(6, dim=1)]

# Path A: DiTBlock.forward (TRUTH)
with torch.no_grad():
    out_A = b0(x, c).detach()
rms_A = out_A.norm().item() / math.sqrt(out_A.numel())

# Path B: Manual reconstruction
norm1 = torch.nn.functional.layer_norm(x, [1024], eps=1e-5)
mod1 = norm1 * (1 + sca) + sa

attn = b0.attn
q = attn.q_proj(mod1); k = attn.k_proj(mod1); v = attn.v_proj(mod1)
q = q.view(1, 8, 16, 64).transpose(1, 2)
k = k.view(1, 8, 16, 64).transpose(1, 2)
v = v.view(1, 8, 16, 64).transpose(1, 2)
q, k = attn.q_norm(q), attn.k_norm(k)
rotary_emb = attn.rotary(torch.arange(8))
q, k = apply_rotary_pos_emb(rotary_emb, q), apply_rotary_pos_emb(rotary_emb, k)
mask_inf = torch.triu(torch.ones(8,8)*float('-inf'), diagonal=1)
attn_w = torch.softmax((q @ k.transpose(-2,-1)) / math.sqrt(64) + mask_inf, dim=-1)
attn_out = (attn_w @ v).transpose(1,2).contiguous().view(1, 8, 1024)
attn_final = attn.o_proj(attn_out)
after_attn = x + ga.unsqueeze(0) * attn_final

norm2 = torch.nn.functional.layer_norm(after_attn, [1024], eps=1e-5)
mod2 = norm2 * (1 + scf) + sf

ffn_out = b0.ffn(mod2)
out_B = (after_attn + gf.unsqueeze(0) * ffn_out).detach()
rms_B = out_B.norm().item() / math.sqrt(out_B.numel())

print(f"A (DiTBlock.forward): RMS={rms_A:.4f}")
print(f"B (manual):          RMS={rms_B:.4f}")
print(f"max_diff A vs B: {torch.abs(out_A - out_B).max().item():.6f}")

# Check FFN internals
norm2_dit = b0.norm2(after_attn)
mod2_dit = norm2_dit * (1 + scf) + sf
print(f"norm2 manual vs dit: {torch.abs(norm2_dit - norm2).max().item():.8f}")
print(f"mod2 manual vs dit:  {torch.abs(mod2_dit - mod2).max().item():.8f}")

fc1_ffn = b0.ffn.fc1(mod2)  # via b0.ffn
fc1_manual = mod2 @ b0.ffn.fc1.weight.T + b0.ffn.fc1.bias
print(f"fc1 manual vs ffn:   {torch.abs(fc1_ffn - fc1_manual).max().item():.8f}")

print(f"GELU fn: {b0.ffn.act}")
gelu_ffn = b0.ffn.act(fc1_manual)
gelu_erf = torch.nn.functional.gelu(fc1_manual)
gelu_tanh = torch.nn.functional.gelu(fc1_manual, approximate='tanh')
print(f"GELU ffn vs erf:     {torch.abs(gelu_ffn - gelu_erf).max().item():.6f}")
print(f"GELU ffn vs tanh:    {torch.abs(gelu_ffn - gelu_tanh).max().item():.6f}")

fc2_ffn = b0.ffn.fc2(gelu_ffn)
fc2_manual = gelu_tanh @ b0.ffn.fc2.weight.T + b0.ffn.fc2.bias
print(f"fc2 manual vs ffn:   {torch.abs(fc2_ffn - fc2_manual).max().item():.6f}")
