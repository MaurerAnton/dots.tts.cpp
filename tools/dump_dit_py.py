#!/usr/bin/env python3.12
"""Dump Python block 0 attention intermediates for C++ calibration."""
import sys, os, math
import torch
import numpy as np

MODEL_DIR = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
OUT_DIR = '/home/bym/dots.tts.cpp/debug'

sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime

def write_bin(name, arr):
    data = arr.detach().cpu().numpy().astype(np.float32).ravel()
    path = os.path.join(OUT_DIR, f'py_{name}.bin')
    rms = math.sqrt(np.mean(data**2))
    print(f'  py_{name}: {len(data)} elems, rms={rms:.4f}')
    with open(path, 'wb') as f:
        f.write(data.tobytes())

print("Loading...")
rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
vfp = rt.model.core.velocity_field_predictor; vfp.eval()
core = rt.model.core

seq_len, hidden = 8, 1024
x = torch.zeros(1, seq_len, hidden)
t_emb = vfp.time_embedder(torch.zeros(1,1))
g_cond = core.xvec_proj(torch.zeros(1,512))
c = t_emb + g_cond

with torch.no_grad():
    il_w = vfp.input_layer.weight; il_b = vfp.input_layer.bias
    h = x @ il_w.T + il_b
    write_bin('b0_x_in', h)

    b0 = vfp.blocks[0]
    adaln_out = b0.adaLN_modulation(c)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_out.chunk(6, dim=-1)
    write_bin('b0_shift_msa', shift_msa); write_bin('b0_scale_msa', scale_msa); write_bin('b0_gate_msa', gate_msa)

    # adaLN modulation
    normed = b0.norm1(h)  # RMS norm
    modulated = normed * (1 + scale_msa) + shift_msa
    write_bin('b0_adaln_modulated', modulated)

    # QKV
    attn = b0.attn
    q = attn.q_proj(modulated); k = attn.k_proj(modulated); v = attn.v_proj(modulated)
    write_bin('b0_q', q); write_bin('b0_k', k); write_bin('b0_v', v)

    # Reshape for multi-head
    q = q.view(1, seq_len, 16, 64).transpose(1, 2)  # [1,16,8,64]
    k = k.view(1, seq_len, 16, 64).transpose(1, 2)
    v = v.view(1, seq_len, 16, 64).transpose(1, 2)

    # qk_norm
    q = attn.q_norm(q); k = attn.k_norm(k)
    write_bin('b0_q_normed', q); write_bin('b0_k_normed', k)

    # RoPE
    q_rope, k_rope = attn.rotary_emb(q, k)
    write_bin('b0_q_rope', q_rope); write_bin('b0_k_rope', k_rope)

    # Scores
    scale = 1.0 / math.sqrt(64)
    scores = (q_rope @ k_rope.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    attn_w = torch.softmax(scores, dim=-1)
    write_bin('b0_attn_weights', attn_w)

    # Attn output
    attn_out = attn_w @ v
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, seq_len, hidden)
    write_bin('b0_attn_before_o', attn_out)
    attn_out = attn.o_proj(attn_out)
    write_bin('b0_attn_out', attn_out)

    # Residual
    h = h + gate_msa * attn_out
    write_bin('b0_after_attn', h)

    # FFN
    normed = b0.norm2(h)
    modulated = normed * (1 + scale_mlp) + shift_mlp
    ffn_out = b0.ffn(modulated)
    h = h + gate_mlp * ffn_out
    write_bin('b0_out', h)

print("Done.")
