#!/usr/bin/env python3.12
"""Dump Python block 0 FFN intermediates for comparison."""
import sys, os, math
import torch
import numpy as np

MODEL_DIR = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
DEBUG_DIR = '/home/bym/dots.tts.cpp/debug'

sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime

def write_bin(name, arr):
    data = arr.detach().cpu().numpy().astype(np.float32).ravel()
    with open(os.path.join(DEBUG_DIR, f'py_{name}.bin'), 'wb') as f:
        f.write(data.tobytes())

print("Loading...")
rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
vfp = rt.model.core.velocity_field_predictor; vfp.eval()
core = rt.model.core

cpp_input = np.fromfile(f'{DEBUG_DIR}/cpp_dit_input.bin', dtype=np.float32)
seq_len, hidden = cpp_input.size // 1024, 1024
x_input = torch.from_numpy(cpp_input.reshape(1, seq_len, 1024).copy()).float()

t = torch.zeros(1)
spk_emb = torch.zeros(1, 512)
g_cond = core.xvec_proj(spk_emb)

with torch.no_grad():
    t_emb = vfp.time_embedder(t)
    c = t_emb + g_cond
    x = vfp.input_layer(x_input)
    
    b0 = vfp.blocks[0]
    adaln = b0.adaLN_modulation(c)
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = adaln.chunk(6, dim=-1)
    write_bin('b0_ffn_shift', shift_f)
    write_bin('b0_ffn_scale', scale_f)
    write_bin('b0_ffn_gate', gate_f)
    
    # Attention step (verified correct)
    x_normed = torch.nn.functional.layer_norm(x, [1024], eps=1e-5)
    x_mod = x_normed * (1 + scale_a) + shift_a
    attn = b0.attn
    q = attn.q_proj(x_mod); k = attn.k_proj(x_mod); v = attn.v_proj(x_mod)
    q = q.view(1, seq_len, 16, 64).transpose(1, 2)
    k = k.view(1, seq_len, 16, 64).transpose(1, 2)
    v = v.view(1, seq_len, 16, 64).transpose(1, 2)
    q, k = attn.q_norm(q), attn.k_norm(k)
    rotary_emb = attn.rotary(torch.arange(seq_len))
    from dots_tts.modules.backbone.layers import apply_rotary_pos_emb
    q, k = apply_rotary_pos_emb(rotary_emb, q), apply_rotary_pos_emb(rotary_emb, k)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(64)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    attn_w = torch.softmax(scores, dim=-1)
    attn_out = attn_w @ v
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, seq_len, 1024)
    attn_final = attn.o_proj(attn_out)
    h_attn = x + gate_a * attn_final
    
    write_bin('b0_ffn_input', h_attn)
    
    # FFN step
    x_normed2 = torch.nn.functional.layer_norm(h_attn, [1024], eps=1e-5)
    write_bin('b0_ffn_normed', x_normed2)
    x_mod2 = x_normed2 * (1 + scale_f) + shift_f
    write_bin('b0_ffn_modulated', x_mod2)
    
    # fc1 (Linear 1024->4096)
    fc1_w = b0.ffn.fc1.weight  # [4096, 1024]
    fc1_b = b0.ffn.fc1.bias    # [4096]
    fc1_out = x_mod2 @ fc1_w.T + fc1_b
    write_bin('b0_ffn_fc1', fc1_out)
    
    # GELU
    gelu_out = torch.nn.functional.gelu(fc1_out)
    write_bin('b0_ffn_gelu', gelu_out)
    
    # fc2 (Linear 4096->1024)
    fc2_w = b0.ffn.fc2.weight  # [1024, 4096]
    fc2_b = b0.ffn.fc2.bias    # [1024]
    ffn_out = gelu_out @ fc2_w.T + fc2_b
    write_bin('b0_ffn_fc2', ffn_out)
    
    # Residual
    block_out = h_attn + gate_f * ffn_out
    write_bin('b0_ffn_output', block_out)
    
    print(f"Python block 0 (full): RMS={block_out.norm().item()/math.sqrt(block_out.numel()):.4f}")

print("Done.")
