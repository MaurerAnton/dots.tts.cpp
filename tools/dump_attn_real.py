#!/usr/bin/env python3.12
"""Dump Python block 0 attention intermediates using C++ input (real text)."""
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
    rms = math.sqrt(np.mean(data**2))
    print(f'  py_{name}: {len(data)} elems, rms={rms:.4f}')
    with open(path, 'wb') as f: f.write(data.tobytes())
    return data

def load_cpp(name):
    path = os.path.join(DEBUG_DIR, f'cpp_{name}.bin')
    return np.fromfile(path, dtype=np.float32)

print("Loading...")
rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
vfp = rt.model.core.velocity_field_predictor; vfp.eval()
core = rt.model.core

# Load C++ input
cpp_input = np.fromfile(f'{DEBUG_DIR}/cpp_dit_input.bin', dtype=np.float32)
seq_len = cpp_input.size // 1024
assert seq_len == 8, f"Expected 8 tokens, got {seq_len}"
x_input = torch.from_numpy(cpp_input.reshape(1, seq_len, 1024).copy()).float()

# Same setup as C++: t=0, speaker=zeros
t = torch.zeros(1)
spk_emb = torch.zeros(1, 512)
g_cond = core.xvec_proj(spk_emb)

hidden, n_heads, head_dim = 1024, 16, 64

with torch.no_grad():
    t_emb = vfp.time_embedder(t)
    c = t_emb + g_cond
    write_bin('b0_cond', c.reshape(-1, 1024))  # [1,1024]
    
    cs = torch.nn.functional.silu(c)
    write_bin('b0_cond_silu', cs.reshape(-1, 1024))
    
    # input_layer
    x = vfp.input_layer(x_input)
    write_bin('b0_x_in', x)
    
    b0 = vfp.blocks[0]
    adaln_out = b0.adaLN_modulation(c)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_out.chunk(6, dim=-1)
    write_bin('b0_shift_msa', shift_msa)
    write_bin('b0_scale_msa', scale_msa)
    write_bin('b0_gate_msa', gate_msa)
    
    # norm1 (LayerNorm without affine, eps=1e-5)
    x_normed = torch.nn.functional.layer_norm(x, [1024], eps=1e-5)
    write_bin('b0_normed', x_normed)
    
    # modulated
    x_mod = x_normed * (1 + scale_msa) + shift_msa
    write_bin('b0_adaln_modulated', x_mod)
    
    # Q/K/V projections
    attn = b0.attn
    q = attn.q_proj(x_mod)
    k = attn.k_proj(x_mod)
    v = attn.v_proj(x_mod)
    write_bin('b0_q_raw', q)
    write_bin('b0_k_raw', k)
    write_bin('b0_v_raw', v)
    
    # Reshape to [1, 16, 8, 64]
    q = q.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    v = v.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    
    # qk_norm - RMSNorm with per-head weight
    qn = attn.q_norm(q)
    kn = attn.k_norm(k)
    write_bin('b0_q_normed', qn)
    write_bin('b0_k_normed', kn)
    
    # RoPE
    rotary_emb = attn.rotary(torch.arange(seq_len))
    from dots_tts.modules.backbone.layers import apply_rotary_pos_emb
    q_rope = apply_rotary_pos_emb(rotary_emb, qn)
    k_rope = apply_rotary_pos_emb(rotary_emb, kn)
    write_bin('b0_q_rope', q_rope)
    write_bin('b0_k_rope', k_rope)
    
    # Attention scores (raw, before mask)
    scale = 1.0 / math.sqrt(head_dim)
    scores_raw = (q_rope @ k_rope.transpose(-2, -1)) * scale
    write_bin('b0_scores_raw', scores_raw)
    
    # Causal mask + softmax
    mask = torch.triu(torch.ones(seq_len, seq_len, device=scores_raw.device), diagonal=1).bool()
    scores = scores_raw.clone()
    scores.masked_fill_(mask, float('-inf'))
    attn_w = torch.softmax(scores, dim=-1)
    write_bin('b0_attn_weights', attn_w)
    
    # Attention output (weighted sum of V)
    attn_out = attn_w @ v  # [1, 16, 8, 64]
    attn_out_flat = attn_out.transpose(1, 2).contiguous().view(1, seq_len, hidden)
    write_bin('b0_attn_pre_o', attn_out_flat)
    
    # O-projection
    attn_final = attn.o_proj(attn_out_flat)
    write_bin('b0_attn_out', attn_final)
    
    # Residual: x + gate * attn_out
    after_attn = x + gate_msa * attn_final
    write_bin('b0_after_attn', after_attn)

print("\nDone. Now comparing...\n")

# Compare
for name in ['b0_x_in', 'b0_cond', 'b0_cond_silu', 'b0_shift_msa', 'b0_scale_msa', 'b0_gate_msa',
             'b0_normed', 'b0_adaln_modulated', 'b0_q_raw', 'b0_k_raw', 'b0_v_raw',
             'b0_q_normed', 'b0_k_normed', 'b0_q_rope', 'b0_k_rope',
             'b0_scores_raw', 'b0_attn_weights', 'b0_attn_pre_o', 'b0_attn_out', 'b0_after_attn']:
    cpp = load_cpp(name)
    py_path = os.path.join(DEBUG_DIR, f'py_{name}.bin')
    if not os.path.exists(py_path):
        print(f"  {name}: MISSING py dump")
        continue
    py_data = np.fromfile(py_path, dtype=np.float32)
    if len(cpp) != len(py_data):
        print(f"  {name}: SIZE MISMATCH cpp={len(cpp)} py={len(py_data)}")
        continue
    max_diff = np.abs(cpp - py_data).max()
    corr = np.corrcoef(cpp, py_data)[0, 1] if len(cpp) > 1 else 1.0
    cpp_rms = math.sqrt(np.mean(cpp**2))
    py_rms = math.sqrt(np.mean(py_data**2))
    status = "✓" if max_diff < 0.001 else ("~" if max_diff < 0.1 else "✗")
    print(f"  {name}: max_diff={max_diff:.6f} corr={corr:.6f} cpp_rms={cpp_rms:.4f} py_rms={py_rms:.4f} {status}")
