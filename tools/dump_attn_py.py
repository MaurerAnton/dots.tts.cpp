#!/usr/bin/env python3.12
"""Dump Python attention intermediates for block 0."""
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
    with open(path, 'wb') as f: f.write(data.tobytes())

print("Loading...")
rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
vfp = rt.model.core.velocity_field_predictor; vfp.eval()
core = rt.model.core

seq_len, hidden = 8, 1024
n_heads, head_dim = 16, 64
x = torch.zeros(1, seq_len, hidden)
t = torch.zeros(1)
spk = torch.zeros(1, 512)
g_cond = core.xvec_proj(spk)

# Hooks
il_out = {}
b0_attn_in = {}
b0_attn_out = {}
b0_out = {}
h1 = vfp.input_layer.register_forward_hook(lambda m,i,o: il_out.__setitem__('out', o.detach()))
h2 = vfp.blocks[0].attn.register_forward_hook(lambda m,i,o: (b0_attn_in.__setitem__('in', i[0].detach()), b0_attn_out.__setitem__('out', o.detach())))
h3 = vfp.blocks[0].register_forward_hook(lambda m,i,o: b0_out.__setitem__('out', o.detach()))

with torch.no_grad():
    _ = vfp(x, t, g_cond=g_cond)
h1.remove(); h2.remove(); h3.remove()

os.makedirs(OUT_DIR, exist_ok=True)

# Now compute attention step by step using the ACTUAL modulated input
attn = vfp.blocks[0].attn
modulated = b0_attn_in['in']  # [1, 8, 1024] — input to attention
write_bin('b0_attn_input', modulated)

with torch.no_grad():
    # QKV
    q = attn.q_proj(modulated)  # [1, 8, 1024]
    k = attn.k_proj(modulated)
    v = attn.v_proj(modulated)
    write_bin('b0_q_raw', q)
    write_bin('b0_k_raw', k)
    write_bin('b0_v_raw', v)
    
    # Reshape: [1, 8, 1024] -> [1, 16, 8, 64]
    q = q.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    v = v.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    write_bin('b0_q_reshaped', q)
    write_bin('b0_k_reshaped', k)
    write_bin('b0_v_reshaped', v)
    
    # qk_norm
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
    write_bin('b0_v_rope', v)  # V doesn't change
    
    # Scores
    scale = 1.0 / math.sqrt(head_dim)
    scores = (q_rope @ k_rope.transpose(-2, -1)) * scale
    write_bin('b0_scores', scores)
    
    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    write_bin('b0_scores_masked', scores)
    
    # Softmax
    attn_w = torch.softmax(scores, dim=-1)
    write_bin('b0_attn_weights', attn_w)
    
    # Output
    attn_out = attn_w @ v  # [1, 16, 8, 64]
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, seq_len, hidden)
    write_bin('b0_attn_pre_o', attn_out)
    
    # O projection
    attn_final = attn.o_proj(attn_out)
    write_bin('b0_attn_final', attn_final)

print("Done.")
