#!/usr/bin/env python3.12
"""Compare key DiT intermediate values C++ vs Python with fresh dump."""
import sys, os, math
import torch
import numpy as np

MODEL_DIR = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
DEBUG_DIR = '/home/bym/dots.tts.cpp/debug'

sys.path.insert(0, '/home/bym/llama/dots.tts/src')
from dots_tts.runtime import DotsTtsRuntime

cpp_input = np.fromfile(f'{DEBUG_DIR}/cpp_dit_input.bin', dtype=np.float32)
seq_len = cpp_input.size // 1024

rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision='float32')
vfp = rt.model.core.velocity_field_predictor
core = rt.model.core
vfp.eval()

x_t = torch.from_numpy(cpp_input.reshape(1, seq_len, 1024).copy()).float()
g_cond = core.xvec_proj(torch.zeros(1, 512))

def rms(t): return t.norm().item()/math.sqrt(t.numel())

print("C++ reported values (from e2e_pipeline run):")
print("  cpp_il_out:    rms=2.4144 first3=[-3.2842,-5.2810,1.1392]")
print("  cpp_attn_normed: rms=1.0000 first3=[-0.4929,-0.7968,0.1803]")
print("  cpp_attn_mod:  rms=0.5563 first3=[-0.0564,0.2117,0.2282]")
print("  cpp_normed:    rms=1.0000 first3=[-0.4899,-0.7651,0.1014]")
print("  cpp_mod:       rms=0.5569")
print("  cpp_attn_out:  rms=2.9007")
print("  cpp_after_attn: rms=5.1547")
print("  cpp_ffn_out:   rms=1.3217")
print("  cpp_b0:        rms=6.3832 first3=[-3.5935,-5.5781,0.6119]")
print()

with torch.no_grad():
    # Python matching
    t_emb = vfp.time_embedder(torch.zeros(1))
    c = t_emb + g_cond
    x = vfp.input_layer(x_t)
    print(f"Python il_out:    rms={rms(x):.4f} first3=[{x[0,0,0]:.4f},{x[0,0,1]:.4f},{x[0,0,2]:.4f}]")
    
    b0 = vfp.blocks[0]
    
    # adaLN
    adaln = b0.adaLN_modulation(c)
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = adaln.chunk(6, dim=-1)
    print(f"Python shift_msa: rms={rms(shift_a):.4f}  scale_msa: rms={rms(scale_a):.4f}  gate_msa: rms={rms(gate_a):.4f}")
    
    # norm1 (LayerNorm without affine)
    x_normed = torch.nn.functional.layer_norm(x, [1024], eps=1e-5)
    print(f"Python normed:    rms={rms(x_normed):.4f} first3=[{x_normed[0,0,0]:.4f},{x_normed[0,0,1]:.4f},{x_normed[0,0,2]:.4f}]")
    
    # modulated
    x_mod = x_normed * (1 + scale_a) + shift_a
    print(f"Python modulated: rms={rms(x_mod):.4f} first3=[{x_mod[0,0,0]:.4f},{x_mod[0,0,1]:.4f},{x_mod[0,0,2]:.4f}]")
    
    # Attention
    attn = b0.attn
    q = attn.q_proj(x_mod); k = attn.k_proj(x_mod); v = attn.v_proj(x_mod)
    q = q.view(1, seq_len, 16, 64).transpose(1, 2)
    k = k.view(1, seq_len, 16, 64).transpose(1, 2)
    v = v.view(1, seq_len, 16, 64).transpose(1, 2)
    q = attn.q_norm(q); k = attn.k_norm(k)
    
    # RoPE
    from dots_tts.modules.backbone.layers import apply_rotary_pos_emb
    rotary_emb = attn.rotary(torch.arange(seq_len, device=q.device))
    q_rope = apply_rotary_pos_emb(rotary_emb, q)
    k_rope = apply_rotary_pos_emb(rotary_emb, k)
    
    # Attention scores
    scale = 1.0 / math.sqrt(64)
    scores = (q_rope @ k_rope.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    attn_w = torch.softmax(scores, dim=-1)
    attn_out = attn_w @ v
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, seq_len, 1024)
    attn_out = attn.o_proj(attn_out)
    print(f"Python attn_out:  rms={rms(attn_out):.4f}")
    
    # After attention
    x2 = x + gate_a * attn_out
    print(f"Python after_attn: rms={rms(x2):.4f}")
    
    # FFN
    x_normed2 = torch.nn.functional.layer_norm(x2, [1024], eps=1e-5)
    x_mod2 = x_normed2 * (1 + scale_f) + shift_f
    print(f"Python norm2:     rms={rms(x_normed2):.4f} first3=[{x_normed2[0,0,0]:.4f},{x_normed2[0,0,1]:.4f},{x_normed2[0,0,2]:.4f}]")
    print(f"Python mod2:      rms={rms(x_mod2):.4f}")
    
    ffn_out = b0.ffn(x_mod2)
    print(f"Python ffn_out:   rms={rms(ffn_out):.4f}")
    
    x3 = x2 + gate_f * ffn_out
    print(f"Python block 0:   rms={rms(x3):.4f} first3=[{x3[0,0,0]:.4f},{x3[0,0,1]:.4f},{x3[0,0,2]:.4f}]")
    
    # Compare with C++
    # C++ block 0 output: rms=6.3832 first3=[-3.5935,-5.5781,0.6119]
    
print("\nBlock 0 comparison:")
print(f"  C++:  rms=6.3832  first3=[-3.5935,-5.5781, 0.6119]")
print(f"  Py:   rms={rms(x3):.4f}  first3=[{x3[0,0,0]:.4f},{x3[0,0,1]:.4f},{x3[0,0,2]:.4f}]")
