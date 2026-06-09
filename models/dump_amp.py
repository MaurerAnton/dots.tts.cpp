#!/usr/bin/env python3.12
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026  Anton Maurer

"""Dump first AMP block internals for exact comparison."""
import sys, numpy as np, torch
sys.path.insert(0, '/tmp/dots_tts_py')
from dots_tts.runtime import DotsTtsRuntime

np.random.seed(42)
latents = np.random.randn(16, 128).astype(np.float32) * 0.5

rt = DotsTtsRuntime.from_pretrained('rednote-hilab/dots.tts-soar', precision='float32')
vocoder = rt.model.vocoder
z = torch.from_numpy(latents).float().unsqueeze(0).transpose(1,2)

with torch.no_grad():
    x = vocoder.post_proj(z)
    x = x.permute(0, 2, 1)
    x = vocoder.dec_mi_layer(x)
    x = x.permute(0, 2, 1)
    x = vocoder.decoder.conv_pre(x)
    
    # Stage 0 upsample
    x = vocoder.decoder.ups[0][0](x)
    
    # First resblock (rb 0): AMPBlock1 with channels=768, kernel_size=3, dilation=[1,3,5]
    rb = vocoder.decoder.resblocks[0]
    print(f"RB0: ch={x.shape[1]}, len={x.shape[2]}")
    
    # Trace through the 3 conv pairs
    x_copy = x.clone()
    
    # Pair 0 (dilation=1)
    acts1 = rb.activations[::2]  # [0,2,4]
    acts2 = rb.activations[1::2]  # [1,3,5]
    
    for pair_idx, (c1, c2, a1, a2) in enumerate(zip(rb.convs1, rb.convs2, acts1, acts2)):
        xt = a1(x_copy)
        np.save(f'amp_pair{pair_idx}_a1.npy', xt.squeeze(0).numpy())
        print(f"  pair{pair_idx} a1: rms={xt.std().item():.4f}")
        
        xt = c1(xt)
        np.save(f'amp_pair{pair_idx}_c1.npy', xt.squeeze(0).numpy())
        print(f"  pair{pair_idx} c1: rms={xt.std().item():.4f}")
        
        xt = a2(xt)
        np.save(f'amp_pair{pair_idx}_a2.npy', xt.squeeze(0).numpy())
        print(f"  pair{pair_idx} a2: rms={xt.std().item():.4f}")
        
        xt = c2(xt)
        np.save(f'amp_pair{pair_idx}_c2.npy', xt.squeeze(0).numpy())
        print(f"  pair{pair_idx} c2: rms={xt.std().item():.4f}")
        
        x_copy = xt + x_copy
    
    # Save rb0 output
    rb_out = x_copy
    np.save('amp_rb0_out.npy', rb_out.squeeze(0).numpy())
    print(f"  rb0 out: rms={rb_out.std().item():.4f}")
    
    # Also save the conv_pre and upsample for reference
    conv_pre_val = vocoder.decoder.conv_pre(x if x.shape[1]==128 else z)  
    # Actually x is after conv_pre already, let me redo
    print("\nRedoing from conv_pre for clean reference...")
    
print("Done dumping AMP internals")
