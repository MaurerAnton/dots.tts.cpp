#!/usr/bin/env python3.12
"""Dump intermediate BigVGAN decoder values from Python for C++ comparison."""
import sys, numpy as np, torch
sys.path.insert(0, '/tmp/dots_tts_py')
from dots_tts.runtime import DotsTtsRuntime

# Generate or load test latents
latent_path = sys.argv[1] if len(sys.argv) > 1 else "latents.bin"
if len(sys.argv) > 1:
    data = np.fromfile(latent_path, dtype=np.float32)
    n_frames = len(data) // 128
    latents = data.reshape(n_frames, 128).astype(np.float32)
else:
    n_frames = 16
    np.random.seed(42)
    latents = np.random.randn(n_frames, 128).astype(np.float32) * 0.5
    latents.tofile("latents.bin")

print(f"Latents: {n_frames} frames, RMS={np.sqrt(np.mean(latents**2)):.4f}")

rt = DotsTtsRuntime.from_pretrained('rednote-hilab/dots.tts-soar', precision='float32')
vocoder = rt.model.vocoder
z = torch.from_numpy(latents).float().unsqueeze(0).transpose(1,2)  # [1, 128, T]

# === Step 1: post_proj ===
with torch.no_grad():
    x = vocoder.post_proj(z)  # [1, 128, T]
    np.save("ref_postproj.npy", x.squeeze(0).numpy())  # [128, T]
    print(f"Step 1 post_proj: RMS={x.std().item():.4f}")

    # === Step 2: permute + dec_mi_layer ===
    x = x.permute(0, 2, 1)  # [1, T, 128]
    x = vocoder.dec_mi_layer(x)  # [1, T, 128]
    np.save("ref_milayer.npy", x.squeeze(0).numpy())  # [T, 128]
    print(f"Step 2 dec_mi_layer: RMS={x.std().item():.4f}")

    # === Step 3: permute back ===
    x = x.permute(0, 2, 1)  # [1, 128, T]
    
    # === Step 4: conv_pre ===
    x = vocoder.decoder.conv_pre(x)  # [1, 1536, T]
    np.save("ref_convpre.npy", x.squeeze(0).numpy())  # [1536, T]
    print(f"Step 3 conv_pre: RMS={x.std().item():.4f} shape={list(x.shape)}")

    # === Step 5: Each upsampling stage ===
    for stage in range(6):
        x = vocoder.decoder.ups[stage][0](x)
        np.save(f"ref_stage{stage}_ups.npy", x.squeeze(0).numpy())
        print(f"  Stage {stage} ups: RMS={x.std().item():.4f} shape={list(x.shape)}")

        xs = None
        for j in range(3):  # 3 resblocks per stage
            rb_idx = stage * 3 + j
            rb = vocoder.decoder.resblocks[rb_idx]
            rb_out = rb(x)
            np.save(f"ref_stage{stage}_rb{j}.npy", rb_out.squeeze(0).numpy())
            if xs is None:
                xs = rb_out
            else:
                xs = xs + rb_out
        x = xs / 3.0
        np.save(f"ref_stage{stage}_amp.npy", x.squeeze(0).numpy())
        print(f"  Stage {stage} amp: RMS={x.std().item():.4f}")

    # === Step 6: activation_post ===
    x = vocoder.decoder.activation_post(x)
    np.save("ref_actpost.npy", x.squeeze(0).numpy())
    print(f"Step 4 activation_post: RMS={x.std().item():.4f} shape={list(x.shape)}")

    # === Step 7: conv_post ===
    x = vocoder.decoder.conv_post(x)
    np.save("ref_convpost.npy", x.squeeze(0).numpy())
    print(f"Step 5 conv_post: RMS={x.std().item():.4f} shape={list(x.shape)}")

    # === Step 8: clamp ===
    x = torch.clamp(x, min=-1.0, max=1.0)
    np.save("ref_final.npy", x.squeeze(0).numpy())
    print(f"Final: RMS={x.std().item():.4f} min={x.min().item():.4f} max={x.max().item():.4f}")

print("\nAll reference tensors dumped to ref_*.npy")
