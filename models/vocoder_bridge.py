#!/usr/bin/env python3.12
"""Decode C++ DiT latents with real BigVGAN vocoder using full dots.tts runtime."""
import sys, os, numpy as np, torch
from huggingface_hub import hf_hub_download
import safetensors.torch

def main():
    latent_path = sys.argv[1] if len(sys.argv) > 1 else "latents.bin"
    wav_path = sys.argv[2] if len(sys.argv) > 2 else "output_real.wav"

    data = np.fromfile(latent_path, dtype=np.float32)
    n_frames = len(data) // 128
    latents = data.reshape(n_frames, 128)
    print(f"Latents: {n_frames} frames, rms={np.sqrt(np.mean(latents**2)):.4f}")

    # Load just the vocoder part from full checkpoint
    print("Loading vocoder weights...")
    ckpt = hf_hub_download('rednote-hilab/dots.tts-base', 'vocoder.safetensors')
    
    # Load ALL tensors from vocoder.safetensors (they're all vocoder-related)
    state = {}
    with safetensors.safe_open(ckpt, framework='pt') as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
    print(f"Extracted {len(state)} vocoder tensors")

    # Load AudioVAE
    from dots_tts.modules.vocoder.bigvgan import AudioVAE
    from dots_tts.modules.vocoder.config import AudioVAEConfig
    cfg = AudioVAEConfig(
        sample_rate=48000, latent_dim=128, use_tanh_at_final=False,
        upsample_rates=[10,6,4,2,2,2], upsample_kernel_sizes=[20,12,8,4,4,4],
        upsample_initial_channel=1536, resblock="1",
        resblock_kernel_sizes=[3,7,11], resblock_dilation_sizes=[[1,3,5]]*3,
        downsample_rates=[2,2,2,4,6,10],
        downsample_channels=[12,24,48,96,192,384,768],
        causal=True, causal_encoder=True, mi_num_layers=4,
        activation="snakebeta", snake_logscale=True, use_bias_at_final=False,
    )
    model = AudioVAE(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.remove_weight_norm()  # Required for inference
    print("AudioVAE loaded")

    # Decode: the model expects z in [B, latent_dim, T] format
    # Our latents are already VAE latents, so feed directly through pre_proj -> dec_mi -> decoder
    z = torch.from_numpy(latents).float().unsqueeze(0).transpose(1,2)  # [1, 128, N]
    
    with torch.no_grad():
        # post_proj (Conv1d 128->128, kernel=1)
        x = model.post_proj(z)
        # dec_mi_layer: permute, Linear->SLSTM->Linear, permute back
        x = model.dec_mi_layer(x.transpose(1,2)).transpose(1,2)
        # decoder (BigVGAN)
        audio = model.decoder(x)
    
    audio_np = audio.squeeze().cpu().numpy()
    print(f"Audio: {len(audio_np)} samples, min={audio_np.min():.4f}, max={audio_np.max():.4f}")

    import soundfile as sf
    sf.write(wav_path, audio_np, 48000)
    print(f"Saved {wav_path} ({len(audio_np)/48000:.2f}s @ 48kHz)")

if __name__ == "__main__":
    main()
