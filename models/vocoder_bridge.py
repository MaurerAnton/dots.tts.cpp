#!/usr/bin/env python3.12
"""BigVGAN bridge using full dots.tts runtime (guaranteed correct)."""
import sys, os, numpy as np, torch
sys.path.insert(0, '/tmp/dots_tts_py')
from dots_tts.runtime import DotsTtsRuntime

def main():
    latent_path = sys.argv[1] if len(sys.argv) > 1 else "latents.bin"
    wav_path = sys.argv[2] if len(sys.argv) > 2 else "output.wav"
    
    data = np.fromfile(latent_path, dtype=np.float32)
    n_frames = len(data) // 128
    latents = data.reshape(n_frames, 128)
    print(f"Latents: {n_frames} frames, RMS={np.sqrt(np.mean(latents**2)):.4f}")
    
    # Load via full runtime (guaranteed correct weights)
    rt = DotsTtsRuntime.from_pretrained('rednote-hilab/dots.tts-soar', precision='float32')
    
    # Decode using the SAME function as the CLI
    z = torch.from_numpy(latents).float().unsqueeze(0).transpose(1,2)
    with torch.no_grad():
        audio = rt.model.vocoder.inference_from_latents(z, do_sample=False)
    audio_np = audio.squeeze().cpu().numpy()
    
    print(f"Audio: {len(audio_np)} samples, min={audio_np.min():.4f}, max={audio_np.max():.4f}")
    print(f"Mean={audio_np.mean():.4f} ZC={np.sum(np.abs(np.diff(np.signbit(audio_np))))/len(audio_np):.4f}")
    
    import soundfile as sf
    sf.write(wav_path, audio_np, 48000)
    print(f"Saved {wav_path}")

if __name__ == "__main__":
    main()
