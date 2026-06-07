#!/usr/bin/env python3.12
"""dots.tts guaranteed audio generation using model.generate()"""
import sys, os, numpy as np, torch
sys.path.insert(0, '/tmp/dots_tts_py')

def generate(text, output="output.wav"):
    print(f"Generating: '{text}'")
    
    from dots_tts.models.dots_tts.model import DotsTtsModel
    from omegaconf import OmegaConf
    from huggingface_hub import hf_hub_download
    import safetensors.torch
    
    # Load model
    ckpt = hf_hub_download('rednote-hilab/dots.tts-soar', 'model.safetensors')
    state = safetensors.torch.load_file(ckpt)
    
    with open('/tmp/dots_tts_py/configs/dots_tts.yaml') as f:
        cfg = OmegaConf.create(f.read())
    
    model = DotsTtsModel(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()
    print("  Model loaded")
    
    with torch.no_grad():
        audio, info = model.generate(text, max_length=500)
    
    audio_np = audio.squeeze().cpu().numpy()
    sr = info.get('sample_rate', 48000)
    
    import soundfile as sf
    sf.write(output, audio_np, sr)
    
    print(f"  Wrote {output}: {len(audio_np)} samples, {len(audio_np)/sr:.1f}s @ {sr}Hz")
    print(f"  Min: {audio_np.min():.4f}, Max: {audio_np.max():.4f}, RMS: {np.sqrt(np.mean(audio_np**2)):.4f}")
    return output

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello world"
    generate(text, sys.argv[2] if len(sys.argv) > 2 else "output.wav")
