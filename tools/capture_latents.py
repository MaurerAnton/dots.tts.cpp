#!/usr/bin/env python3.12
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026  Anton Maurer
"""Capture Python LM hidden_proj + noise for C++ --pyctx mode.
Generates py_call%d_fmseq.bin and py_noise_call%d.bin files.
Usage: python3 tools/capture_latents.py --text "hello world"
"""
import sys, torch, numpy as np, math, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default='hello world')
    parser.add_argument('--model', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model_dir = args.model or '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35'
    
    sys.path.insert(0, '/home/bym/llama/dots.tts/src')
    from dots_tts.runtime import DotsTtsRuntime
    import dots_tts.models.dots_tts.model as model_mod

    torch.manual_seed(args.seed)

    call_fmseq = []
    call_noise = []

    # Patch _decode_next_audio to capture fm_sequence
    orig_decode = model_mod.DotsTtsModel._decode_next_audio
    def patched_decode(self, state, **kw):
        seq = state.fm_sequence[0, :state.fm_seq_len].cpu().numpy().astype(np.float32)
        call_fmseq.append(seq)
        return orig_decode(self, state, **kw)
    model_mod.DotsTtsModel._decode_next_audio = patched_decode

    # Patch _flow_matching_step_fm to capture noise
    orig_fm = model_mod.DotsTtsModel._flow_matching_step_fm
    def patched_fm(self, *, patch_size, **kw):
        noise = torch.randn(1, patch_size, self.latent_dim)
        call_noise.append(noise.squeeze().cpu().numpy().astype(np.float32))
        return orig_fm(self, patch_size=patch_size, **kw)
    model_mod.DotsTtsModel._flow_matching_step_fm = patched_fm

    print(f"Loading model...")
    rt = DotsTtsRuntime.from_pretrained(model_dir, precision='float32')
    
    print(f"Generating '{args.text}' (seed={args.seed})...")
    result = rt.generate(text=args.text, speaker_scale=1.0, guidance_scale=1.0)
    
    print(f"Generated {len(call_fmseq)} audio calls")
    for i, (fs, ns) in enumerate(zip(call_fmseq, call_noise)):
        fs.tofile(f'py_call{i}_fmseq.bin')
        ns.tofile(f'py_noise_call{i}.bin')
        print(f"  Call {i}: fmseq={fs.shape}, noise RMS={math.sqrt(np.mean(ns**2)):.4f}")
    
    # Save reference audio
    audio = result['audio'].squeeze().cpu().numpy().astype(np.float32)
    audio.tofile('py_reference_audio.bin')
    print(f"Saved py_reference_audio.bin: {len(audio)} samples")

if __name__ == '__main__':
    main()
