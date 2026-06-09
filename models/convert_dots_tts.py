#!/usr/bin/env python3.12
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026  Anton Maurer

"""Convert dots.tts safetensors to GGUF format for dots.tts.cpp.

Usage: python3.12 convert_dots_tts.py --out-model dots_tts_f16.gguf

Downloads model.safetensors from HuggingFace and writes a GGUF file
containing DiT + PatchEncoder weights in the format expected by dots.tts.cpp.
"""

import sys, os, struct, json, argparse
import numpy as np
from huggingface_hub import hf_hub_download
from gguf import GGUFWriter, GGMLQuantizationType, gguf_writer

# ---------------------------------------------------------------------------
# Architecture constants (must match dots_tts.h)
# ---------------------------------------------------------------------------
DIT_NUM_LAYERS = 18
DIT_HIDDEN_SIZE = 1024
DIT_NUM_HEADS = 16
DIT_HEAD_SIZE = 64
DIT_FFN_SIZE = 4096
DIT_ADA_DIM = 1024
DIT_SPEAKER_DIM = 512

PATCHENC_NUM_LAYERS = 24
PATCHENC_HIDDEN = 1024
PATCHENC_NUM_HEADS = 16
PATCHENC_HEAD_SIZE = 64
PATCHENC_FFN_SIZE = 4096
PATCHENC_LATENT_DIM = 128
PATCHENC_PATCH_SIZE = 4

VAE_LATENT_DIM = 128
VAE_HOP_SAMPLES = 960
VAE_SAMPLE_RATE = 48000

# ---------------------------------------------------------------------------
# Tensor name mapping: safetensors -> GGUF (C++ naming)
# ---------------------------------------------------------------------------

def map_tensor_name(sf_name):
    """Map safetensors tensor name to GGUF name expected by dots.tts.cpp."""

    # PatchEncoder: patch_encoder.encoder.layers.{i}.*
    if sf_name.startswith("patch_encoder.encoder.layers."):
        parts = sf_name.split(".")
        layer_idx = int(parts[3])
        sub = ".".join(parts[4:])

        # Attention projections
        if sub == "attn.q_proj.weight":
            return f"pe.layers.{layer_idx}.attn.q.weight"
        if sub == "attn.k_proj.weight":
            return f"pe.layers.{layer_idx}.attn.k.weight"
        if sub == "attn.v_proj.weight":
            return f"pe.layers.{layer_idx}.attn.v.weight"
        if sub == "attn.o_proj.weight":
            return f"pe.layers.{layer_idx}.attn.o.weight"
        if sub == "attn.o_proj.bias":
            return f"pe.layers.{layer_idx}.attn.o.bias"

        # Norms
        if sub == "attn_norm.weight":
            return f"pe.layers.{layer_idx}.attn_norm.weight"
        if sub == "ffn_norm.weight":
            return f"pe.layers.{layer_idx}.ffn_norm.weight"

        # FFN
        if sub == "ffn.fc1.weight":
            return f"pe.layers.{layer_idx}.ffn.w1.weight"
        if sub == "ffn.fc1.bias":
            return f"pe.layers.{layer_idx}.ffn.w1.bias"
        if sub == "ffn.fc2.weight":
            return f"pe.layers.{layer_idx}.ffn.w2.weight"
        if sub == "ffn.fc2.bias":
            return f"pe.layers.{layer_idx}.ffn.w2.bias"

    # PatchEncoder projections
    if sf_name == "patch_encoder.ds_proj.weight":
        return "pe.conv.weight"
    if sf_name == "patch_encoder.ds_proj.bias":
        return "pe.conv.bias"
    if sf_name == "patch_encoder.in_proj.weight":
        return "pe.in_proj.weight"
    if sf_name == "patch_encoder.in_proj.bias":
        return "pe.in_proj.bias"
    if sf_name == "patch_encoder.out_proj.weight":
        return "pe.out_proj.weight"
    if sf_name == "patch_encoder.out_proj.bias":
        return "pe.out_proj.bias"

    # DiT (velocity_field_predictor)
    if sf_name.startswith("velocity_field_predictor.blocks."):
        parts = sf_name.split(".")
        block_idx = int(parts[2])
        sub = ".".join(parts[3:])

        # adaLN
        if sub == "adaLN_modulation.1.weight":
            return f"dit.layers.{block_idx}.adaln.linear.weight"
        if sub == "adaLN_modulation.1.bias":
            return f"dit.layers.{block_idx}.adaln.linear.bias"

        # Attention
        if sub == "attn.q_proj.weight":
            return f"dit.layers.{block_idx}.attn.q.weight"
        if sub == "attn.k_proj.weight":
            return f"dit.layers.{block_idx}.attn.k.weight"
        if sub == "attn.v_proj.weight":
            return f"dit.layers.{block_idx}.attn.v.weight"
        if sub == "attn.o_proj.weight":
            return f"dit.layers.{block_idx}.attn.o.weight"
        if sub == "attn.o_proj.bias":
            return f"dit.layers.{block_idx}.attn.o.bias"
        if sub == "attn.q_norm.weight":
            return f"dit.layers.{block_idx}.q_norm.weight"
        if sub == "attn.k_norm.weight":
            return f"dit.layers.{block_idx}.k_norm.weight"

        # FFN
        if sub == "ffn.fc1.weight":
            return f"dit.layers.{block_idx}.ffn.w1.weight"
        if sub == "ffn.fc1.bias":
            return f"dit.layers.{block_idx}.ffn.w1.bias"
        if sub == "ffn.fc2.weight":
            return f"dit.layers.{block_idx}.ffn.w2.weight"
        if sub == "ffn.fc2.bias":
            return f"dit.layers.{block_idx}.ffn.w2.bias"

        # Norms
        if sub == "attn.norm1.weight":
            return f"dit.layers.{block_idx}.attn_norm.weight"
        if sub == "ffn.norm2.weight":
            return f"dit.layers.{block_idx}.ffn_norm.weight"

    # DiT time embedder
    if sf_name == "velocity_field_predictor.time_embedder.mlp.0.weight":
        return "dit.t_embed.w1.weight"
    if sf_name == "velocity_field_predictor.time_embedder.mlp.0.bias":
        return "dit.t_embed.w1.bias"
    if sf_name == "velocity_field_predictor.time_embedder.mlp.2.weight":
        return "dit.t_embed.w2.weight"
    if sf_name == "velocity_field_predictor.time_embedder.mlp.2.bias":
        return "dit.t_embed.w2.bias"

    # DiT input/output layers
    if sf_name == "velocity_field_predictor.input_layer.weight":
        return "dit.input_layer.weight"
    if sf_name == "velocity_field_predictor.input_layer.bias":
        return "dit.input_layer.bias"
    if sf_name == "velocity_field_predictor.output_layer.linear.weight":
        return "dit.out_proj.weight"
    if sf_name == "velocity_field_predictor.output_layer.linear.bias":
        return "dit.out_proj.bias"

    # DiT speaker projection
    if sf_name == "xvec_proj.0.weight":
        return "dit.spk_proj.w1.weight"
    if sf_name == "xvec_proj.0.bias":
        return "dit.spk_proj.w1.bias"
    if sf_name == "xvec_proj.1.weight":
        return "dit.spk_proj.w2.weight"
    if sf_name == "xvec_proj.1.bias":
        return "dit.spk_proj.w2.bias"

    # DiT input projections (hidden_proj, latent_proj, coordinate_proj)
    if sf_name == "hidden_proj.weight":
        return "dit.hidden_proj.weight"
    if sf_name == "hidden_proj.bias":
        return "dit.hidden_proj.bias"
    if sf_name == "latent_proj.weight":
        return "dit.latent_proj.weight"
    if sf_name == "latent_proj.bias":
        return "dit.latent_proj.bias"
    if sf_name == "coordinate_proj.weight":
        return "dit.coord_proj.weight"
    if sf_name == "coordinate_proj.bias":
        return "dit.coord_proj.bias"

    # Skip LLM tensors (handled by llama.cpp)
    if sf_name.startswith("llm."):
        return None

    # Skip vocoder (separate file)
    if sf_name.startswith("vocoder."):
        return None

    print(f"  [skip] {sf_name}")
    return None


# ---------------------------------------------------------------------------
# Weight transposition: PyTorch -> GGML convention
# ---------------------------------------------------------------------------
# PyTorch Linear weights: [out_features, in_features]
# GGML mul_mat convention: weight stored as [in_features, out_features]
# So we TRANSPOSE all 2D linear weights.
#
# Conv1d weights stay as-is: [out_ch, in_ch, kernel]
# 1D biases stay as-is
# Norm weights stay as-is

def transpose_for_ggml(name, arr):
    """Transpose weight tensor from PyTorch [out,in] to GGML [in,out] convention."""
    # Linear weights: [out, in] -> [in, out]
    if name.endswith(".weight") and len(arr.shape) == 2:
        # PyTorch Linear: weight shape = [out_features, in_features]
        # GGML expects: [in_features, out_features]
        print(f"  transpose {name}: {list(arr.shape)} -> ", end="")
        arr = np.ascontiguousarray(arr.T)
        print(f"{list(arr.shape)}")
        return arr
    return arr


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert dots.tts to GGUF")
    parser.add_argument("--out-model", default="dots_tts_f16.gguf", help="Output GGUF file")
    parser.add_argument("--out-type", default="f16", choices=["f32", "f16"], help="Output dtype")
    parser.add_argument("--model-repo", default="rednote-hilab/dots.tts-base", help="HF repo")
    args = parser.parse_args()

    print(f"Converting {args.model_repo} -> {args.out_model}")

    # Download safetensors
    print("Downloading model.safetensors...")
    sf_path = hf_hub_download(args.model_repo, "model.safetensors")

    print(f"Loading {sf_path}...")
    import safetensors.torch
    tensors = safetensors.torch.load_file(sf_path)

    # Map and transpose weights
    gguf_weights = {}
    skipped = 0
    for sf_name, tensor in tensors.items():
        gguf_name = map_tensor_name(sf_name)
        if gguf_name is None:
            skipped += 1
            continue
        arr = tensor.numpy()
        arr = transpose_for_ggml(gguf_name, arr)
        gguf_weights[gguf_name] = arr

    print(f"Mapped {len(gguf_weights)} tensors, skipped {skipped}")

    # Write GGUF
    print(f"Writing {args.out_model}...")
    writer = GGUFWriter(args.out_model, "dots_tts")

    # Architecture metadata
    writer.add_architecture()
    writer.add_uint32("dots_tts.dit.num_layers", DIT_NUM_LAYERS)
    writer.add_uint32("dots_tts.dit.hidden_size", DIT_HIDDEN_SIZE)
    writer.add_uint32("dots_tts.dit.num_heads", DIT_NUM_HEADS)
    writer.add_uint32("dots_tts.dit.head_dim", DIT_HEAD_SIZE)
    writer.add_uint32("dots_tts.dit.ffn_size", DIT_FFN_SIZE)
    writer.add_uint32("dots_tts.patchenc.num_layers", PATCHENC_NUM_LAYERS)
    writer.add_uint32("dots_tts.patchenc.hidden_size", PATCHENC_HIDDEN)
    writer.add_uint32("dots_tts.vae.latent_dim", VAE_LATENT_DIM)
    writer.add_uint32("dots_tts.vae.hop_samples", VAE_HOP_SAMPLES)
    writer.add_uint32("dots_tts.vae.sample_rate", VAE_SAMPLE_RATE)

    dtype_map = {"f32": GGMLQuantizationType.F32, "f16": GGMLQuantizationType.F16}
    out_type = dtype_map[args.out_type]

    for name, arr in sorted(gguf_weights.items()):
        if args.out_type == "f16" and arr.dtype == np.float32:
            arr = arr.astype(np.float16)
        writer.add_tensor(name, arr, raw_dtype=out_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Print size
    size_mb = os.path.getsize(args.out_model) / (1024 * 1024)
    print(f"Done: {args.out_model} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
