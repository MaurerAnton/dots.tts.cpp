#!/usr/bin/env python3.12
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026  Anton Maurer

"""Extract Qwen2.5-1.5B LLM weights from dots.tts safetensors -> GGUF.
Usage: python3.12 extract_llm_gguf.py [--input model.safetensors] [--output llm.gguf]
"""
import sys, struct, json, os, argparse
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

QWEN2_CONFIG = {
    "architecture": "qwen2",
    "hidden_size": 1536,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "intermediate_size": 8960,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "vocab_size": 151672,
    "tie_word_embeddings": True,
    "attention_bias": False,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-base/blobs/69dbad797566b24003506e1dd698597937149920f6df9782d84214bf477acb48")
    parser.add_argument("--output", default="llm_qwen25_1.5b.gguf")
    parser.add_argument("--out-type", default="f16", choices=["f32", "f16"])
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}")
        sys.exit(1)

    print(f"Reading safetensors header from {args.input}...")
    with open(args.input, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))

    llm_tensors = {}
    total_bytes = 0
    for name, info in header.items():
        if name.startswith("llm.model.") and not name.startswith("__"):
            shape = info["shape"]
            dtype = info["dtype"]
            offsets = info["data_offsets"]
            llm_tensors[name] = {
                "shape": shape,
                "dtype": dtype,
                "start": offsets[0],
                "end": offsets[1],
            }
            total_bytes += offsets[1] - offsets[0]

    print(f"Found {len(llm_tensors)} LLM tensors, {total_bytes/1e9:.2f} GB")

    # Write GGUF
    writer = GGUFWriter(args.output, "qwen2")

    # Architecture metadata
    for k, v in QWEN2_CONFIG.items():
        if isinstance(v, int):
            writer.add_uint32(f"qwen2.{k}", v)
        elif isinstance(v, float):
            writer.add_float32(f"qwen2.{k}", v)
        elif isinstance(v, bool):
            writer.add_bool(f"qwen2.{k}", v)
    writer.add_uint32("general.architecture", 0)  # placeholder, will be set by add_architecture
    writer.add_string("tokenizer.ggml.model", "gpt2")

    out_type = GGMLQuantizationType.F16 if args.out_type == "f16" else GGMLQuantizationType.F32

    # Map tensor names: llm.model.* -> blk.* (llama.cpp convention)
    # Qwen2 uses the same naming as LLaMA in llama.cpp GGUF
    data_start = 8 + header_len

    with open(args.input, "rb") as f:
        for sf_name, info in sorted(llm_tensors.items()):
            # Map name: llm.model.embed_tokens.weight -> token_embd.weight
            # llm.model.layers.{i}.self_attn.q_proj.weight -> blk.{i}.attn_q.weight
            # etc.
            gguf_name = sf_name
            gguf_name = gguf_name.replace("llm.model.", "")
            gguf_name = gguf_name.replace("embed_tokens", "token_embd")
            gguf_name = gguf_name.replace("lm_head", "output")
            gguf_name = gguf_name.replace("model.norm", "output_norm")
            gguf_name = gguf_name.replace("layers.", "blk.")
            gguf_name = gguf_name.replace("self_attn.q_proj", "attn_q")
            gguf_name = gguf_name.replace("self_attn.k_proj", "attn_k")
            gguf_name = gguf_name.replace("self_attn.v_proj", "attn_v")
            gguf_name = gguf_name.replace("self_attn.o_proj", "attn_o")
            gguf_name = gguf_name.replace("mlp.gate_proj", "ffn_gate")
            gguf_name = gguf_name.replace("mlp.up_proj", "ffn_up")
            gguf_name = gguf_name.replace("mlp.down_proj", "ffn_down")
            gguf_name = gguf_name.replace("input_layernorm", "attn_norm")
            gguf_name = gguf_name.replace("post_attention_layernorm", "ffn_norm")

            # Read tensor data
            f.seek(data_start + info["start"])
            raw = f.read(info["end"] - info["start"])

            shape = info["shape"]
            arr = np.frombuffer(raw, dtype=np.float16 if info["dtype"] == "F16" else np.float32)
            arr = arr.reshape(shape)

            # Transpose 2D weights for ggml: [out, in] -> [in, out]
            if len(shape) == 2 and "weight" in sf_name:
                arr = arr.T.copy()

            if args.out_type == "f16" and arr.dtype == np.float32:
                arr = arr.astype(np.float16)

            writer.add_tensor(gguf_name, arr, raw_dtype=out_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote {args.output} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
