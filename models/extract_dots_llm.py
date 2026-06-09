#!/usr/bin/env python3.12
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026  Anton Maurer

"""Extract dots.tts internal Mistral LLM to GGUF."""
import sys, struct, json, os
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

INPUT = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/model.safetensors'
OUTPUT = '/home/bym/dots.tts.cpp/models/dots_llm.gguf'

MISTRAL_CONFIG = {
    "hidden_size": 1536,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "intermediate_size": 8960,
    "max_position_embeddings": 8192,
    "context_length": 8192,
    "rms_norm_eps": 1e-5,
    "rope_theta": 1000000.0,
    "vocab_size": 151672,
    "tie_word_embeddings": False,
    "head_dim": 128,
}

print(f"Reading {INPUT}...")
with open(INPUT, "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_len))

llm_tensors = {}
for name, info in sorted(header.items()):
    if name.startswith("llm.model.") and not name.startswith("__"):
        llm_tensors[name] = info

print(f"Found {len(llm_tensors)} LLM tensors")

# Write GGUF as "llama" architecture (Mistral)
writer = GGUFWriter(OUTPUT, "llama")

for k, v in MISTRAL_CONFIG.items():
    if k == "max_position_embeddings": continue  # not needed by llama.cpp
    if isinstance(v, int):
        writer.add_uint32(f"llama.{k}", v)
    elif isinstance(v, float):
        writer.add_float32(f"llama.{k}", v)
    elif isinstance(v, bool):
        writer.add_bool(f"llama.{k}", v)

writer.add_uint32("llama.attention.layer_norm_rms_epsilon", int(1e-5))
writer.add_string("tokenizer.ggml.model", "gpt2")
writer.add_uint32("tokenizer.ggml.bos_token_id", 1)
writer.add_uint32("tokenizer.ggml.eos_token_id", 2)
# We need tokenizer data — skip for now (just for hidden state extraction)
writer.add_string("general.architecture", "llama")

data_start = 8 + header_len

with open(INPUT, "rb") as f:
    for sf_name, info in sorted(llm_tensors.items()):
        gguf_name = sf_name.replace("llm.model.", "")
        gguf_name = gguf_name.replace("embed_tokens", "token_embd")
        gguf_name = gguf_name.replace("lm_head", "output")
        gguf_name = gguf_name.replace("model.norm", "output_norm")
        gguf_name = gguf_name.replace("layers.", "blk.")
        gguf_name = gguf_name.replace("self_attn.q_proj", "attn_q")
        gguf_name = gguf_name.replace("self_attn.k_proj", "attn_k")
        gguf_name = gguf_name.replace("self_attn.v_proj", "attn_v")
        gguf_name = gguf_name.replace("self_attn.o_proj", "attn_output")
        gguf_name = gguf_name.replace("mlp.gate_proj", "ffn_gate")
        gguf_name = gguf_name.replace("mlp.up_proj", "ffn_up")
        gguf_name = gguf_name.replace("mlp.down_proj", "ffn_down")
        gguf_name = gguf_name.replace("input_layernorm", "attn_norm")
        gguf_name = gguf_name.replace("post_attention_layernorm", "ffn_norm")

        f.seek(data_start + info["data_offsets"][0])
        raw = f.read(info["data_offsets"][1] - info["data_offsets"][0])
        
        shape = info["shape"]
        dt = info["dtype"]
        arr = np.frombuffer(raw, dtype=np.float16 if dt == "F16" else np.float32).reshape(shape)
        
        # Transpose 2D weights: [out, in] -> [in, out] for ggml
        if len(shape) == 2 and "weight" in sf_name:
            arr = arr.T.copy()
        
        writer.add_tensor(gguf_name, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
print(f"Wrote {OUTPUT} ({size_mb:.1f} MB)")
