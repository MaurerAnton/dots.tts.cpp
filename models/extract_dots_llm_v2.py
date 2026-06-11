#!/usr/bin/env python3.12
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026  Anton Maurer

"""Extract dots.tts internal Mistral LLM to GGUF."""
import sys, struct, json, os
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

INPUT = '/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/model.safetensors'
OUTPUT = '/home/bym/dots.tts.cpp/models/dots_llm.gguf'

# Map HuggingFace config keys to llama.cpp GGUF keys
MISTRAL_CONFIG = {
    "embedding_length": 1536,
    "block_count": 28,
    "attention.head_count": 12,
    "attention.head_count_kv": 2,
    "feed_forward_length": 8960,
    "context_length": 8192,
    "attention.layer_norm_rms_epsilon": 1e-6,
    "rope.freq_base": 1000000.0,
    "vocab_size": 151672,
    "rope.dimension_count": 128,
    "tie_word_embeddings": True,
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
    if isinstance(v, bool):
        writer.add_uint32(f"llama.{k}", 1 if v else 0)
    elif isinstance(v, int):
        writer.add_uint32(f"llama.{k}", v)
    elif isinstance(v, float):
        writer.add_float32(f"llama.{k}", v)

writer.add_string("tokenizer.ggml.model", "none")  # no vocab needed (we tokenize externally)
writer.add_uint32("tokenizer.ggml.bos_token_id", 1)
writer.add_uint32("tokenizer.ggml.eos_token_id", 2)
# We need tokenizer data — skip for now (just for hidden state extraction)

data_start = 8 + header_len

with open(INPUT, "rb") as f:
    for sf_name, info in sorted(llm_tensors.items()):
        gguf_name = sf_name
        # Special case: llm.model.norm -> output_norm (must do before stripping prefix)
        gguf_name = gguf_name.replace("llm.model.norm", "output_norm")
        # Strip llm.model. prefix for remaining tensors
        gguf_name = gguf_name.replace("llm.model.", "")
        gguf_name = gguf_name.replace("embed_tokens", "token_embd")
        gguf_name = gguf_name.replace("lm_head", "output")
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
        
        # GGUF library reverses dimensions: numpy (out,in) -> GGUF (in,out)
        # No manual transpose needed!
        
        writer.add_tensor(gguf_name, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
print(f"Wrote {OUTPUT} ({size_mb:.1f} MB)")
