# New Model Architecture Parameters (2025-2026)

Research compiled from HuggingFace config.json files and public sources.

---

## 1. Qwen3.6-35B-A3B (Qwen3 MoE)

- **Source**: HuggingFace `Qwen/Qwen3.6-35B-A3B` config.json (retrieved 2026-06-09)
- **Model type**: `qwen3_5_moe` (Qwen3.5 MoE architecture)
- **Architecture**: `Qwen3_5MoeForConditionalGeneration` (multimodal: text + vision)
- **Total params**: ~35B
- **Active params per token**: ~3B

| Parameter | Value |
|---|---|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 40 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 2 |
| `head_dim` | 256 |
| `num_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 512 |
| `shared_expert_intermediate_size` | 512 |
| `vocab_size` | 248,320 |
| `max_position_embeddings` | 262,144 (256K) |
| `rope_theta` | 10,000,000 |
| `partial_rotary_factor` | 0.25 |

**Layer types**: 30 linear_attention + 10 full_attention (every 4th layer is full attention)

**Attention details**:
- Linear attention uses `linear_conv_kernel_dim: 4`, `linear_key_head_dim: 128`
- Full attention interval: 4

**Vision encoder** (ViT): 27 layers, hidden_size 1152, patch_size 16

---

## 2. Gemma 4 (Google)

- **Source**: HuggingFace `google/gemma-4-*` config.json files
- **Common**: `vocab_size: 262144`, head_dim: 256, global_head_dim: 512, sliding+full attention interleaving

### Gemma 4 26B-A4B (MoE) — `google/gemma-4-26B-A4B-it`

| Parameter | Value |
|---|---|
| `hidden_size` | 2816 |
| `num_hidden_layers` | 30 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `num_global_key_value_heads` | 2 |
| `intermediate_size` (shared) | 2112 |
| `moe_intermediate_size` | 704 |
| `num_experts` | 128 |
| `top_k_experts` | 8 |
| `max_position_embeddings` | 262,144 |
| `sliding_window` | 1024 |

~26B total, ~4B active per token. Every layer uses MoE.

### Gemma 4 31B (Dense) — `google/gemma-4-31B-it`

| Parameter | Value |
|---|---|
| `hidden_size` | 5376 |
| `num_hidden_layers` | 60 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 16 |
| `num_global_key_value_heads` | 4 |
| `intermediate_size` | 21,504 |
| `max_position_embeddings` | 262,144 |

Fully dense. Largest Gemma 4.

### Gemma 4 12B (Dense, Unified) — `google/gemma-4-12B-it`

| Parameter | Value |
|---|---|
| `hidden_size` | 3840 |
| `num_hidden_layers` | 48 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `num_global_key_value_heads` | 1 |
| `intermediate_size` | 15,360 |

### Gemma 4 E4B (~4B Dense) — `google/gemma-4-E4B-it`

| Parameter | Value |
|---|---|
| `hidden_size` | 2560 |
| `num_hidden_layers` | 42 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 2 |
| `intermediate_size` | 10,240 |
| `sliding_window` | 512 |

### Gemma 4 E2B (~2B Dense) — `google/gemma-4-E2B-it`

| Parameter | Value |
|---|---|
| `hidden_size` | 1536 |
| `num_hidden_layers` | 35 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 1 |
| `intermediate_size` | 6144 |
| `sliding_window` | 512 |
| `use_double_wide_mlp` | True |

---

## 3. GPT-OSS (OpenAI)

- **Source**: HuggingFace `openai/gpt-oss-20b` and `openai/gpt-oss-120b` config.json
- **Release**: Q1-Q2 2026
- **Model type**: `gpt_oss`, architecture: `GptOssForCausalLM`
- **Key design**: DeepSeek-like MoE with small hidden_size, sliding/full attention interleaving

### GPT-OSS 20B

| Parameter | Value |
|---|---|
| `hidden_size` | 2880 |
| `num_hidden_layers` | 24 |
| `num_attention_heads` | 64 |
| `num_key_value_heads` | 8 |
| `head_dim` | 64 |
| `intermediate_size` | 2880 |
| `num_local_experts` | 32 |
| `experts_per_token` | 4 |
| `vocab_size` | 201,088 |
| `max_position_embeddings` | 131,072 |
| `sliding_window` | 128 |
| `rope_theta` | 150,000 |
| `rope_type` | yarn |
| `attention_bias` | True |
| `swiglu_limit` | 7.0 |

Layer types: 12 sliding + 12 full (strict alternating)

### GPT-OSS 120B

| Parameter | Value |
|---|---|
| `hidden_size` | 2880 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 64 |
| `num_key_value_heads` | 8 |
| `head_dim` | 64 |
| `intermediate_size` | 2880 |
| `num_local_experts` | 128 |
| `experts_per_token` | 4 |
| `vocab_size` | 201,088 |
| `max_position_embeddings` | 131,072 |
| `sliding_window` | 128 |
| `rope_theta` | 150,000 |

Layer types: 18 sliding + 18 full (alternating). Same hidden_size as 20B; only deeper + more experts.

### Architecture notes

- Same hidden_size (2880) for both 20B and 120B — scales via depth and expert count only
- Very small sliding window (128 tokens) with full attention for long-range
- Unique `swiglu_limit: 7.0` caps SwiGLU activations
- YARN RoPE, attention bias, MXFP4 quantization

---

## 4. Llama 4 (Meta)

- **Source**: HuggingFace config.json from public mirrors (`RedHatAI/`), Meta model card, official blog
- **Release**: April 5, 2025
- **Model type**: `llama4`, architecture: `Llama4ForConditionalGeneration`
- **Common**: hidden_size 5120, 48 layers, 40 attn heads, 8 KV heads, head_dim 128, vocab 202,048, rope_theta 500,000, use_qk_norm

### Llama 4 Scout (17B active / 109B total)

| Parameter | Value |
|---|---|
| `hidden_size` | 5120 |
| `num_hidden_layers` | 48 |
| `num_attention_heads` | 40 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `intermediate_size` (shared) | 8192 |
| `intermediate_size_mlp` (expert FFN) | 16,384 |
| `num_local_experts` | 16 |
| `num_experts_per_tok` | 1 |
| `interleave_moe_layer_step` | 1 (every layer MoE) |
| `max_position_embeddings` | 10,485,760 (10M) |
| `attention_chunk_size` | 8192 |

### Llama 4 Maverick (17B active / 400B total)

| Parameter | Value |
|---|---|
| `hidden_size` | 5120 |
| `num_hidden_layers` | 48 |
| `num_attention_heads` | 40 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `intermediate_size` (shared) | 8192 |
| `intermediate_size_mlp` (expert FFN) | 16,384 |
| `num_local_experts` | 128 |
| `num_experts_per_tok` | 1 |
| `interleave_moe_layer_step` | 2 (every other layer MoE) |
| `max_position_embeddings` | 1,048,576 (1M) |

### Llama 4 Behemoth (NOT RELEASED)

- 288B active, 16 experts, ~2T total params
- Used as teacher for Maverick codistillation
- Still training; no config.json available

### Llama Guard 4 12B (Dense)

| Parameter | Value |
|---|---|
| `hidden_size` | 5120 |
| `num_hidden_layers` | 48 |
| `num_attention_heads` | 40 |
| `num_key_value_heads` | 8 |
| `intermediate_size` | 8192 |
| `intermediate_size_mlp` | 8192 |
| `interleave_moe_layer_step` | 0 (dense) |

Same base architecture as Scout/Maverick but fully dense.

---

## Summary Comparison Table

| Model | hidden_size | layers | heads | KV heads | head_dim | FFN size | MoE? | Total | Active |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 2048 | 40 | 16 | 2 | 256 | 512 (moe) | 256x8 | 35B | ~3B |
| Gemma4-26B-A4B | 2816 | 30 | 16 | 8+2g | 256/512 | 704 (moe) | 128x8 | 26B | ~4B |
| Gemma4-31B | 5376 | 60 | 32 | 16+4g | 256/512 | 21504 | No | 31B | 31B |
| Gemma4-12B | 3840 | 48 | 16 | 8+1g | 256/512 | 15360 | No | 12B | 12B |
| Gemma4-E4B | 2560 | 42 | 8 | 2 | 256/512 | 10240 | No | ~4B | ~4B |
| Gemma4-E2B | 1536 | 35 | 8 | 1 | 256/512 | 6144 | No | ~2B | ~2B |
| GPT-OSS-20B | 2880 | 24 | 64 | 8 | 64 | 2880 (moe) | 32x4 | 20B | 20B |
| GPT-OSS-120B | 2880 | 36 | 64 | 8 | 64 | 2880 (moe) | 128x4 | 120B | 120B |
| Llama4-Scout | 5120 | 48 | 40 | 8 | 128 | 16384 (moe) | 16x1 | 109B | 17B |
| Llama4-Maverick | 5120 | 48 | 40 | 8 | 128 | 16384 (moe) | 128x1 | 400B | 17B |
| Llama4-Behemoth | ? | ? | ? | ? | ? | ? | 16 experts | ~2T | 288B |

---

## Key Design Trends

1. **MoE everywhere**: All new flagship models use Mixture of Experts
2. **Small hidden_size + many experts**: GPT-OSS 2880/128exp, Qwen 2048/256exp
3. **Interleaved attention**: Linear/sliding + full attention patterns dominate
4. **Shared + routed experts**: Qwen and Llama 4 both use this pattern
5. **Early fusion multimodal**: Native image understanding in Qwen, Gemma, Llama 4
6. **DeepSeek influence**: GPT-OSS closely follows DeepSeek V2/V3 design

---

*Data retrieved from HuggingFace model configs and official model cards, June 2026.*
