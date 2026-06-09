# Large Model Architecture Parameters — Hidden Size Comparison

**Date**: 2026-06-09
**Methodology**: Configs pulled from HuggingFace model repos (config.json) where available. GPT-4 and Claude values are from public analysis/rumors — not officially confirmed.

---

## Key Finding: The hidden_size Ceiling

| hidden_size | Models |
|---|---|
| **16384** | Llama-3.1-405B (dense) |
| **8192** | Qwen2.5-72B (dense) |
| **7168** | DeepSeek-V3/R1, DeepSeek-V4-Pro, Kimi-K2.6, Kimi-K2-Instruct, Kimi-K2.5 |
| **6144** | GLM-5, GLM-5.1 |
| **4096** | DeepSeek-V4-Flash |

**Llama-3.1-405B at 16384 is the highest confirmed hidden_size among open-weight models.** Among MoE models, 7168 is the current ceiling. GPT-4 is rumored to use 16384–20480 but unconfirmed. No open model exceeds 16384.

---

## 1. DeepSeek Models

### DeepSeek-V3 / DeepSeek-R1
- **Source**: HuggingFace config.json (`deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-R1`)
- **Architecture**: DeepseekV3ForCausalLM (MoE)
- **Total params**: 671B (37B active per token)
- **hidden_size**: 7168
- **num_hidden_layers**: 61
- **intermediate_size** (dense FFN): 18432
- **num_attention_heads**: 128
- **num_key_value_heads**: 128 (MLA with KV compression)
- **kv_lora_rank**: 512
- **q_lora_rank**: 1536
- **head_dim** (v_head_dim): 128
- **vocab_size**: 129280
- **max_position_embeddings**: 163840 (YaRN rope)
- **MoE config**:
  - n_routed_experts: 256
  - n_shared_experts: 1
  - num_experts_per_tok: 8
  - moe_intermediate_size: 2048
  - first_k_dense_replace: 3 (first 3 layers are dense)
  - moe_layer_freq: 1 (every layer after first 3 is MoE)
  - scoring_func: sigmoid
  - routed_scaling_factor: 2.5
- **Note**: DeepSeek-R1 is a reasoning fine-tune of V3-Base; architecture is identical.

### DeepSeek-V4-Pro
- **Source**: HuggingFace config.json (`deepseek-ai/DeepSeek-V4-Pro`)
- **Architecture**: DeepseekV4ForCausalLM (MoE)
- **hidden_size**: 7168
- **num_hidden_layers**: 61
- **num_attention_heads**: 128
- **num_key_value_heads**: 1 (extreme KV compression in V4)
- **head_dim**: 512
- **q_lora_rank**: 1536
- **o_lora_rank**: 1024
- **o_groups**: 16
- **vocab_size**: 129280
- **max_position_embeddings**: 1048576 (1M tokens, YaRN)
- **MoE config**:
  - n_routed_experts: 384 (up from 256 in V3)
  - n_shared_experts: 1
  - num_experts_per_tok: 6 (down from 8 in V3)
  - moe_intermediate_size: 3072 (up from 2048 in V3)
  - scoring_func: sqrtsoftplus
  - routed_scaling_factor: 2.5
- **Key changes from V3**:
  - Increased experts (256→384), reduced active/token (8→6), larger per-expert FFN (2048→3072)
  - Extreme KV compression: 1 KV head (vs 128 in V3)
  - MLA output also compressed (o_lora_rank)
  - Hash-based expert routing (num_hash_layers: 3, index_head_dim: 128)
  - Sliding window attention (128)
  - Context compression (compress_rope_theta, compress_ratios)
  - 1M context window
  - No dense intermediate_size — all layers use MoE FFN

### DeepSeek-V4-Flash
- **Source**: HuggingFace config.json (`deepseek-ai/DeepSeek-V4-Flash`)
- **Architecture**: DeepseekV4ForCausalLM (MoE)
- **hidden_size**: 4096
- **num_hidden_layers**: 43
- **num_attention_heads**: 64
- **num_key_value_heads**: 1
- **head_dim**: 512
- **q_lora_rank**: 1024
- **vocab_size**: 129280
- **max_position_embeddings**: 1048576
- **MoE config**:
  - n_routed_experts: 256
  - n_shared_experts: 1
  - num_experts_per_tok: 6
  - moe_intermediate_size: 2048

---

## 2. GLM-5 / GLM-5.1 (Zhipu AI / THUDM)

### GLM-5
- **Source**: HuggingFace config.json (`zai-org/GLM-5`)
- **Architecture**: GlmMoeDsaForCausalLM (MoE with Dynamic Sparse Attention)
- **hidden_size**: 6144
- **num_hidden_layers**: 78
- **intermediate_size** (dense FFN): 12288
- **num_attention_heads**: 64
- **num_key_value_heads**: 64
- **head_dim**: 64
- **kv_lora_rank**: 512 (MLA-style KV compression)
- **q_lora_rank**: 2048
- **vocab_size**: 154880
- **max_position_embeddings**: 202752
- **MoE config**:
  - n_routed_experts: 256
  - n_shared_experts: 1
  - num_experts_per_tok: 8
  - moe_intermediate_size: 2048
  - first_k_dense_replace: 3
  - moe_layer_freq: 1

### GLM-5.1
- **Source**: HuggingFace config.json (`zai-org/GLM-5.1`)
- **Architecture**: Identical to GLM-5 (same config)
- Presumably an improved fine-tune or training iteration.

---

## 3. Kimi K2 Series (Moonshot AI)

### Kimi-K2-Instruct (Kimi K2 base)
- **Source**: HuggingFace config.json (`moonshotai/Kimi-K2-Instruct-0905`)
- **Architecture**: DeepseekV3ForCausalLM (derived from DeepSeek-V3 architecture)
- **hidden_size**: 7168
- **num_hidden_layers**: 61
- **intermediate_size** (dense FFN): 18432
- **num_attention_heads**: 64
- **num_key_value_heads**: 64
- **kv_lora_rank**: 512
- **q_lora_rank**: 1536
- **vocab_size**: 163840
- **max_position_embeddings**: 262144
- **MoE config**:
  - n_routed_experts: 384
  - n_shared_experts: 1
  - num_experts_per_tok: 8
  - moe_intermediate_size: 2048
  - first_k_dense_replace: 1 (only first layer is dense)
  - scoring_func: sigmoid
  - routed_scaling_factor: 2.827

### Kimi-K2.5 (multimodal)
- **Source**: HuggingFace config.json (`moonshotai/Kimi-K2.5`)
- **Architecture**: KimiK25ForConditionalGeneration (multimodal, DeepseekV3-based text backbone)
- **Text backbone** (text_config):
  - hidden_size: 7168
  - num_hidden_layers: 61
  - intermediate_size: 18432
  - num_attention_heads: 64
  - num_key_value_heads: 64
  - kv_lora_rank: 512
  - q_lora_rank: 1536
  - vocab_size: 163840
  - max_position_embeddings: 262144
  - MoE: 384 experts, 8 active, moe_intermediate_size=2048
- **Vision config**: mm_hidden_size=1152, VT hidden=1152, VT intermediate=4304, VT layers=27

### Kimi-K2.6 (multimodal, latest)
- **Source**: HuggingFace config.json (`moonshotai/Kimi-K2.6`)
- **Architecture**: KimiK25ForConditionalGeneration (multimodal)
- **Text backbone** (text_config): Identical to K2.5 text config
  - hidden_size: 7168
  - num_hidden_layers: 61
  - intermediate_size: 18432
  - num_attention_heads: 64
  - num_key_value_heads: 64
  - MoE: 384 experts, 8 active, moe_intermediate_size=2048
  - vocab_size: 163840
  - max_position_embeddings: 262144
- **Vision config**: mm_hidden_size=1152, VT hidden=1152, VT intermediate=4304, VT layers=27
- **Key difference from K2.5**: 4-bit quantized weights (compressed-tensors format)

---

## 4. Comparison Models

### Llama-3.1-405B (Meta)
- **Source**: Meta Llama 3.1 paper (gated on HF, config verified from public documentation)
- **Architecture**: Dense transformer (no MoE)
- **Total params**: 405B
- **hidden_size**: 16384  ← **LARGEST confirmed**
- **num_hidden_layers**: 126
- **intermediate_size**: 53248
- **num_attention_heads**: 128
- **num_key_value_heads**: 8 (GQA)
- **vocab_size**: 128256
- **max_position_embeddings**: 131072

### Qwen2.5-72B (Alibaba)
- **Source**: HuggingFace config.json (`Qwen/Qwen2.5-72B-Instruct`)
- **Architecture**: Dense transformer (no MoE)
- **hidden_size**: 8192
- **num_hidden_layers**: 80
- **intermediate_size**: 29568
- **num_attention_heads**: 64
- **num_key_value_heads**: 8 (GQA)
- **vocab_size**: 152064
- **max_position_embeddings**: 32768

### GPT-4 (OpenAI) — RUMORED
- **Source**: Various public analyses (SemiAnalysis, etc.), NOT officially confirmed
- **Architecture**: MoE
- **Total params**: ~1.76T (rumored)
- **hidden_size**: Uncertain — estimates range from 12288 to 20480
  - Most common rumor: **16384** or **20480**
  - Some analyses suggest 16 experts of ~111B each
- **MoE config (rumored)**: 8–16 experts, 2 active per token
- **Note**: All values are speculative. OpenAI has never released architecture details.

### Claude 3.5 Sonnet (Anthropic) — UNKNOWN
- **Source**: No official architecture disclosure
- **hidden_size**: Completely unknown. Speculation based on inference costs places it in the 8192–16384 range.
- Anthropic has never published model architecture details for any Claude model.

---

## 5. Summary Table

| Model | hidden_size | layers | intermediate | MoE? | Experts | Active/tok | Total Params |
|---|---|---|---|---|---|---|---|
| **Llama-3.1-405B** | **16384** | 126 | 53248 | No | — | — | 405B |
| **GPT-4 (rumored)** | 16384–20480 | ? | ? | Yes | ~16 | 2 | ~1.76T |
| **Qwen2.5-72B** | 8192 | 80 | 29568 | No | — | — | 72B |
| **DeepSeek-V3** | 7168 | 61 | 18432 | Yes | 256+1 | 8 | 671B/37B |
| **DeepSeek-R1** | 7168 | 61 | 18432 | Yes | 256+1 | 8 | 671B/37B |
| **DeepSeek-V4-Pro** | 7168 | 61 | — (MoE only) | Yes | 384+1 | 6 | TBD |
| **DeepSeek-V4-Flash** | 4096 | 43 | — (MoE only) | Yes | 256+1 | 6 | TBD |
| **GLM-5** | 6144 | 78 | 12288 | Yes | 256+1 | 8 | TBD |
| **GLM-5.1** | 6144 | 78 | 12288 | Yes | 256+1 | 8 | TBD |
| **Kimi-K2-Instruct** | 7168 | 61 | 18432 | Yes | 384+1 | 8 | TBD |
| **Kimi-K2.5** | 7168 | 61 | 18432 | Yes | 384+1 | 8 | TBD |
| **Kimi-K2.6** | 7168 | 61 | 18432 | Yes | 384+1 | 8 | TBD |
| **Claude 3.5 Sonnet** | unknown | unknown | unknown | unknown | unknown | unknown | unknown |

---

## 6. Analysis: The hidden_size Ceiling

### Does any open model exceed 8192?
**Yes.** Llama-3.1-405B uses **hidden_size=16384** — this is the highest confirmed value.

### Does any model exceed 16384?
**Not confirmed for any open model.** GPT-4 is rumored to potentially use 16384 or 20480, but this is unconfirmed. No open-weight model currently exceeds 16384.

### Why MoE models use smaller hidden_size
All MoE models cluster at hidden_size=7168 (DeepSeek, Kimi) or 6144 (GLM-5). This is a deliberate trade-off:
- MoE models gain capacity through many experts, not width
- A wider hidden_size would explode parameter count when combined with large expert counts
- Dense intermediate_size of 18432 already provides significant FFN capacity
- The MoE adds routing to specialized experts with moe_intermediate_size=2048–3072

### Trend: hidden_size is NOT the scaling axis
Modern models scale through:
1. **More layers** (Llama-3.1-405B: 126 layers at 16384 width)
2. **More experts** (DeepSeek-V4: 384 experts, Kimi K2: 384 experts)
3. **Longer context** (DeepSeek-V4-Pro: 1M, Kimi K2: 262K, GLM-5: 202K)
4. **MLA/DeepSeek-style KV compression** (allowing efficient attention with lower KV heads)

Hidden_size expansion appears to have plateaued. The jump from 8192 (GPT-3 era) to 16384 (Llama-3.1-405B) may be the last major doubling for a while, as MoE architectures achieve capacity more efficiently.

---

## 7. Architectural Patterns Observed

### DeepSeek-V3 architecture family (most influential design)
DeepSeek-V3's architecture has become a template adopted by multiple labs:
- **Kimi K2 series**: Uses DeepseekV3ForCausalLM directly
- **GLM-5**: Independent implementation but with similar MLA+MoE design
- **DeepSeek-V4**: Evolution of the same architecture

Key shared traits:
- MLA (Multi-head Latent Attention) with KV compression (kv_lora_rank)
- Q compression (q_lora_rank)
- MoE with shared expert + many routed experts
- First-k dense replace (usually 1–3 dense layers before MoE)
- YaRN rope scaling for long context
- Sigmoid or sqrtsoftplus gating
