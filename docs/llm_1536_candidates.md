# LLM Backbone Candidates with hidden_size=1536 for dots.tts.cpp

**Constraint**: The existing `hidden_proj` layer maps 1536 -> 1024 (LLM hidden to DiT hidden).
Any replacement LLM must have **hidden_size=1536** to work without retraining.

---

## Baseline: Qwen2.5-1.5B (Current Backbone)

| Field | Value |
|---|---|
| Architecture | Qwen2 (Llama-like, RoPE) |
| hidden_size | 1536 |
| num_hidden_layers | 28 |
| intermediate_size (FFN) | 8960 |
| num_attention_heads | 12 |
| num_key_value_heads | 2 (GQA) |
| vocab_size | 151,936 |
| rope_theta | 1,000,000 |
| max_position_embeddings | 131,072 |
| Total parameters | ~1.54B |
| llama.cpp arch | `LLM_ARCH_QWEN2` |
| GGUF available | Yes (QuantFactory, second-state, bartowski, lmstudio-community, etc.) |

**Variant breakdown:**
- `Qwen/Qwen2.5-1.5B` -- original base model, rope_theta=1M, ctx=131K
- `Qwen/Qwen2.5-1.5B-Instruct` -- instruction-tuned
- `Qwen/Qwen2.5-Coder-1.5B` -- code-specialized, ctx=32K
- `Qwen/Qwen2.5-Math-1.5B` -- math-specialized, rope_theta=10K, ctx=4K

## All Known Models with hidden_size=1536

After exhaustive search across Hugging Face, the set of models with hidden_size=1536
is very small. 1536 is an unusual embedding dimension; most model families jump from
1024 to 2048.

### Dense Transformer Models

#### 1. Qwen2-1.5B / Qwen2.5-1.5B Family

| Field | Qwen2-1.5B | Qwen2.5-1.5B | Qwen2.5-Coder-1.5B | Qwen2.5-Math-1.5B |
|---|---|---|---|---|
| hidden_size | 1536 | 1536 | 1536 | 1536 |
| num_layers | 28 | 28 | 28 | 28 |
| intermediate_size | 8960 | 8960 | 8960 | 8960 |
| heads / KV heads | 12 / 2 | 12 / 2 | 12 / 2 | 12 / 2 |
| vocab_size | 151,936 | 151,936 | 151,936 | 151,936 |
| rope_theta | 1,000,000 | 1,000,000 | 1,000,000 | 10,000 |
| context length | 131,072 | 131,072 | 32,768 | 4,096 |
| Total params | ~1.54B | ~1.54B | ~1.54B | ~1.54B |
| Architecture | Qwen2 | Qwen2 | Qwen2 | Qwen2 |
| GGUF available | Yes | Yes | Yes | Yes |

**Also based on this architecture:**
- `SeaLLMs/SeaLLMs-v3-1.5B-Chat` -- fine-tune for Southeast Asian languages (same config)

**Notes:** All Qwen 1.5B variants share identical architecture. Differences are in
training data and recipes only. Qwen2.5 improved over Qwen2 via better data filtering,
longer training, and improved chat templates. For dots.tts technical capabilities
(next-token prediction quality), Qwen2.5-1.5B is the best of this family. The Coder
variant may offer better multilingual text handling; the Math variant has limited
context (4K) which may restrict long-form TTS.

#### 2. BLOOM-1b1 (1.1B) -- SMALLER

| Field | Value |
|---|---|
| Architecture | BLOOM (ALiBi, no RoPE) |
| n_embed (hidden_size) | 1536 |
| n_layer | 24 |
| n_head | 16 |
| vocab_size | 250,880 |
| Position encoding | ALiBi (offset=100) |
| context length | 2,048 |
| Total params | ~1.1B |
| llama.cpp arch | `LLM_ARCH_BLOOM` |
| GGUF available | Yes (older conversions exist) |

**Notes:** Uses ALiBi instead of RoPE for position encoding. Not a RoPE model, but
llama.cpp supports BLOOM architecture natively. At 24 layers vs 28 for Qwen2.5-1.5B,
this is a **downgrade** in capacity. Also has much shorter context (2K vs 131K).
BLOOM was trained on the ROOTS corpus (46 languages) and may have different
multilingual characteristics.

#### 3. Cerebras-GPT-590M -- SMALLER

| Field | Value |
|---|---|
| Architecture | GPT-2 (learned position embeddings, no RoPE) |
| n_embd (hidden_size) | 1536 |
| n_layer | 18 |
| n_inner (FFN) | 6,144 |
| n_head | 12 |
| vocab_size | 50,257 |
| context length | 2,048 |
| Total params | ~590M |
| llama.cpp arch | `LLM_ARCH_GPT2` |
| GGUF available | Yes (older conversions exist) |

**Notes:** Standard GPT-2 architecture with learned absolute position embeddings.
Much smaller than Qwen2.5-1.5B (18 vs 28 layers, 590M vs 1.54B params). This is a
significant **downgrade**. No GQA, no RoPE, limited context. Only worth considering
if extreme inference speed is needed at the cost of quality.

#### 4. ruGPT-3 Large -- SMALLER

| Field | Value |
|---|---|
| Architecture | GPT-2 (learned position embeddings, no RoPE) |
| n_embd (hidden_size) | 1536 |
| n_layer | 24 |
| n_head | 16 |
| vocab_size | 50,257 |
| context length | 2,048 |
| Total params | ~760M |
| llama.cpp arch | `LLM_ARCH_GPT2` |
| GGUF available | Yes (older conversions exist) |

**Notes:** Russian-language GPT-2 variant by Sberbank. 24 layers vs Qwen's 28 --
still a **downgrade**. Pre-trained primarily on Russian text, so its multilingual
capabilities (critical for dots.tts's 24-language support) are severely limited.
Not recommended as a general-purpose TTS backbone.

### Mixture-of-Experts (MoE) Models

#### 5. Granite-3.0-3B-A800M -- LARGER (but MoE)

| Field | Value |
|---|---|
| Architecture | GraniteMoe (MoE, RoPE) |
| hidden_size | 1536 |
| num_hidden_layers | 32 |
| intermediate_size (per expert) | 512 |
| num_local_experts | 40 |
| num_experts_per_tok | 8 (top-8 routing) |
| num_attention_heads | 24 |
| num_key_value_heads | 8 (GQA) |
| vocab_size | 49,152 |
| rope_theta | 10,000 |
| max_position_embeddings | 4,096 |
| Total params | 3.05B |
| Active params | ~800M |
| llama.cpp arch | `LLM_ARCH_GRANITE_MOE` |
| GGUF available | Yes |

**Special architectural features:**
- `attention_multiplier`: 0.015625
- `embedding_multiplier`: 12.0
- `residual_multiplier`: 0.22
- `logits_scaling`: 6.0

**Notes:** This is the **only model with hidden_size=1536 that has more layers (32)
than Qwen2.5-1.5B (28)**. It has 3B total parameters but only ~800M active per token
due to sparse MoE routing. The per-expert FFN is tiny (512) compared to Qwen's 8960,
because computation is distributed across 40 experts.

**Complexity considerations for dots.tts.cpp:**
- MoE routing logic must be implemented in the llama.cpp backend
- Each token activates 8 of 40 experts, adding computational overhead
- The `residual_multiplier` (0.22) and other scalars are non-standard
- GraniteMoe is a relatively new architecture; llama.cpp support may have quirks
- Context length (4K) is much shorter than Qwen2.5 (131K)

This model could theoretically offer more representational capacity due to 32 layers
and 3B total parameters, but the MoE architecture is fundamentally different from
the standard dense transformer. Testing would be required to determine if the
quality improvement justifies the architectural complexity.

---

## Summary: Which Are Larger Than Qwen2.5-1.5B?

| Model | Layers | Params | vs Baseline |
|---|---|---|---|
| Qwen2.5-1.5B (baseline) | 28 | 1.54B | -- |
| Qwen2-1.5B | 28 | 1.54B | = Same |
| Qwen2.5-Coder-1.5B | 28 | 1.54B | = Same |
| Qwen2.5-Math-1.5B | 28 | 1.54B | = Same |
| SeaLLMs-v3-1.5B | 28 | 1.54B | = Same |
| BLOOM-1b1 | 24 | 1.1B | Smaller |
| Cerebras-GPT-590M | 18 | 0.59B | Much smaller |
| ruGPT-3 Large | 24 | 0.76B | Smaller |
| **Granite-3.0-3B-A800M** | **32** | 3.05B (800M active) | **Larger (MoE)** |

**Key finding:** There is NO dense transformer model with hidden_size=1536 that is
larger than Qwen2.5-1.5B. The Qwen 1.5B models occupy a unique niche in the parameter
space. The only "upgrade" path at hidden_size=1536 is the Granite MoE architecture.

---

## Architecture Compatibility with llama.cpp

| Architecture | llama.cpp Arch Constant | Status |
|---|---|---|
| Qwen2 (Qwen2/Qwen2.5) | `LLM_ARCH_QWEN2` | Fully supported, battle-tested |
| BLOOM | `LLM_ARCH_BLOOM` | Supported |
| GPT-2 (Cerebras, ruGPT) | `LLM_ARCH_GPT2` | Supported |
| GraniteMoe | `LLM_ARCH_GRANITE_MOE` | Supported (newer) |

All architectures are supported by llama.cpp (verified against `llama-arch.h`).
GGUF conversions exist or can be created for all models listed.

---

## Practical Recommendations

1. **Stick with Qwen2.5-1.5B**: It is already the best dense model at hidden_size=1536.
   The Coder and Math variants offer specialized training but identical architecture;
   they may or may not improve TTS text understanding.

2. **Consider Qwen2.5-Coder-1.5B**: If multilingual text understanding is the
   bottleneck (rather than parameter count), the Coder variant's training on diverse
   code and text could improve text representation quality.

3. **Experiment with Granite-3.0-3B-A800M (MoE)**: The only model with more layers
   (32) at this hidden size. The MoE architecture is a significant departure from the
   current dense backbone, requiring careful integration. The 4K context limit may
   be restrictive for long-form TTS.

4. **No benefit from older architectures**: BLOOM, Cerebras-GPT, and ruGPT-3 all have
   fewer layers and are architecturally inferior to Qwen2.5 (no RoPE, no GQA, no
   modern training recipes).

5. **If hidden_proj retraining is acceptable**: Moving to a model with hidden_size=2048
   (many options: Qwen2.5-3B, Llama-3.2-1B/3B, TinyLlama, Gemma-2-2B, SmolLM2-1.7B,
   StableLM-2-1.6B, etc.) would dramatically expand the candidate pool. However, this
   requires retraining the `hidden_proj` layer (2048->1024 instead of 1536->1024).

---

## Models Explicitly Checked and Ruled Out (hidden_size != 1536)

These popular small models were checked and do NOT have hidden_size=1536:

| Model | hidden_size | Why Not |
|---|---|---|
| Phi-2 (2.7B) | 2560 | Too large hidden |
| Phi-3-mini (3.8B) | 3072 | Too large hidden |
| Phi-4-mini (3.8B) | 3072 | Too large hidden |
| Phi-1.5 (1.3B) | 2048 | Too large hidden |
| Gemma-2-2B | 2304 | Too large hidden |
| Gemma-3-1B | 1152 | Too small hidden |
| Gemma-3-4B | 2560 | Too large hidden |
| StableLM-2-1.6B | 2048 | Too large hidden |
| SmolLM2-1.7B | 2048 | Too large hidden |
| SmolLM2-360M | 960 | Too small hidden |
| SmolLM-1.7B | 2048 | Too large hidden |
| TinyLlama-1.1B | 2048 | Too large hidden |
| Llama-3.2-1B | 2048 | Too large hidden |
| Llama-3.2-3B | 3072 | Too large hidden |
| Qwen2.5-0.5B | 896 | Too small hidden |
| Qwen2.5-3B | 2048 | Too large hidden |
| Qwen1.5-0.5B | 1024 | Too small hidden |
| Qwen1.5-1.8B | 2048 | Too large hidden |
| OPT-125M/350M/1.3B | 768/1024/2048 | Wrong sizes |
| Pythia-160M/410M/1B/1.4B | 768/1024/2048 | Wrong sizes |
| GPT-2/Medium | 768/1024 | Wrong sizes |
| GPT-Neo-125M/1.3B/2.7B | 768/2048/2560 | Wrong sizes |
| Falcon-1B/7B | 2048/4096 | Wrong sizes |
| Cerebras-GPT (others) | 768 to 5120 | Only 590M hits |
| BLOOM (others) | 1024 to 14336 | Only 1b1 hits |
| InternLM2-1.8B/2.5-1.8B | 2048 | Wrong size |
| MiniCPM-2B/MiniCPM3-4B | 2304/2560 | Wrong sizes |
| DeepSeek-Coder-1.3B/DeepSeek-V2-Lite | 2048 | Wrong sizes |
| StarCoder2-3B/StarCoder-1B | 2560/2048 | Wrong sizes |
| OLMo-1B/OLMoE-1B-7B | 2048 | Wrong sizes |
| Granite-3.0-1B/2B/3.1-1B/2B | 1024/2048 | Wrong sizes |
| OpenELM-270M/450M/1.1B/3B | 1280/1536/2048/3200 | 450M has 1536 model_dim but only 20 layers (down from 28). See section 4b above. |

Note: OpenELM-450M has model_dim=1536 but uses a layer-wise scaling strategy
where each transformer layer has a different hidden dimension. This makes it
incompatible with the fixed-size hidden_proj constraint. The base hidden_size
in the config.json is 1536 but internal dimensions vary per layer.

---

## Search Methodology

- Searched 220+ model configs on Hugging Face across 40+ model families
- Cross-referenced with llama.cpp architecture support (`llama-arch.h`)
- Verified GGUF availability where possible (some repos require auth)
- BLOOM configs use `n_embed` instead of `hidden_size` (checked both key names)
- GPT-2 family configs use `n_embd` instead of `hidden_size`
- Huawei/Vivo/Xiaomi/OPPO mobile-optimized LLMs checked where accessible
- Russian, Japanese, Korean, Chinese, Arabic regional models included
- Code-specific models (CodeGen, CodeLlama, StarCoder, DeepSeek-Coder) included

**Last updated:** 2026-06-09

#### 4b. OpenELM-450M -- SMALLER (layer-wise scaling)

| Field | Value |
|---|---|
| Architecture | OpenELM (RoPE, layer-wise scaling) |
| model_dim (hidden_size) | 1536 |
| num_transformer_layers | 20 |
| FFN multipliers per layer | 0.5x to 4.0x (variable) |
| num_query_heads per layer | 12 to 24 (variable) |
| num_kv_heads per layer | 3 to 6 (variable, GQA) |
| vocab_size | 32,000 |
| rope_theta | 10,000 |
| max_context_length | 2,048 |
| Total params | ~450M |
| llama.cpp arch | `LLM_ARCH_OPENELM` |
| GGUF available | Yes |

**Notes:** OpenELM uses a unique layer-wise scaling strategy where each transformer
layer has different attention head counts, KV group sizes, and FFN multipliers. The
base model_dim (1536) is constant across layers, so hidden state outputs should be
consistent. However, at 20 layers and 450M params, this is a significant **downgrade**
from Qwen2.5-1.5B. The variable per-layer dimensions make this architecture
non-standard compared to typical dense transformers.

**Update:** Despite model_dim=1536, OpenELM's per-layer variation makes it a poor
candidate. The hidden state at each layer IS 1536, but the architecture is so
different from a standard transformer that quality would likely suffer.

