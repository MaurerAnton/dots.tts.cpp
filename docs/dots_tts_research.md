# dots.tts Research Findings

## Sources

- **GitHub README**: https://github.com/rednote-hilab/dots.tts
- **arXiv Paper (Technical Report)**: https://arxiv.org/abs/2606.07080 (PDF: https://arxiv.org/pdf/2606.07080)
- **Hugging Face Collection**: https://huggingface.co/collections/rednote-hilab/dotstts

---

## 1. The 24 Languages

Evaluated on the MiniMax Multilingual benchmark. Full list from the README:

1. Arabic
2. Cantonese
3. Chinese
4. Czech
5. Dutch
6. English
7. Finnish
8. French
9. German
10. Greek
11. Hindi
12. Indonesian
13. Italian
14. Japanese
15. Korean
16. Polish
17. Portuguese
18. Romanian
19. Russian
20. Spanish
21. Thai
22. Turkish
23. Ukrainian
24. Vietnamese

**Source**: GitHub README, MiniMax Multilingual (24 languages) table.
**URL**: https://github.com/rednote-hilab/dots.tts#minimax-multilingual-24-languages

dots.tts (SCA) achieved highest avg speaker similarity (83.9), leading SIM on 19/24 languages outright.

---

## 2. LLM Backbone

### GitHub README (Architecture section)

> "LLM — initialized from Qwen2.5-1.5B-Base, consumes BPE text directly (no phonemes), and emits one hidden state per audio step."

**URL**: https://github.com/rednote-hilab/dots.tts#-architecture

### arXiv Technical Report (Section 2.3, LLM backbone)

> "We initialize the LLM from Qwen2.5-1.5B Base (Qwen Team, 2024) and feed it text directly as BPE instead of phonemes. On the audio side it runs at 6.25 Hz: the semantic encoder (Section 2.4) compresses each 25 Hz VAE latent patch into one audio-semantic embedding for the next LLM step, and the AR-FM head (Section 2.5) consumes the LLM hidden state to emit the next four-frame VAE patch."

**URL**: https://arxiv.org/pdf/2606.07080 (Page 4, Section 2.3)

### Acknowledgments (README)

> "Qwen2.5 — LLM backbone initialization."

**URL**: https://github.com/rednote-hilab/dots.tts#-acknowledgments

---

## 3. Parameter Counts per Component

Total system: **2B parameters** (stated in abstract, README header, and conclusion).

### Paper abstract:

> "We present dots.tts, a 2B-parameter continuous autoregressive text-to-speech (TTS) foundation model"

### Component breakdown:

The paper does NOT give exact parameter counts per component individually. What IS provided:

| Component | Detail | Source |
|-----------|--------|--------|
| LLM | Qwen2.5-1.5B-Base (~1.5B) | Paper Section 2.3 |
| Semantic Encoder | 24-layer causal Transformer, hidden dim 1024, FFN dim 4096 | Paper Section 2.4 |
| AR-FM Head (DiT) | 18 layers, hidden dim 1024, FFN dim 4096, RoPE | Paper Section 2.5 |
| AudioVAE Encoder | Causal conv stack, strides [2,2,2,4,6,10], 128-dim latent at 25 Hz | Paper Section 2.2 |
| AudioVAE Decoder | BigVGAN-v2 style causal decoder, 48 kHz output | Paper Section 2.2 |
| Speaker Encoder | Frozen CAM++ x-vector | Paper Section 2.5 |
| Stop Head | 2-layer MLP | Paper Section 2.6.2 |

Remaining ~0.5B params distributed across semantic encoder, AR-FM head, AudioVAE, CAM++, and small projections.

### Exact quote: Semantic Encoder (Section 2.4)

> "Architecturally, a strided causal-convolution projector first halves the input frame rate, followed by a 24-layer causal Transformer with hidden dim 1024 and FFN dim 4096; the encoder output is then grouped in pairs along time and linearly projected to the LLM embedding dimension, yielding an end-to-end 4x downsampling from 25 Hz latent frames to 6.25 Hz LLM tokens."

### Exact quote: AR-FM Head (Section 2.5)

> "Following DiTAR and ARDiT, we use a Diffusion Transformer (DiT) (Peebles and Xie, 2023) as the velocity-field predictor, instantiated with 18 layers, hidden dim 1024, and FFN dim 4096, with RoPE"

---

## Summary

dots.tts by rednote-hilab is a 2B-parameter fully continuous autoregressive TTS. LLM backbone: Qwen2.5-1.5B-Base. Supports 24 languages. All code and checkpoints Apache 2.0.
