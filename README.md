# dots.tts.cpp

First C++ implementation of [dots.tts](https://github.com/rednote-hilab/dots.tts) — a 2,386,220,193-parameter multilingual text-to-speech model supporting 24 languages, with zero-shot voice cloning and 48 kHz output.

**Status: Active development** — Full pipeline (BPE → LLM → PatchEncoder → DiT → VAE → BigVGAN) is end-to-end. Ongoing: quality refinement, byte-level verification against the Python reference.

## Supported Languages (24)

The 24 languages from the [MiniMax Multilingual benchmark](https://github.com/rednote-hilab/dots.tts#minimax-multilingual-24-languages):

Arabic, Cantonese, Chinese, Czech, Dutch, English, Finnish, French, German,
Greek, Hindi, Indonesian, Italian, Japanese, Korean, Polish, Portuguese,
Romanian, Russian, Spanish, Thai, Turkish, Ukrainian, Vietnamese

## Architecture

```
Text → BPE Tokenizer → LLM (Qwen2.5-1.5B) → hidden_proj
     → PatchEncoder → FM buffer → DiT (flow matching) → AudioVAE → 48kHz WAV
```

> **LLM backbone**: Initialized from [Qwen2.5-1.5B-Base](https://github.com/rednote-hilab/dots.tts#-architecture).
> "We initialize the LLM from Qwen2.5-1.5B Base and feed it text directly as BPE instead of phonemes."
> — [arXiv:2606.07080](https://arxiv.org/abs/2606.07080), Section 2.3

### Components

| Component | Layers | Hidden | Params (exact) | Status |
|-----------|--------|--------|----------------|--------|
| **LLM** (Qwen2.5-1.5B-Base) | 28 | 1536 | 1,545,672,706 | Via llama.cpp |
| **DiT** (AR flow-matching head) | 18 | 1024 | 346,920,320 | Done |
| **PatchEncoder** (semantic encoder) | 24 | 1024 | 305,498,752 | Done |
| **BigVGAN decoder** | 6 stages, 18 AMP blocks | 24–1536 | 136,509,072 | Done |
| **AudioVAE encoder** | 7 Conv1d stages | 12–768 | 44,360,140 | Done |
| **CAM++** (speaker encoder) | 53 D-TDNN | 512 | 7,259,203 | Done |
| **BPE Tokenizer** | — | [vocab=151,672](models/token_vocab.txt#L151672) | — | Via llama.cpp |
| **Total** | 6/6 components | — | **2,386,220,193** | |

### DiT (flow-matching head)

- 18 layers, hidden=1024, 16 heads, FFN=4096
- adaLN modulation: timestep + speaker x-vector → shift/scale/gate
- Self-attention with RoPE (theta=10000) and qk_norm
- Predicts velocity field v_t; solved with Euler/Midpoint ODE + classifier-free guidance

### PatchEncoder (VAE semantic encoder)

- 24-layer causal Transformer, hidden=1024, FFN=4096
- Downsampling: 25 Hz VAE latents → 6.25 Hz LLM tokens (4× compression)
- Streaming decode_step for autoregressive generation

### BigVGAN decoder

- 6 upsampling stages, 3 SnakeBeta AMP blocks per stage
- 1920-sample hop, 48 kHz output, SLSTM bottleneck (4 layers, hidden=512)

### AudioVAE encoder (causal conv stack)

- 7 strided Conv1d stages: channels 12→24→48→96→192→384→768→128
- Strides: [2, 2, 2, 4, 6, 10], total 1920× downsampling (48kHz → 25Hz)
- Each stage: transition conv + ResStack (6 dilated conv pairs, dil=1,2,4,8,16,32)
- Weight normalization (weight_g + weight_v) + LeakyReLU(0.2)
- Fully causal: left-padding only, zero lookahead
- Output: 128-dim VAE latents at 25 Hz

### CAM++ (speaker encoder)

- 3D-Speaker CAM++ architecture (Alibaba DAMO Academy, Apache 2.0)
- Input: 16kHz PCM audio → FBank (80-dim mel) → 512-dim x-vector
- FCM head: 2× Conv2d + 4× BasicResBlock (32→32, stride 2/2/2/1)
- D-TDNN backbone: 3 blocks (12+24+16 CAMDenseTDNN layers), growth=32
- CAM context aggregation + StatsPool → Dense projection
- Frozen weights from `speaker_encoder.safetensors` (7,259,203 params)

## File Layout

```
dots.tts.cpp/
├── src/
│   ├── e2e_pipeline.cpp         # End-to-end CLI
│   ├── gpt2_bpe_tokenizer.cpp   # BPE tokenizer (GPT-2 / Mistral)
│   ├── gguf_extract.cpp         # GGUF weight extraction utility
│   ├── modules/
│   │   ├── backbone/
│   │   │   ├── dit_forward.cpp  # DiT forward pass
│   │   │   ├── dit_attention.cpp# DiT self-attention
│   │   │   └── patchenc.cpp     # PatchEncoder
│   │   └── vocoder/
│   │       ├── bigvgan_cpp.cpp  # BigVGAN decoder
│   │       ├── audiovae.cpp     # AudioVAE wrapper
│   │       ├── audiovae_encoder.cpp # AudioVAE encoder
│   │       └── lstm.cpp         # SLSTM bottleneck
│   │   └── speaker/
│   │       └── campp.cpp        # CAM++ speaker encoder
│   └── utils/
│       ├── safetensors.cpp      # safetensors parser
│       ├── dit_loader.cpp       # DiT weight loader
│       └── patchenc_loader.cpp  # PatchEncoder weight loader
├── include/                     # Public headers
│   ├── dots_tts.h               # Architecture constants
│   ├── dit.h, patchenc.h        # DiT / PatchEncoder APIs
│   ├── bigvgan_cpp.h, audiovae.h# Vocoder APIs
│   ├── campp.h                  # CAM++ speaker encoder API
│   ├── safetensors.h, gpt2_bpe_tokenizer.h
│   └── ...
├── models/                      # Model conversion & verification tools (Python)
│   ├── convert_dots_tts.py      # safetensors → GGUF (DiT+PatchEncoder)
│   ├── extract_llm_gguf.py      # Qwen2.5 LLM → GGUF
│   ├── extract_dots_llm.py      # Mistral LLM → GGUF (legacy)
│   ├── debug_dump.py            # BigVGAN intermediate dump (reference)
│   ├── dump_amp.py              # AMP block dump (reference)
│   ├── vocoder_bridge.py        # C++ latents → Python vocoder → WAV
│   ├── hybrid_tts.py            # Full Python dots.tts generation
│   ├── tok.py                   # HuggingFace tokenizer test
│   ├── decode_latents.py        # DiT latent decoder
│   └── token_vocab.txt          # BPE vocabulary
├── tools/                       # Auxiliary tools
│   ├── compare_pipelines.py     # Byte-level C++ vs Python comparison
│   ├── web_demo.py              # Flask web UI
│   └── web_server.cpp           # C++ HTTP server
├── test/                        # Test sources
│   ├── test_llm.cpp             # LLM integration test
│   ├── test_tokenizer.cpp       # BPE tokenizer test
│   └── test_vocoder.cpp         # Vocoder test
├── attic/                       # Old/experimental code
├── CMakeLists.txt
├── README.md
└── LICENSE                      # GPL-3.0-or-later
```

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp ../llama
cd dots.tts.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./e2e_pipeline "Hello world"
```

Requires: C++17, CMake 3.18+, ggml (from llama.cpp)

## Python Tools

The `models/` directory contains **offline tools** — no Python is needed at runtime:

| Tool | Purpose |
|------|---------|
| `convert_dots_tts.py` | Convert DiT+PatchEncoder safetensors → GGUF (one-time) |
| `extract_llm_gguf.py` | Extract Qwen2.5 LLM → GGUF (one-time) |
| `compare_pipelines.py` | Byte-level C++ vs Python verification |
| `debug_dump.py` / `dump_amp.py` | Dump PyTorch intermediates for C++ debugging |

## License

GNU General Public License v3.0 or later (SPDX: GPL-3.0-or-later).

See [LICENSE](LICENSE) for the full text. Copyright (C) 2026 Anton Maurer.

## References

- [rednote-hilab/dots.tts](https://github.com/rednote-hilab/dots.tts) — original Python implementation (Apache 2.0)
- [arXiv:2606.07080](https://arxiv.org/abs/2606.07080) — technical report
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — ggml tensor library
