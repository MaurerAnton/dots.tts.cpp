# dots.tts.cpp

First C++ implementation of [dots.tts](https://github.com/rednote-hilab/dots.tts) — a 2,386,220,193-parameter multilingual text-to-speech model with 24-language support (including Russian, English, German), zero-shot voice cloning, and 48kHz output.

**Status: Active development / testing & polishing** — The full pipeline (BPE → LLM → PatchEncoder → DiT → VAE → BigVGAN) is implemented end-to-end. Currently refining output quality, verifying against the Python reference, and polishing the codebase.

**Phase 1 MVP** — DiT + Flow Matching + PatchEncoder compile, build graphs, and execute correctly on CPU via ggml.

## Architecture

```
Text -> BPE Tokenizer -> Qwen2.5-1.5B LLM -> PatchEncoder
     -> FM buffer -> DiT (flow matching) -> AudioVAE -> 48kHz WAV
```

### Components

| Component | Layers | Hidden | Params | Status |
|-----------|--------|--------|--------|--------|
| **DiT** (flow-matching head) | 18 | 1024 | 346,920,320 | Done |
| **PatchEncoder** (VAE semantic encoder) | 24 | 1024 | 305,498,752 | Done |
| **Flow Matching ODE** (Euler, Midpoint, CFG) | - | - | - | Done |
| **AudioVAE encoder** (7 Conv1d stages) | 7 | - | 44,360,140 | TODO |
| **BigVGAN decoder** (6 upsampling + 18 AMP blocks) | 6 | - | 136,509,072 | Done |
| **LLM backbone** (Qwen2.5-1.5B) | 28 | 1536 | 1,545,672,706 | Reuses llama.cpp |
| **CAM++** (speaker encoder) | - | 512 | 7,259,203 | TODO |
| **BPE Tokenizer** | - | vocab=151,936 | - | Reuses llama.cpp |

### What works now

- 18-layer DiT with adaLN modulation (timestep + speaker x-vector -> shift/scale/gate)
- Self-attention with RoPE and qk_norm
- FFN with SiLU (1024 -> 4096 -> 1024)
- 24-layer PatchEncoder with causal attention and streaming decode_step
- Euler and Midpoint (RK2) ODE solvers with classifier-free guidance
- Full ggml graph build and compute on CPU
- Dummy weight loading for testing (random init)

### Test output

```
=== DiT forward pass ===
Output: [128 x 32] min=-0.021282 max=0.018719 mean=0.000095

=== PatchEncoder forward pass ===
PatchEncoder output: [1536 x 4] min=-0.002728 max=0.002307 mean=-0.000006

=== PatchEncoder streaming ===
  patch 0: LLM emb mean=-0.000007
  patch 1: LLM emb mean=0.000006
  patch 2: LLM emb mean=-0.000008
  patch 3: LLM emb mean=-0.000011
PatchEncoder streaming: OK
```

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp ../llama
cd dots.tts.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./dots_tts
```

Requires: C++17, CMake 3.18+, ggml (from llama.cpp)

## License

Apache 2.0 — same as upstream dots.tts

## Reference

- [rednote-hilab/dots.tts](https://github.com/rednote-hilab/dots.tts) — original Python implementation
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — ggml tensor library
