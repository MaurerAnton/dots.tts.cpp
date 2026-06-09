# C++ TTS and Voice Cloning Ecosystem (2026)

Comprehensive survey of C++ Text-to-Speech and voice cloning implementations
as of June 2026. Focus on native C++ inference (no Python wrappers), GGML/GGUF
or ONNX backends, and active maintenance (commits in 2025-2026).

---

## Table of Contents

1. [Reference: ggml Ecosystem](#reference-ggml-ecosystem)
2. [Active GGML-Based TTS Engines](#active-ggml-based-tts-engines)
3. [Active ONNX-Based TTS Engines](#active-onnx-based-tts-engines)
4. [Classical / Formant TTS](#classical--formant-tts)
5. [Inactive / Stale C++ TTS](#inactive--stale-cpp-tts)
6. [STT Reference](#stt-reference)
7. [Python-Only Projects (No C++ Port)](#python-only-projects-no-c-port)
8. [Summary Matrix](#summary-matrix)
9. [Recommendations](#recommendations)

---

## Reference: ggml Ecosystem

Core infrastructure projects that C++ TTS engines build upon.

### ggml

- URL: https://github.com/ggml-org/ggml
- Stars: 14,779
- License: MIT
- Language: C++
- Status: Active (pushed 2026-06-08)
- Role: Tensor library for ML inference

The foundational tensor computation library. All GGML-based TTS projects
depend on this.

### llama.cpp

- URL: https://github.com/ggml-org/llama.cpp
- Stars: 115,720
- License: MIT
- Language: C++
- Status: Active (pushed 2026-06-09)
- Role: LLM inference reference; GGUF format originator

Not a TTS engine itself, but the GGUF model format and many ggml backend
implementations (CUDA, Metal, Vulkan, ROCm, SYCL) originate here and are
reused by TTS projects.

---

## Active GGML-Based TTS Engines

Projects using the GGML/GGUF ecosystem for native C++ TTS inference.

### qwentts.cpp

- URL: https://github.com/ServeurpersoCom/qwentts.cpp
- Stars: 34
- License: MIT
- Language: C++17
- Status: **Active** (pushed 2026-06-09)
- Backend: GGML (CPU, CUDA, ROCm, Metal, Vulkan)
- Model Format: GGUF (pre-converted on HuggingFace)
- Upstream Model: Qwen3-TTS (QwenLM/Alibaba)

**Model Support:**
- Qwen3-TTS 12Hz (0.6B and 1.7B parameter sizes)
- Modes: base, customvoice (named speakers), voicedesign
- Q8_0 and Q4_K_M quantization for talker backbone
- RVQ codec paths kept at F32
- Pre-converted GGUFs: https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF

**Voice Cloning:** Zero-shot from reference clip (x-vector or in-context with transcript)

**Voice Design:** Free-text attribute instruction (gender, age, pitch, style)

**Languages:** 11 languages with Mandarin dialect overrides

**Features:**
- Streaming synthesis with autoregressive frame loop
- Two-stage generation (Talker LM + code predictor MTP head)
- KV-cached inference
- Seedable Philox PRNG with HF-aligned sampling chain
- Two CLI tools: qwen-tts (text to WAV) and qwen-codec (WAV to/from RVQ)

**Assessment:** The most complete GGML port of Qwen3-TTS. Highly active.
Best option for voice cloning + voice design in pure C++ with GGML.

---

### cosyvoice.cpp

- URL: https://github.com/Lourdle/cosyvoice.cpp
- Stars: 26
- License: MIT
- Language: C++
- Status: **Active** (pushed 2026-06-09)
- Backend: GGML (CPU, CUDA, Metal, SYCL)
- Model Format: GGUF
- Upstream Model: CosyVoice3 (FunAudioLLM/Alibaba)

**Model Support:**
- CosyVoice3 0.5B (Fun-CosyVoice3-0.5B-2512)
- GGUF quantization: Q2_K through F16
- KV Cache quantization (f32/f16/q8_0/q5_1/q4_0)

**Voice Cloning:** Yes - prompt speech embedding with reuse across synthesis runs

**Languages:** Chinese primary (mandarin), English support

**Features:**
- OpenAI Speech API-compatible server (POST /v1/audio/speech)
- Interactive REPL CLI
- Prompt speech reuse (encode reference once, reuse many times)
- Audio backend plugins (MINIAUDIO, FFMPEG for MP3/AAC/FLAC/OPUS/M4A)
- Text splitting for long inputs
- Pre-converted GGUFs on ModelScope and HuggingFace
- Cross-platform: Windows (x64), Linux (x86_64), macOS (arm64) - CI tested
- Vulkan backend has known issues

**Assessment:** Excellent C++/GGML port focused on CosyVoice3. Voice cloning
with prompt reuse. OpenAI-compatible server makes it drop-in ready.

---

### kokopop

- URL: https://github.com/tterrasson/kokopop
- Stars: 6
- License: MIT
- Language: C++
- Status: **Active** (pushed 2026-05-29)
- Backend: GGML (CPU, CUDA, Metal, Vulkan)
- Model Format: GGUF
- Upstream Model: Kokoro-82M (hexgrad)

**Model Support:**
- Kokoro-82M in GGUF format
- Multiple built-in voices: af_heart, ff_siwis, zf_xiaoxiao, im_nicola

**Voice Cloning:** No (deterministic voice styles embedded in GGUF)

**Languages:** English, French, Chinese, Japanese (via Kokoro-82M voices)

**Features:**
- Zero Python dependencies (only libespeak-ng + ggml)
- Streaming API for real-time audio
- Chunked synthesis for long-form text
- WAV, PCM (float32/s16), Ogg/Opus output
- HTTP server with /v1/audio/speech endpoint
- Full C and C++ API (usable via FFI from Rust, Go, etc.)
- Voice listing endpoint (/voices)

**Assessment:** Clean, focused Kokoro-82M GGML runtime. Good for
embedded/standalone use. Small but well-designed.

---

### TTS.cpp

- URL: https://github.com/mmwillet/TTS.cpp
- Stars: 241
- License: MIT
- Language: C++
- Status: Active-ish (pushed 2025-10-05)
- Backend: GGML (CPU, Metal; CUDA/Vulkan planned)
- Model Format: GGUF
- Primary Platform: macOS

**Model Support:**

| Model | CPU | Metal | Quantization | GGUF Available |
|-------|-----|-------|--------------|----------------|
| Kokoro-82M | Yes | No | Yes | Yes |
| Orpheus-TTS | Yes | No | No | Yes |
| Parler TTS Mini | Yes | Yes | Yes | Yes |
| Parler TTS Large | Yes | Yes | Yes | Yes |
| Dia | Yes | Yes | Yes | Yes |

GGUF models: https://huggingface.co/mmwillet2

**Voice Cloning:** No (uses model-inherent voices)

**Languages:** Depends on model (Kokoro: en/fr/zh/ja; Orpheus: en; Parler: en)

**Features:**
- Server mode
- Custom ggml fork with TTS-specific ops (conv transpose, STFT, iSTFT, round, mod, cumsum)
- Kokoro is the recommended model (reliable, articulate)
- Proof-of-concept status - mostly tested on macOS

**Assessment:** Ambitious multi-model GGML TTS. Proof-of-concept quality.
Important for establishing GGML TTS patterns. Not production-ready but
valuable reference for the Orpheus-TTS GGUF conversion.

---

## Active ONNX-Based TTS Engines

Projects using ONNX Runtime for native C++ TTS inference.

### sherpa-onnx

- URL: https://github.com/k2-fsa/sherpa-onnx
- Stars: 12,864
- License: Apache 2.0
- Language: C++
- Status: **Highly Active** (pushed 2026-06-08)
- Backend: ONNX Runtime
- Model Format: ONNX

**TTS Model Support:**
- Piper models (English, German, 30+ languages via community voices)
- VITS models (multilingual)
- Matcha-TTS (Chinese, English, Chinese+English)
- Kokoro-82M (via VoxSherpa integration)
- ZipVoice (voice cloning, Chinese+English)
- Pocket TTS (voice cloning, English)

**Voice Cloning:** Yes - ZipVoice and Pocket TTS

**Languages:** 50+ languages via Piper/VITS/Matcha/Kokoro models

**Key Features:**
- Also does STT, speaker diarization, VAD, audio tagging, keyword spotting, source separation, speech enhancement
- 12 programming languages supported
- WebAssembly, Android, iOS, HarmonyOS, Raspberry Pi, RISC-V
- NPU support: Rockchip, Qualcomm, Ascend, Axera
- Pre-built APKs, Flutter plugins, NPM packages
- HuggingFace spaces for demos

**Assessment:** The most comprehensive C++ speech processing framework.
TTS is one of many capabilities. Production-ready, exceptional platform
support. The gold standard for ONNX-based C++ TTS.

---

### piper1-gpl (Piper Successor)

- URL: https://github.com/OHF-Voice/piper1-gpl
- Stars: 4,371
- License: GPL-3.0
- Language: C++
- Status: **Active** (pushed 2026-04-07)
- Backend: ONNX Runtime (inference); custom C++ phonemizer
- Model Format: ONNX

This is the successor to the original (now archived) Piper project by
rhasspy. Maintained by OHF-Voice (Open Home Foundation).

**Model Support:**
- VITS-based Piper voice models
- 30+ languages via community-trained voices
- ONNX format models

**Voice Cloning:** No (fixed per-language voices)

**Languages:** 30+ languages (English, German, French, Spanish, Japanese, Chinese, Korean, Vietnamese, and more)

**Features:**
- Fast, local neural TTS
- Custom C++ phonemization (piper-phonemize, also archived)
- Low resource usage
- Embedded-friendly

**Assessment:** The go-to lightweight neural TTS engine. Widely deployed
in Home Assistant, Rhasspy, and other voice assistant projects.

---

### Original piper (Archived)

- URL: https://github.com/rhasspy/piper
- Stars: 11,067
- License: MIT
- Language: C++
- Status: **ARCHIVED** (last push 2025-08-26)

Development moved to piper1-gpl. Kept for historical reference.

---

### MeloTTS.cpp

- URL: https://github.com/apinge/MeloTTS.cpp
- Stars: 104
- License: Apache 2.0
- Language: C++
- Status: Active (pushed 2025-09-26)
- Backend: OpenVINO (Intel)
- Model Format: OpenVINO IR

**Model Support:**
- MeloTTS models (MyShell.ai)
- BERT + TTS + DeepFilterNet pipeline

**Voice Cloning:** No

**Languages:** Chinese (mixed English), English. Japanese planned.

**Features:**
- CPU, GPU, NPU device support via OpenVINO
- Multi-branch: EN only, ZH_MIX_EN, multilang-develop
- DeepFilterNet for denoising (int8 quantization noise)
- Works on Intel CPUs/GPUs/NPUs

**Assessment:** Good option for Intel hardware. OpenVINO-specific.
Limited model/language support compared to sherpa-onnx.

---

### babylon (Babylon.cpp)

- URL: https://github.com/Mobile-Artificial-Intelligence/babylon
- Stars: 35
- License: MIT
- Language: C++
- Status: Active (pushed 2026-04-14)
- Backend: ONNX Runtime
- Model Format: ONNX

**Model Support:**
- VITS models for speech synthesis
- DeepPhonemizer ONNX for grapheme-to-phoneme
- Piper models compatible after conversion

**Voice Cloning:** Yes (listed in topics; needs verification)

**Languages:** Model-dependent

**Features:**
- Grapheme-to-phoneme + TTS pipeline
- Piper model compatibility with conversion script
- Talks about ElevenLabs-level quality aspirations

**Assessment:** Interesting hybrid G2P+TTS library. Piper compatibility
is a key feature.

---

### kokoro.cpp (Various Forks)

Multiple small C++ ports of Kokoro-82M using ONNX Runtime:

- koth/kokoro.cpp (13 stars) - ONNX Runtime, last 2025-11-30, includes Mandarin
- olokobayusuf/kokoro.cpp (3 stars) - Single-header C++ library, last 2026-02-02
- szsteven008/kokoro.cpp (4 stars) - ONNX Runtime, last 2025-03-17

The GGML-based **kokopop** (see above) is more actively maintained.

---

### Other Small ONNX C++ TTS Projects

- StyleTTS2-onnx-cpp (3 stars) - StyleTTS2 + espeak-ng, last 2025-12-23
- Vits2-onnx-cpp (18 stars) - VITS2 + espeak-ng, last 2024-04-17
- vits.cpp (13 stars) - VITS + ONNX Runtime, last 2023-12-25
- SparkTTS.cpp (10 stars) - SparkTTS C++ inference, last 2025-09-04

---

## Classical / Formant TTS

### espeak-ng

- URL: https://github.com/espeak-ng/espeak-ng
- Stars: 6,544
- License: GPL-3.0
- Language: C
- Status: **Active** (pushed 2026-04-27)

**Model Type:** Formant synthesis (not neural)
**Voice Cloning:** No
**Languages:** 100+ languages and accents
**Features:**
- Extremely fast, tiny footprint
- Used as phonemizer by Kokoro, Piper, and others
- Android, Linux, macOS, Windows
- MBROLA voice support

**Assessment:** The universal phonemization backend for neural TTS systems.
Also useful standalone for accessibility and low-resource use cases.

---

## Inactive / Stale C++ TTS

Projects that were significant but are no longer actively maintained.

### bark.cpp

- URL: https://github.com/PABannier/bark.cpp
- Stars: 863
- License: MIT
- Language: C++
- Status: **Inactive** (last push 2024-11-16)
- Backend: GGML
- Upstream: Suno AI's Bark

GGML port of Suno's Bark TTS model. Was influential in establishing
the GGML-for-TTS pattern but has not seen updates in 18 months.

### tortoise.cpp

- URL: https://github.com/balisujohn/tortoise.cpp
- Stars: 194
- License: MIT
- Language: C++
- Status: **Inactive** (last push 2024-08-20)
- Backend: GGML
- Upstream: tortoise-tts

GGML reimplementation of tortoise-tts. Voice cloning capable but stale.

### pocket-tts.cpp

- URL: https://github.com/Codes4Fun/pocket-tts.cpp
- Stars: 6
- Language: C++
- Status: Inactive (last push 2026-02-05)
- Backend: GGML
- Upstream: Kyutai's Pocket TTS

GGML port of Pocket TTS. Voice cloning capable. Small, no recent activity.

---

## STT Reference

### whisper.cpp

- URL: https://github.com/ggml-org/whisper.cpp
- Stars: 50,585
- License: MIT
- Language: C++
- Status: **Highly Active** (pushed 2026-06-09)
- Backend: GGML (CPU, CUDA, Metal, Vulkan, and more)
- Model Format: GGML (custom binary format)

Speech-to-text (not TTS), included as the reference GGML voice AI project.
- OpenAI Whisper models (tiny through large-v3)
- 99 languages
- Streaming and non-streaming
- The blueprint for GGML C++ inference ports

---

## Python-Only Projects (No C++ Port)

Major TTS/voice cloning projects that are Python-only with no working
C++ inference port as of June 2026.

| Project | Stars | Description | C++ Status |
|---------|-------|-------------|------------|
| OpenVoice (myshell-ai) | 36,644 | Instant voice cloning by MIT/MyShell | No C++ port |
| Fish Speech (fishaudio) | 30,720 | SOTA Open Source TTS | No C++ inference port |
| Qwen3-TTS (QwenLM) | 11,823 | Alibaba TTS with voice cloning | C++ port exists: qwentts.cpp |
| Orpheus-TTS (canopyai) | 6,175 | Human-sounding speech | No standalone C++ port (GGUF via TTS.cpp) |
| Kokoro-82M (hexgrad/HF) | 13M+ downloads | High quality 82M param TTS | Multiple C++ ports exist |

Note: Qwen3-TTS and Kokoro-82M have community C++ ports but no official ones.

---

## Summary Matrix

### C++ TTS Engines - Comparison

| Engine | Stars | Backend | Format | Voice Clone | Languages | Active | Quality |
|--------|-------|---------|--------|-------------|-----------|--------|---------|
| sherpa-onnx | 12.9k | ONNX | ONNX | Yes (ZipVoice, Pocket TTS) | 50+ | Daily | Production |
| piper1-gpl | 4.4k | ONNX | ONNX | No | 30+ | Active | Production |
| espeak-ng | 6.5k | Native | Formant | No | 100+ | Active | Production |
| qwentts.cpp | 34 | GGML | GGUF | Yes (zero-shot) | 11 | Daily | Beta |
| cosyvoice.cpp | 26 | GGML | GGUF | Yes (prompt embedding) | CN+EN | Daily | Beta |
| kokopop | 6 | GGML | GGUF | No | 4 | Active | Beta |
| TTS.cpp | 241 | GGML | GGUF | No | Model-dep | Slow | PoC |
| MeloTTS.cpp | 104 | OpenVINO | IR | No | CN+EN | Slow | Working |
| babylon | 35 | ONNX | ONNX | Yes | Model-dep | Active | Working |
| bark.cpp | 863 | GGML | GGML | No | Multi | Stale | Legacy |
| tortoise.cpp | 194 | GGML | GGML | Yes | EN | Stale | Legacy |

### Backend Distribution

| Backend | Projects | Notes |
|---------|----------|-------|
| GGML/GGUF | qwentts.cpp, cosyvoice.cpp, kokopop, TTS.cpp, bark.cpp, tortoise.cpp | Growing ecosystem; homegrown format |
| ONNX Runtime | sherpa-onnx, piper1-gpl, babylon, kokoro.cpp forks | Mature, broad hardware support |
| OpenVINO | MeloTTS.cpp | Intel-specific |
| Native/Formant | espeak-ng | Classic, tiny footprint |

---

## Recommendations

### For Production Use (2026)

1. **sherpa-onnx** - The clear winner for production C++ TTS. Supports
   the most models (Piper, VITS, Matcha, Kokoro, ZipVoice), the most
   platforms, and has voice cloning. Actively maintained daily.

2. **piper1-gpl** - Best lightweight option. 30+ languages, tiny
   footprint, battle-tested in Home Assistant and voice assistants.

### For Voice Cloning in C++

1. **qwentts.cpp** - Most complete voice cloning C++ port. Zero-shot
   cloning from reference audio. Voice design from text prompts.
   GGML-backed with GPU acceleration.

2. **cosyvoice.cpp** - Strong Chinese voice cloning with prompt reuse.
   OpenAI-compatible API server.

3. **sherpa-onnx** (ZipVoice/Pocket TTS) - Production voice cloning
   via ONNX Runtime.

### For GGML/GGUF Ecosystem Integration

1. **kokopop** - Clean, minimal dependency GGML runtime. Good reference
   implementation for Kokoro-82M.

2. **qwentts.cpp** - Most advanced GGML TTS codebase. Shows patterns
   for caching, quantization, streaming.

### Gaps and Opportunities

- No official C++ port from Orpheus-TTS, Fish Speech, or OpenVoice teams
- GGML Orpheus-TTS exists only as proof-of-concept in TTS.cpp
- Fish Speech has no C++ inference port at all (30k stars, no port)
- OpenVoice has no C++ inference port despite 36k stars
- Growing but fragmented GGML TTS ecosystem - no unified "tts.cpp" like whisper.cpp

---

*Research conducted: June 9, 2026. Star counts approximate, from GitHub API.*
