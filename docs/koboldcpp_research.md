# KoboldCpp Architecture Research

## Overview

KoboldCpp (github.com/LostRuins/koboldcpp) is a massive, self-contained AI inference engine that builds on **llama.cpp** and **stable-diffusion.cpp** (SDCPP) in a single codebase. It supports **LLM text generation, image generation (SD/Flux/WAN), speech-to-text (Whisper), text-to-speech (OuteTTS, Kokoro, Parler, Dia, Orpheus, Qwen3-TTS), embeddings, and music generation (AceStep)** — all sharing the same ggml backend infrastructure.

The project is structured as a C/C++ shared library (`expose.cpp`/`expose.h`) with a Python frontend (`koboldcpp.py`) that loads the native library via ctypes and exposes a web UI and REST API.

---

## 1. Architecture: Multi-Model Pipeline

### 1.1 Modular Adapter Pattern

KoboldCpp's core insight is a **clean separation of model types into independent adapter modules**, all linked into a single shared library:

```
expose.cpp          — Flat C ABI for Python ctypes (the "public API")
├── gpttype_adapter.cpp   — LLM text generation (llama.cpp-based)
├── sdtype_adapter.cpp    — Stable Diffusion / Flux / WAN image generation
├── whisper_adapter.cpp   — Whisper speech-to-text
├── tts_adapter.cpp       — TTS (OuteTTS, Kokoro, Parler, Dia, Orpheus, Qwen3-TTS)
├── embeddings_adapter.cpp — Text embeddings
└── music_adapter.cpp     — AceStep music generation
```

Each adapter is compiled into its own static library in CMake:
```cmake
add_library(tts_adapter otherarch/tts_adapter.cpp)
add_library(sdtype_adapter otherarch/sdcpp/sdtype_adapter.cpp ...)
add_library(whisper_adapter otherarch/whispercpp/whisper_adapter.cpp)
add_library(music_adapter otherarch/acestep/music_adapter.cpp)
add_library(embeddings_adapter otherarch/embeddings_adapter.cpp)
```

The final shared library links all adapters:
```cmake
target_link_libraries(koboldcpp_cublas PUBLIC
    gpttype_adapter whisper_adapter music_adapter tts_adapter
    embeddings_adapter sdtype_adapter ...)
```

### 1.2 Separate Model Loading Per Type

Each model type has its own load/generate flow — models are **loaded independently** and can coexist in memory:

```c
// From expose.h — separate structs per model type
struct sd_load_model_inputs { ... };    // SD image gen
struct whisper_load_model_inputs { ... }; // STT
struct tts_load_model_inputs { ... };    // TTS
struct music_load_model_inputs { ... };  // Music
struct embeddings_load_model_inputs { ... }; // Embeddings
```

Python calls them independently:
```python
handle.sd_load_model(sd_inputs)
handle.tts_load_model(tts_inputs)
handle.whisper_load_model(whisper_inputs)
```

### 1.3 TTS Sub-Architecture: Three Backends

The TTS adapter (`otherarch/tts_adapter.cpp`) detects the model architecture from GGUF metadata and routes to one of three backends:

1. **TTS.CPP** (Parler, Kokoro, Dia, Orpheus) — `otherarch/ttscpp/`
2. **Qwen3-TTS** — `otherarch/qwen3tts/` (has GPU offload support!)
3. **OuteTTS** — uses two llama.cpp contexts directly

```c
detectedarch = gguf_get_model_arch(modelfile_ttc);
if (detectedarch == "qwen3-tts") {
    is_qwen3tts_file = true;
} else if (TTSCPP_SUPPORTED_ARCHITECTURES.find(detectedarch) != ...) {
    is_ttscpp_file = true;
} else {
    // OuteTTS fallback
}
```

---

## 2. TTS / Speech Capabilities

### 2.1 TTS Models Supported

| Architecture | GGUF Arch Tag | Vocoder | Notes |
|---|---|---|---|
| **Kokoro** | `kokoro` | Built-in | 82M params, fastest, uses espeak-ng phonemizer |
| **Parler-TTS** | `parler-tts` | DAC | Cross-attention based, supports voice descriptions |
| **Dia** | `dia` | DAC | 1.6B, high quality |
| **Orpheus** | `orpheus` | SNAC | Supports emotional TTS |
| **Qwen3-TTS** | `qwen3-tts` | Audio tokenizer | Has GPU offload! Uses shared ggml backend |
| **OuteTTS** | Generic LLM GGUF | WavTokenizer (separate GGUF) | Voice cloning via speaker embeddings |

### 2.2 Qwen3-TTS GPU Offload Pattern (Key for dots.tts.cpp!)

The Qwen3-TTS implementation in `otherarch/qwen3tts/` is the **only TTS backend with GPU support**. Its pattern:

```cpp
// gguf_loader.cpp — shared backend with ref counting
struct shared_backend_state {
    ggml_backend_t backend = nullptr;
    int32_t ref_count = 0;
};

ggml_backend_t init_preferred_backend(const char * component_name, ...,
                                       bool allow_gpu) {
    // Try iGPU first, then GPU, then ACCEL, then CPU
    ggml_backend_t backend = ggml_backend_init_by_type(
        GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    if (allow_gpu) {
        if (!backend)
            backend = ggml_backend_init_by_type(
                GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    // ...
}
```

The `qwen3_tts.cpp` manages:
- `transformer_` (TTSTransformer) — the main TTS model
- `audio_encoder_` (AudioTokenizerEncoder)
- `audio_decoder_` (AudioTokenizerDecoder)
- `text_tokenizer_` (TextTokenizer)

Each component gets its own ggml context and optionally shares the backend.

### 2.3 OuteTTS Voice Cloning

Uses two separate llama.cpp models (TTS + WavTokenizer) with traditional token-based generation:
- Speaker voice is generated first and cached for reuse
- Speaker voice codes are stored as JSON for sharing (see `examples/outetts/speakers/`)
- Voice cloning done via Python `outetts` package, output is a JSON file of audio tokens

### 2.4 Whisper (Speech-to-Text)

Simple adapter in `otherarch/whispercpp/whisper_adapter.cpp` wrapping the whisper.cpp API:
```c
bool whispertype_load_model(const whisper_load_model_inputs inputs);
whisper_generation_outputs whispertype_generate(const whisper_generation_inputs inputs);
```

---

## 3. Interesting Code Patterns for dots.tts.cpp

### 3.1 GGUF-Based Model Architecture Detection

KoboldCpp uses GGUF metadata to auto-detect model type:

```cpp
std::string detectedarch = gguf_get_model_arch(filename);
if (detectedarch == "kokoro") { ... }
else if (detectedarch == "parler-tts") { ... }
else if (detectedarch == "qwen3-tts") { ... }
```

TTS.CPP defines its supported architectures in a compile-time map:
```cpp
const std::map<std::string, tts_arch> TTSCPP_SUPPORTED_ARCHITECTURES = {
    { "parler-tts", PARLER_TTS_ARCH },
    { "kokoro", KOKORO_ARCH },
    { "dia", DIA_ARCH },
    { "orpheus", ORPHEUS_ARCH }
};
```

### 3.2 GGML Context Per Model + Weight Assignment

TTS.CPP creates **one ggml_context for weights** loaded from GGUF, then **assigns weights by name** to model-specific structs:

```cpp
struct tts_runner * kokoro_from_file(gguf_context * meta_ctx,
    ggml_context * weight_ctx, int n_threads, ...) {

    kokoro_model * model = new kokoro_model;
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);

    // Weight assignment: iterate all tensors, assign by name
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx);
         cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        runner->assign_weight(cur->name, cur);
    }
    runner->prepare_post_load();
    gguf_free(meta_ctx);
    ggml_free(weight_ctx);  // Free weight context after assignment
}
```

Each model has its own `build_new_*_context()` that creates a **compute context**:
```cpp
struct kokoro_context * kctx = build_new_kokoro_context(model, n_threads, cpu_only);
```

### 3.3 GGML Backend Abstraction

KoboldCpp uses llama.cpp's ggml backend system extensively. The key abstraction layers:

1. **`ggml_backend_t`** — abstract device (CPU, CUDA, Vulkan, Metal, RPC)
2. **`ggml_backend_buffer_type_t`** — memory allocation strategy
3. **`ggml_backend_meta`** — multi-device orchestrator (splits compute across GPUs)
4. **`ggml_backend_reg`** — backend registration system

Multiple backends coexist:
- CPU (always available)
- CUDA (GGML_USE_CUDA)
- Vulkan (GGML_USE_VULKAN)
- Metal (GGML_USE_METAL)
- RPC (GGML_USE_RPC) — remote compute

The `#define` pattern for backend-aware compilation:
```makefile
# Makefile
VULKAN_FLAGS = -DGGML_USE_VULKAN -DSD_USE_VULKAN
CUBLAS_FLAGS = -DGGML_USE_CUDA -DSD_USE_CUDA
# Metal
CFLAGS += -DGGML_USE_METAL -DSD_USE_METAL
```

### 3.4 RPC Backend for Multi-Machine/Device

KoboldCpp includes an **RPC backend** (`ggml/src/ggml-rpc/`) that allows offloading compute to remote machines:

```cpp
// Serializes ggml_tensor over the wire
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    // ...
};
```

Commands include: `ALLOC_BUFFER`, `SET_TENSOR`, `GRAPH_COMPUTE`, `COPY_TENSOR`, etc.

Exposed via `load_model_inputs.rpc_mode`:
```c
const int rpc_mode = 0; // 0=disabled, 1=connect, 2=host
const char * rpc_targets = nullptr;
```

### 3.5 Graph Cutting for Multi-Device (SDCPP)

The SDCPP integration includes a sophisticated **ggml graph cutting** system (`otherarch/sdcpp/ggml_graph_cut.cpp`) that splits computation graphs across devices:

```cpp
namespace sd::ggml_graph_cut {
    // Plan splits a ggml_cgraph into segments assigned to different backends
    struct Plan {
        struct InputShape { ggml_type type; int64_t ne[4]; };
        // ...
    };
    // Segments are computed on different devices
    struct Segment {
        size_t compute_buffer_size;
        size_t input_param_bytes;
        size_t input_previous_cut_bytes;
        // ...
    };
}
```

This enables SD models to span multiple GPUs or GPU+CPU.

### 3.6 GGML Extend (SDCPP Helper Library)

`otherarch/sdcpp/ggml_extend.hpp` provides high-level tensor operations on top of raw ggml:

```cpp
// n-mode tensor-matrix product
ggml_tensor* ggml_ext_mul_n_mode(ggml_context* ctx, ggml_tensor* a,
                                  ggml_tensor* b, int mode = 0);

// Attention with optional mask
ggml_tensor* ggml_ext_attention(ggml_context* ctx, ...);

// Layer norm, group norm, SiLU, GELU, etc.
```

This is a **great pattern for dots.tts.cpp** — build a small extension library over ggml for TTS-specific ops.

### 3.7 Continuous Batching

The LLM adapter supports continuous batching:
```c
const int continuous_batching_slots = 0;
```

With a batch API:
```c
int gpttype_batch_generate_submit(const generation_inputs inputs);
bool gpttype_batch_generate_has_finished(int request_id);
const char * gpttype_batch_generate_new_token(int request_id, int idx);
```

---

## 4. Vulkan Backend Usage

### 4.1 Build System

Vulkan is a first-class backend. Build with:
```bash
make LLAMA_VULKAN=1
```

Key flags:
```makefile
VULKAN_FLAGS = -DGGML_USE_VULKAN -DSD_USE_VULKAN
```

The `SD_USE_VULKAN` flag propagates Vulkan support to the SDCPP image generation pipeline.

### 4.2 Vulkan Shader Compilation

KoboldCpp bundles `glslc` (both Linux and Windows binaries) for shader compilation. Shaders are compiled into C++ headers at build time:
```makefile
VKGEN_HPP = ggml/src/ggml-vulkan-shaders$(VKGEN_SUFFIX).hpp
VKGEN_CPP = ggml/src/ggml-vulkan-shaders$(VKGEN_SUFFIX).cpp
```

The Vulkan backend (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`, ~924KB) supports:
- Cooperative matrix extensions (VK_NV_cooperative_matrix, VK_NV_cooperative_matrix_decode_vector)
- Integer dot product
- Multiple Vulkan device enumeration and selection

### 4.3 Multi-Architecture Vulkan Builds

KoboldCpp produces **multiple binaries** for different CPU capabilities with Vulkan:
- `koboldcpp_vulkan` — full AVX2
- `koboldcpp_vulkan_noavx2` — no AVX2
- `koboldcpp_vulkan_failsafe` — minimum compatibility

### 4.4 Vulkan Headers

Bundled Vulkan headers live in `include/vulkan/`:
- `vulkan/` — Vulkan core headers
- `spirv-headers/`, `spirv-tools/`, `spirv_cross/` — SPIR-V tooling
- `shaderc/`, `glslang/`, `slang/` — shader compilers
- `dxc/` — DirectX shader compiler

---

## 5. Model Format / GGUF Patterns

### 5.1 Universal GGUF

Everything uses GGUF. Model architectures are identified by the `general.architecture` metadata key:
- LLM: `llama`, `gpt-neox`, `falcon`, `rwkv`, etc.
- TTS: `kokoro`, `parler-tts`, `dia`, `orpheus`, `qwen3-tts`
- SD: `sd1`, `sdxl`, `flux`, `wan`, `sd3`, `qwenimage`, etc.

### 5.2 Multi-File Model Loading

Some models require multiple GGUF files:
```c
// SD: separate models for different components
const char * t5xxl_filename = nullptr;
const char * clip1_filename = nullptr;
const char * clip2_filename = nullptr;
const char * vae_filename = nullptr;

// TTS (OuteTTS): two models
const char * ttc_model_filename = nullptr;  // TTS core
const char * cts_model_filename = nullptr;  // WavTokenizer

// Music: four models
const char * musicllm_filename = nullptr;
const char * musicembedding_filename = nullptr;
const char * musicdiffusion_filename = nullptr;
const char * musicvae_filename = nullptr;
```

### 5.3 Quantization

Models use the standard GGML quantization types (`SD_TYPE_Q4_0`, `SD_TYPE_Q5_K`, etc.) plus newer types:
- `SD_TYPE_TQ1_0`, `SD_TYPE_TQ2_0` — ternary quantization
- `SD_TYPE_MXFP4` — MXFP4 format
- `SD_TYPE_NVFP4` — NVFP4 format

### 5.4 LoRA Support (SD)

SDCPP supports LoRA with dynamic application:
```c
const int lora_len = 0;
const char ** lora_filenames = nullptr;
const float * lora_multipliers = nullptr;
const int lora_apply_mode = 0; // bitfield: dynamic, cached, etc.
```

---

## 6. Streaming / Real-Time Features

### 6.1 Token Streaming

Text generation supports SSE streaming:
```c
const bool stream_sse = false;  // Enable Server-Sent Events
```

The Python layer implements SSE endpoints:
```python
wake_requests = [
    "/api/extra/generate/stream",
    "/api/extra/tts",
    "/api/extra/transcribe",
    "/sdapi/v1/txt2img",
    # ...
]
```

### 6.2 Streaming via Callbacks

The `generated_tokens` vector provides streaming access:
```c
extern std::vector<std::string> generated_tokens;
extern bool generation_finished;

const char * new_token(int idx);   // Get token by index
int get_stream_count();             // Total tokens so far
bool has_finished();                // Generation complete?
```

### 6.3 RPC for Real-Time Distributed Inference

The RPC backend enables real-time distributed inference — tensors and graphs are serialized and sent to remote workers:
```cpp
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    // ...
};
```

---

## 7. Summary of Patterns Relevant to dots.tts.cpp

### 7.1 What to Adopt

| Pattern | Description | Why It's Useful |
|---|---|---|
| **Adapter-per-modality** | Separate load/generate per model type | Clean separation; add TTS without touching LLM code |
| **GGUF arch detection** | `gguf_get_model_arch()` to route to correct backend | Support multiple TTS models in one binary |
| **Weight-by-name assignment** | Load GGUF tensors, assign to model structs by name | Flexible model loading, easy to add new architectures |
| **Shared ggml backend** | Reference-counted backend shared across sub-models | GPU memory efficiency for multi-component TTS |
| **Backend fallback chain** | Try iGPU→GPU→ACCEL→CPU | Works everywhere without user config |
| **ggml_extend pattern** | Helper ops over raw ggml (attention, norms, activations) | Build TTS-specific ops library |
| **Multiple GGUF per model** | Text encoder + decoder + vocoder as separate files | Matches TTS modular architecture |
| **RPC backend** | Remote compute for heavy models | Offload vocoder to separate machine |
| **Vulkan as first-class** | `SD_USE_VULKAN` / `GGML_USE_VULKAN` flags | Cross-platform GPU support including AMD |

### 7.2 What to Avoid / Improve

| Issue | Why | Better Approach |
|---|---|---|
| **#include .cpp files** | `tts_adapter.cpp` uses `#include "ttscpp.cpp"` | Use proper static libraries like SDCPP does |
| **Multiple ggml versions** | Ships ggml_v1, v2, v3 alongside modern ggml | Use single ggml version from llama.cpp upstream |
| **Single shared library** | All adapters linked into one .so/.dll | Consider dynamic plugin loading for optional backends |
| **OuteTTS uses two llama.cpp contexts** | Inefficient, double the backend overhead | Native ggml model implementation (like TTS.CPP) |

### 7.3 Key Files to Study

| File | What It Shows |
|---|---|
| `otherarch/tts_adapter.cpp` | How to route between multiple TTS backends |
| `otherarch/qwen3tts/gguf_loader.cpp` | Shared backend with ref counting |
| `otherarch/qwen3tts/qwen3_tts.cpp` | Multi-component model loading |
| `otherarch/ttscpp/src/ttscpp.cpp` | Model-from-file factory pattern |
| `otherarch/ttscpp/include/ttscommon.h` | Clean abstraction for TTS runners |
| `otherarch/sdcpp/ggml_extend.hpp` | Helper ops library over ggml |
| `otherarch/sdcpp/ggml_graph_cut.cpp` | Multi-device graph partitioning |
| `ggml/src/ggml-backend-meta.cpp` | Multi-backend orchestration |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Remote compute backend |
| `CMakeLists.txt` | How to build modular adapters as static libs |
| `expose.h` | Clean C ABI design for Python interop |
