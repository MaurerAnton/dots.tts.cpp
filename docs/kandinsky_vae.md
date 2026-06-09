# Kandinsky VAE Architecture Research

## Executive Summary

Kandinsky (by Sber AI / SberDevices) uses a **VQGAN-based** autoencoder — specifically **SBER-MoVQGAN (Modulated Vector Quantized GAN)** — for all image-generation versions (2.x and 3.x). Kandinsky 4.0 pivots to video generation and uses a completely different VAE (3D causal CogVideoX VAE). The VQGAN VAE is a discrete-codebook model (not a KL-VAE), meaning it quantizes the latent representation into discrete codes.

---

## 1. VAE Architecture — SBER-MoVQGAN (VQGAN / VQModel)

### What it is

Kandinsky 2.x and 3.x use a **VQGAN** (Vector Quantized GAN), not a KL-VAE (like Stable Diffusion's). The specific implementation is **SBER-MoVQGAN** — Sber's own tuned version of the MoVQGAN architecture.

In HuggingFace Diffusers, the model class is `VQModel` and the directory is named `movq/`.

### Original paper

The architecture traces back to the MoVQGAN paper:

> **"MoVQ: Modulating Quantized Vectors for High-Fidelity Image Generation"**
> Chuanxia Zheng, Long Tung Vuong, Jianfei Cai, Dinh Phung (2022)
> https://arxiv.org/abs/2209.09002

MoVQGAN improves upon VQGAN (Esser et al., 2021) by:
- Adding **spatially conditional normalization** to modulate quantized vectors, inserting spatially-variant information into embedded index maps
- Using **multi-channel quantization** to increase recombination capability without increasing model/codebook cost

### Key differences from Stable Diffusion's VAE

| Property | Stable Diffusion (KL-VAE) | Kandinsky (SBER-MoVQGAN) |
|----------|--------------------------|--------------------------|
| Type | Continuous KL-regularized VAE | Discrete VQGAN (codebook-based) |
| Quantization | None (Gaussian posterior) | Vector quantization (16384 codes) |
| Latent | Continuous Gaussian latents | Discrete code indices + continuous embeddings |
| Compression | f=8 (downsample 8x) | f=8 (downsample 8x) |
| Latent channels | 4 | 4 (vq_embed_dim=4) |
| Latent resolution (256 in) | 32x32 | 32x32 |
| Latent resolution (1024 in) | 128x128 | 128x128 |

---

## 2. Latent Dimension and Compression Ratio

### Kandinsky 2.x (K2.1, K2.2)

From the `movq/config.json` on HuggingFace:

```json
{
  "_class_name": "VQModel",
  "in_channels": 3,
  "out_channels": 3,
  "latent_channels": 4,
  "num_vq_embeddings": 16384,
  "vq_embed_dim": 4,
  "sample_size": 32,
  "scaling_factor": 0.18215,
  "block_out_channels": [128, 256, 256, 512],
  "down_block_types": [
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "AttnDownEncoderBlock2D"
  ],
  "up_block_types": [
    "AttnUpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D"
  ],
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "norm_type": "spatial",
  "act_fn": "silu"
}
```

**Key parameters:**
- **Compression ratio: f=8** — A 256x256 image → 32x32 latent grid; 1024x1024 → 128x128
- **Latent channels: 4**
- **Codebook size: 16,384** discrete codes
- **Embedding dimension: 4** (per code)
- **Scaling factor: 0.18215** (same value as Stable Diffusion, interestingly)
- **4 encoder blocks**: 3× DownEncoderBlock2D + 1× AttnDownEncoderBlock2D
- **4 decoder blocks**: 1× AttnUpDecoderBlock2D + 3× UpDecoderBlock2D
- **Channel progression**: 3 → 128 → 256 → 256 → 512 (bottleneck)
- **Parameter count: ~67M** (K2.x)

### Kandinsky 3.x

Kandinsky 3.0/3.1 uses a **larger MoVQGAN**:
- **Parameter count: ~267M** (4× larger than K2.x)
- Same f=8 compression ratio
- Same 32x32 latent grid for 256x256 images
- Loaded via `get_movq()` from `ai-forever/Kandinsky3.0` (file: `weights/movq.pt`)
- Works in float32 (does not work properly in fp16 for inpainting)

The K3.0 technical report (arxiv:2312.03511) references both:
- https://github.com/ai-forever/MoVQGAN
- https://github.com/ai-forever/tuned-vq-gan

### Kandinsky 4.0

Kandinsky 4.0 is a **video generation** model. It uses a completely different VAE:
- **3D causal CogVideoX VAE** (for spatio-temporal compression)
- CogVideoX paper: https://arxiv.org/abs/2408.06072
- This VAE compresses in both spatial AND temporal dimensions for video
- Not a VQGAN/MoVQGAN

---

## 3. Who Developed It

### Sber AI team (SberDevices / Sberbank AI)

The VAE was developed by the Sber AI team, built on top of open-source foundations:

**Original foundations:**
- VQGAN by Esser et al. (2021) — the base discrete autoencoder concept
- MoVQGAN by Zheng et al. (2022) — introduced spatial modulation and multi-channel quantization

**Sber's modifications (SBER-MoVQGAN):**
- Tuned for higher reconstruction quality, especially on faces and text
- Scaled to multiple sizes: 67M, 102M, 270M parameter variants
- Integrated into the Kandinsky pipeline with latent diffusion

**Key contributors** (from the MoVQGAN repo and Kandinsky papers):
- Anastasia Maltseva
- Arseniy Shakhmatov (cene555)
- Andrey Kuznetsov (kuznetsoffandrey)
- Denis Dimitrov (denndimitrov)
- Vladimir Arkhipkin
- Igor Pavlov
- Andrei Filatov
- + others from Sber AI

**Repo**: https://github.com/ai-forever/MoVQGAN

**Is it borrowed from Stable Diffusion?** No. Stable Diffusion uses a KL-VAE (continuous latents, Gaussian posterior). Kandinsky uses a VQGAN (discrete codebook, vector quantization). These are fundamentally different architectures. The only commonality is the f=8 compression ratio and 4 latent channels (convergent design choices).

---

## 4. Papers and Technical Reports

### Primary references

| Paper | Link | Relevance |
|-------|------|-----------|
| **MoVQGAN** (Zheng et al., 2022) | https://arxiv.org/abs/2209.09002 | Original MoVQGAN architecture; spatial modulation + multi-channel quantization |
| **VQGAN** (Esser et al., 2021) | https://arxiv.org/abs/2012.09841 | Base VQGAN architecture (Taming Transformers) |
| **Kandinsky 3.0 Technical Report** | https://arxiv.org/abs/2312.03511 | Describes K3 architecture including 267M MoVQ encoder/decoder |
| **Kandinsky 3 System Demo** (EMNLP 2024) | https://aclanthology.org/2024.emnlp-demo.48 | Academic paper on K3 system |
| **Kandinsky 2.2** (HF model card) | https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder | Architecture overview; explicitly mentions MoVQGAN |
| **Kandinsky 4.0** (video) | https://github.com/ai-forever/Kandinsky-4 | Uses CogVideoX 3D VAE |
| **CogVideoX** | https://arxiv.org/abs/2408.06072 | 3D causal VAE used in K4.0 |

### Blog posts (Russian, Habr)

- K2.2: https://habr.com/ru/companies/sberbank/articles/747446/
- MoVQGAN: https://habr.com/ru/companies/sberbank/articles/740624/
- K3.0: https://habr.com/ru/companies/sberbank/articles/775590/
- K3.1: https://habr.com/ru/companies/sberbank/articles/805337/
- K4.0: https://habr.com/ru/companies/sberbank/articles/866156/

### Sber's open-source VAE repos

- **SBER-MoVQGAN**: https://github.com/ai-forever/MoVQGAN
- **Tuned-VQ-GAN**: https://github.com/ai-forever/tuned-vq-gan

---

## 5. Evolution Across Kandinsky Versions

### Kandinsky 1.0 (2022)
- Early version, not widely documented
- Used VQGAN-based approach
- FID on COCO: 15.40

### Kandinsky 2.0 / 2.1 (2023)
- **VAE**: SBER-MoVQGAN 67M
- Architecture: CLIP-ViT-L/14 text+image encoder → Diffusion Image Prior (1B) → Latent Diffusion U-Net (1.22B) → MoVQGAN decoder (67M)
- Discrete codebook VAE with f=8 compression
- FID on COCO: 8.21 (K2.1)

### Kandinsky 2.2 (2023)
- **VAE**: Same SBER-MoVQGAN 67M (unchanged)
- Key change: CLIP-ViT-G replaces ViT-L/14 (1.8B image encoder vs 427M)
- MoVQGAN unchanged from K2.1
- Supports 1024x1024 output (128x128 latents)
- Added ControlNet support

### Kandinsky 3.0 (2023)
- **VAE**: SBER-MoVQGAN 267M (4× larger!)
- Architecture: Flan-UL2 text encoder (8.6B) → Latent Diffusion U-Net (3B) → MoVQGAN 267M
- Dropped the CLIP image prior (simpler architecture)
- Same f=8 compression, larger VAE for better reconstruction
- Technical report: arxiv 2312.03511

### Kandinsky 3.1 (2024)
- **VAE**: Same MoVQGAN 267M (unchanged from K3.0)
- Added: Kandinsky Flash (distilled 4-step inference), IP-Adapter, ControlNet, KandiSuperRes
- VAE unchanged

### Kandinsky 4.0 (2024) — Video era
- **VAE**: **3D causal CogVideoX VAE** (completely different!)
- Text-to-video (T2V) and image-to-video (I2V)
- T2V Flash: distilled version, 12-second 480p in 11 seconds on H100
- Uses MMDiT-like transformer (not U-Net)
- V2A (video-to-audio) model also released

### Summary table

| Version | VAE | VAE Params | Compression | Latent Dim | Codebook |
|---------|-----|-----------|-------------|------------|----------|
| K2.x | SBER-MoVQGAN | 67M | f=8 | 4×32×32 | 16384 |
| K3.x | SBER-MoVQGAN (big) | 267M | f=8 | 4×32×32 | 16384 |
| K4.0 | CogVideoX 3D VAE | N/A | 3D (spatial+temporal) | video latents | N/A |

---

## 6. Are the VAE Weights Open-Source?

**Yes, all Kandinsky VAE weights are open-source under Apache 2.0 license.**

### Kandinsky 2.x MoVQGAN (67M)

- **HuggingFace**: `kandinsky-community/kandinsky-2-2-decoder` → `movq/` directory
  - https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder/tree/main/movq
  - Contains `diffusion_pytorch_model.safetensors` and `config.json`
- **Standalone MoVQGAN checkpoints**:
  - 67M: https://huggingface.co/ai-forever/MoVQGAN/resolve/main/movqgan_67M.ckpt
  - 102M: https://huggingface.co/ai-forever/MoVQGAN/resolve/main/movqgan_102M.ckpt
  - 270M: https://huggingface.co/ai-forever/MoVQGAN/resolve/main/movqgan_270M.ckpt
- **Repo**: https://github.com/ai-forever/MoVQGAN (training code + configs)

### Kandinsky 3.x MoVQGAN (267M)

- **HuggingFace**: `ai-forever/Kandinsky3.0` → `weights/movq.pt`
  - https://huggingface.co/ai-forever/Kandinsky3.0
  - Loaded via `get_movq()` from the kandinsky3 Python package
- **Repo**: https://github.com/ai-forever/Kandinsky-3 (inference code)

### Kandinsky 4.0 CogVideoX VAE

- **HuggingFace**: `ai-forever/kandinsky-4-t2v-flash`
  - https://huggingface.co/ai-forever/kandinsky-4-t2v-flash
- **Repo**: https://github.com/ai-forever/Kandinsky-4 (inference code)

### How to load in Diffusers (K2.x example)

```python
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16
)
# The MoVQ VAE is automatically loaded as pipe.movq
```

### License

All Kandinsky models (including VAEs) are released under **Apache 2.0** license — fully permissive for both research and commercial use.

---

## 7. Technical Deep-Dive: SBER-MoVQGAN vs Standard VQGAN

### Standard VQGAN (Esser et al., 2021)
- Encoder: CNN backbone → vector quantization → codebook lookup
- Decoder: CNN backbone with standard convolutions
- Issue: "repeated artifact" — similar adjacent patches get mapped to the same code index, causing visible grid artifacts

### MoVQGAN (Zheng et al., 2022) — Key Innovations

1. **Spatially Conditional Normalization**: Instead of just looking up a codebook vector, the decoder modulates the quantized vectors with spatially-variant information. This breaks the "same code → same output" problem for adjacent regions that happen to share a code.

2. **Multi-channel Quantization**: Uses multiple quantization channels to increase the effective codebook expressiveness without increasing the codebook size.

### SBER-MoVQGAN — Sber's Improvements

Sber took the MoVQGAN paper implementation and:
- Scaled it up (67M → 102M → 270M variants)
- Tuned specifically for faces, text, and complex scenes
- Used codebook size 16384 (larger than original MoVQGAN's 1024)
- Latent grid 32×32×4 (original MoVQGAN used 16×16×4)
- Achieved state-of-the-art reconstruction (FID 0.686, SSIM 0.741, PSNR 27.04 on ImageNet with 270M variant)

### Quantization Mechanism

The VQ process:
1. Encoder produces continuous vectors z_e of shape (B, 4, H/8, W/8)
2. Each spatial position's 4-dim vector is mapped to the nearest of 16,384 learned codebook vectors
3. The discrete code indices are the latent representation
4. Decoder reconstructs from the codebook vectors

For diffusion: the UNet operates directly on the continuous codebook vectors (after lookup), not on the discrete indices. The scaling factor 0.18215 normalizes the latent distribution for the diffusion process.

---

## References

1. Zheng et al. "MoVQ: Modulating Quantized Vectors for High-Fidelity Image Generation." arXiv:2209.09002, 2022.
2. Esser et al. "Taming Transformers for High-Resolution Image Synthesis." CVPR 2021.
3. Arkhipkin et al. "Kandinsky 3.0 Technical Report." arXiv:2312.03511, 2023.
4. Arkhipkin et al. "Kandinsky 3: Text-to-Image Synthesis for Multifunctional Generative Framework." EMNLP 2024.
5. SBER-MoVQGAN GitHub: https://github.com/ai-forever/MoVQGAN
6. Kandinsky-2 GitHub: https://github.com/ai-forever/Kandinsky-2
7. Kandinsky-3 GitHub: https://github.com/ai-forever/Kandinsky-3
8. Kandinsky-4 GitHub: https://github.com/ai-forever/Kandinsky-4
9. HuggingFace: https://huggingface.co/kandinsky-community (K2.x) and https://huggingface.co/ai-forever (K3.x, K4.x)
