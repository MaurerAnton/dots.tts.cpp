# Neural Vocoder/Decoder Lineage: From WaveNet to dots.tts

This document traces the architectural lineage of the BigVGAN-v2-style causal decoder
used in dots.tts, from the earliest neural vocoders through each major innovation.

---

## 1. WaveNet — The First Neural Vocoder

| Field       | Detail                                                        |
|-------------|---------------------------------------------------------------|
| Year        | 2016                                                          |
| Authors     | Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu |
| Institution | DeepMind                                                      |
| Paper       | "WaveNet: A Generative Model for Raw Audio" (arXiv:1609.03499) |

### What It Did

WaveNet introduced **autoregressive generation of raw audio waveforms** using a
stack of dilated causal convolutions. Each 16-bit sample was conditioned on all
previous samples via a deep stack of dilated convolutions with exponentially
growing dilation factors, giving the model an enormous receptive field while
keeping the number of layers manageable. It also introduced **gated activation
units** (inspired by PixelCNN) and conditioned generation on linguistic features
or mel-spectrograms for TTS.

### Key Innovation

- **Dilated causal convolutions** for exponential receptive field growth
- **Gated activation**: `z = tanh(W_f * x) * sigmoid(W_g * x)`
- **μ-law companding** of 16-bit audio to 256-way categorical distribution
- Showed that neural networks can directly model raw audio waveforms at sample
  level with quality surpassing concatenative and parametric TTS

### Problem It Solved

Before WaveNet, TTS used unit-selection concatenation or statistical parametric
methods (HMMs). These produced robotic, muffled speech. WaveNet demonstrated
that a deep generative model could produce nearly indistinguishable-from-human
speech by operating directly on raw waveform samples.

### Limitation

**Inference was painfully slow.** Because it was fully autoregressive at 16,000+
samples per second, real-time generation was impossible — a single utterance
could take minutes on a GPU. This spawned the search for faster alternatives.

---

## 2. WaveGlow — Flow-Based Fast Generation

| Field       | Detail                                                        |
|-------------|---------------------------------------------------------------|
| Year        | 2018                                                          |
| Authors     | Ryan Prenger, Rafael Valle, Bryan Catanzaro                   |
| Institution | NVIDIA                                                        |
| Paper       | "WaveGlow: A Flow-based Generative Network for Speech Synthesis" (arXiv:1811.00002) |

### What It Did

WaveGlow combined ideas from **Glow** (normalizing flows) and **WaveNet**
to create a non-autoregressive waveform generator. It used a stack of
affine coupling layers with invertible 1x1 convolutions, where each coupling
layer contained a WaveNet-like dilated convolution sub-network (the "WN"
block). Unlike WaveNet, it generated the entire waveform in one forward pass
by sampling from a simple Gaussian prior and inverting the flow.

### Key Innovation

- **Invertible affine coupling layers** — each layer splits channels, transforms
  half conditioned on the other half, making the Jacobian triangular and the
  determinant trivial to compute
- **Invertible 1x1 convolutions** between coupling layers to mix channels,
  generalizing the fixed permutation of Glow
- **Non-autoregressive**: single forward pass instead of sample-by-sample

### Problem It Solved

WaveNet's autoregressive bottleneck. WaveGlow could generate high-quality speech
**orders of magnitude faster** than WaveNet — achieving real-time or faster
synthesis on a GPU — while maintaining competitive quality.

### Limitation

The model was large (~87M parameters on a modest config) and still relatively
heavy. The affine coupling + WN blocks were computationally expensive. There was
a quality gap vs autoregressive models, and like all flow models it was
architecturally constrained by the requirement of invertibility.

---

## 3. MelGAN — GANs Enter the Picture

| Field       | Detail                                                        |
|-------------|---------------------------------------------------------------|
| Year        | 2019                                                          |
| Authors     | Kundan Kumar, Rithesh Kumar, Thibault de Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre de Brebisson, Yoshua Bengio, Aaron Courville |
| Institution | MILA / Université de Montréal / Lyrebird (Descript)            |
| Paper       | "MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis" (arXiv:1910.06711) |

### What It Did

MelGAN was the first to apply **GANs** to the vocoding problem. The generator
was a fully convolutional stack of transposed convolutions that upsampled a
mel-spectrogram to raw waveform. The discriminator was a **multi-scale
discriminator (MSD)** operating at three different audio resolutions (original,
2x downsampled, 4x downsampled). Training used a combination of adversarial loss
and **feature matching loss** from discriminator intermediate layers.

### Key Innovation

- **Fully convolutional generator**: no autoregression, no flows — just blocks
  of transposed convolutions with residual connections and dilated convolutions
- **Multi-scale discriminator (MSD)**: three discriminators operating on audio
  at different downsampling rates, forcing the generator to produce correct
  structure at multiple time-scales
- **Feature matching loss**: L1 distance between discriminator feature maps
  for real vs generated audio, stabilizing GAN training
- **Extremely fast**: ~2500x faster than real-time on GPU

### Problem It Solved

WaveGlow was still relatively slow and large. MelGAN showed you could throw away
all the invertibility constraints, use a simple feed-forward generator with
a GAN training setup, and get **real-time CPU inference** with quality
competitive with flow-based models.

### Limitation

Quality was noticeably below WaveNet/WaveGlow on challenging cases (breathy
voices, high-frequency fricatives). The transposed convolution upsampling often
produced **checkerboard artifacts** and lacked the fine harmonic detail of
autoregressive models. The MSD alone was insufficient to capture both local
periodicity and long-range structure.

---

## 4. HiFi-GAN — Multi-Period Discriminator and Multi-Receptive Field Fusion

| Field       | Detail                                                        |
|-------------|---------------------------------------------------------------|
| Year        | 2020                                                          |
| Authors     | Jungil Kong, Jaehyeon Kim, Jaekyoung Bae                      |
| Institution | Kakao Enterprise / Seoul National University                   |
| Paper       | "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (NeurIPS 2020, arXiv:2010.05646) |

### What It Did

HiFi-GAN built on MelGAN's GAN vocoder blueprint with two crucial improvements:
(1) a **multi-period discriminator (MPD)** that specifically targets the
periodic structure of voiced speech, and (2) a **multi-receptive field fusion
(MRF)** generator block that replaces simple transposed convolutions.

### Key Innovation

- **Multi-Period Discriminator (MPD)**: Reshapes 1D audio into 2D at various
  period lengths (2, 3, 5, 7, 11 samples), then applies 2D convolutions.
  This forces the discriminator to explicitly model the periodic nature of
  voiced speech — each "row" in the reshaped 2D view corresponds to one period.
- **Multi-Receptive Field Fusion (MRF)**: Each upsampling block in the generator
  contains multiple parallel residual blocks with different kernel sizes (3, 7, 11)
  and dilation rates. Their outputs are summed, giving the generator access to
  patterns at multiple temporal scales at each resolution level.
- Combined **MSD + MPD** for complete coverage: MSD captures structure across
  time-scales, MPD captures periodic harmonic structure.

### Problem It Solved

MelGAN suffered from two problems: (a) it couldn't reliably model the harmonic
structure of voiced speech (buzzing, metallic artifacts), and (b) the simple
upsampling blocks couldn't capture multi-scale temporal patterns. HiFi-GAN's
MPD explicitly targeted periodicity, and MRF gave the generator multi-scale
representations at each level, closing the quality gap with autoregressive
models while maintaining MelGAN's speed.

### Limitation

While HiFi-GAN achieved excellent 22 kHz quality, scaling to higher sample
rates (44.1/48 kHz) and larger model sizes revealed **aliasing artifacts**
from transposed convolution upsampling. The model also used standard
activations (LeakyReLU) that are suboptimal for modeling the periodic,
sinusoidal components of audio.

---

## 5. BigVGAN — Anti-Aliased Multi-Rate Upsampling and Periodic Activations

| Field       | Detail                                                        |
|-------------|---------------------------------------------------------------|
| Year        | 2023 (accepted at ICLR 2023)                                  |
| Authors     | Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, Sungroh Yoon |
| Institution | NVIDIA / Seoul National University                            |
| Paper       | "BigVGAN: A Universal Neural Vocoder with Large-Scale Training" (ICLR 2023, arXiv:2206.04658) |

### What It Did

BigVGAN scaled up HiFi-GAN's generator to 112M parameters and introduced two
fundamental architectural innovations: **periodic activations** (Snake/SnakeBeta)
for better modeling of waveforms, and **anti-aliased multi-rate upsampling**
(AMP block) that replaces transposed convolutions with a cascade of
upsample-filter-downsample to eliminate aliasing.

### Key Innovation

- **Snake activation**: `snake(x) = x + (1/alpha) * sin^2(alpha * x)` — a trainable
  periodic activation function that can learn to represent sinusoidal components
  directly, rather than approximating them with ReLU-like piecewise functions.
  The parameter alpha controls frequency.
- **SnakeBeta activation**: `snakebeta(x) = x + (1/beta) * sin^2(alpha * x)` — adds a
  second trainable parameter beta, giving the activation both frequency (alpha) and
  amplitude (beta) control.
- **AMP (Anti-aliased Multi-rate) block**: Inspired by StyleGAN3, replaces
  naive transposed convolution upsampling. The block applies:
  1. Upsample (nearest-neighbor or linear interpolation)
  2. Low-pass filter (Kaiser-windowed sinc, cutoff = output_Nyquist / up_factor)
  3. Convolution
  4. Optionally, a second filter + down/up-sample branch for the residual
  This **eliminates aliasing artifacts** that plagued HiFi-GAN at high sample
  rates, particularly for 44.1/48 kHz output.
- **Massive scale**: 112M parameters (vs HiFi-GAN v1's ~14M), trained on
  thousands of hours across multiple languages, demonstrating universal
  vocoding.

### Problem It Solved

HiFi-GAN's transposed convolutions introduced **checkerboard and aliasing
artifacts** that became severe at higher sample rates and larger model sizes.
Standard activations (LeakyReLU) are inefficient at representing periodic
signals — they need many layers to approximate what a single sine wave does.
BigVGAN's Snake activations give the generator native periodic function
capacity, while AMP blocks produce artifact-free upsampling, enabling clean
48 kHz output at scale.

---

## 6. BigVGAN-v2 — CQT Discriminator and Multi-Scale Mel Loss

| Field       | Detail                                                        |
|-------------|---------------------------------------------------------------|
| Year        | 2024                                                          |
| Authors     | Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, Sungroh Yoon |
| Institution | NVIDIA                                                        |
| Paper       | "BigVGAN-v2: Improved Universal Neural Vocoder" arXiv:2411.01705 |

*BigVGAN-v2 is described in updates to the BigVGAN repository and arXiv preprints.
It represents the second iteration of the BigVGAN architecture.*

### What It Did

BigVGAN-v2 kept the generator architecture (SnakeBeta + AMP blocks) largely
unchanged but significantly improved the training framework with two new
discriminator/loss components: a **Constant-Q Transform (CQT) discriminator**
and a **multi-scale mel-spectrogram reconstruction loss**.

### Key Innovation

- **CQT (Constant-Q Transform) discriminator**: Unlike the STFT-based
  multi-resolution spectrogram discriminators in prior work, the CQT uses
  logarithmically-spaced frequency bins — matching the human auditory system's
  frequency resolution. Low frequencies get narrow bins (high resolution),
  high frequencies get wide bins. This gives the discriminator better
  sensitivity to harmonic structure and inter-harmonic noise across the full
  spectrum. The CQT discards phase information (using magnitude-only) and
  applies 2D convolutions in the time-frequency domain.
- **Multi-scale mel-spectrogram loss**: L1 loss between mel-spectrograms of
  generated and ground-truth audio computed at multiple FFT sizes, hop lengths,
  and mel bin counts. This directly penalizes perceptually-relevant spectral
  errors that adversarial losses alone may miss — particularly formant
  structure and spectral tilt.
- **Combined discriminator suite**: MSD (from MelGAN) + MPD (from HiFi-GAN) +
  MRD (multi-resolution STFT) + CQT discriminator, providing comprehensive
  multi-domain adversarial supervision.

### Problem It Solved

BigVGAN v1 occasionally produced subtle artifacts in harmonic structure at
very high frequencies and struggled with perfect spectral envelope matching.
Standard STFT discriminators have uniform frequency resolution, which is
suboptimal for audio where low-frequency harmonics need finer discrimination
than high-frequency harmonics. The CQT discriminator aligns frequency
resolution with human perception, and the multi-scale mel loss directly
optimizes perceptually-weighted spectral fidelity.

---

## 7. HoliTok — BigVGAN as VAE Decoder in a Discrete Tokenizer

| Field       | Detail                                                        |
|-------------|---------------------------------------------------------------|
| Year        | May 2026                                                      |
| Authors     | (SJTU — Shanghai Jiao Tong University team)                   |
| Institution | Shanghai Jiao Tong University                                 |
| Paper       | "HoliTok: A Holistic Speech Tokenizer" (circa May 2026)       |

### What It Did

HoliTok was the **first architecture to use BigVGAN as the decoder inside a
VAE-based neural audio tokenizer**. Rather than treating BigVGAN as a separate
vocoder driven by mel-spectrograms or acoustic features, HoliTok integrated the
BigVGAN generator as the decoder of a variational autoencoder that compresses
raw audio into discrete latent tokens.

### Key Innovation

- **VAE tokenizer with BigVGAN decoder**: The encoder compresses raw waveform
  into a low-frame-rate latent representation (e.g., 25 Hz with 128-dim
  bottleneck). The decoder is a BigVGAN (v1/v2) generator that reconstructs
  the full-band waveform from these latents.
- **Causal convolutional encoder**: The encoder uses purely causal (or
  streaming-causal) convolutions — no look-ahead — making it suitable for
  real-time streaming audio encoding.
- **LSTM bottleneck**: Between the convolutional encoder and the VAE latent
  space, an LSTM layer provides temporal modeling of the compressed
  representation, capturing longer-range dependencies beyond the conv receptive
  field while maintaining causality.
- **Discrete tokenization via VQ/FSQ**: The continuous VAE latent is quantized
  (e.g., using Finite Scalar Quantization or Vector Quantization) to produce
  discrete speech tokens suitable for language model-based TTS.
- **End-to-end waveform reconstruction**: The BigVGAN decoder operates directly
  on VAE latents, not on mel-spectrograms. This eliminates the information
  bottleneck of mel-spectrogram extraction — the VAE can learn an optimal
  compression for the BigVGAN decoder.

### Problem It Solved

Prior neural audio tokenizers (e.g., EnCodec, SpeechTokenizer, DAC) used
their own custom decoders (often simpler conv-transpose stacks or modified
HiFi-GAN generators). These decoders were specifically designed for each
tokenizer and didn't leverage the massive scale and universal quality of
BigVGAN. HoliTok showed that a BigVGAN decoder could be trained end-to-end
inside a VAE tokenizer, giving discrete tokens that reconstruct to BigVGAN
quality. This bridged the gap between discrete token-based TTS and the
state-of-the-art in waveform generation.

The **causal encoder + LSTM bottleneck** also made the tokenizer suitable
for streaming — the encoder processes audio frame-by-frame without future
context, which is essential for real-time speech applications.

---

## 8. dots.tts — Fully Causal BigVGAN-v2 Adaptation

| Field       | Detail                                                        |
|-------------|---------------------------------------------------------------|
| Year        | June 2026                                                     |
| Authors     | rednote-hilab team                                            |
| Institution | rednote-hilab (Xiaohongshu)                                   |
| Paper       | "dots.tts: A Continuous Autoregressive Text-to-Speech Foundation Model" (arXiv:2606.07080) |

### What It Did

dots.tts took the HoliTok blueprint — a VAE tokenizer with BigVGAN decoder —
and made **the entire AudioVAE (encoder + decoder) fully causal**. This goes
beyond HoliTok's streaming-causal encoder: in dots.tts, the BigVGAN-v2 decoder
itself is also causal, using causal convolutions throughout all upsampling
blocks.

### Key Innovation

- **Fully causal AudioVAE**: Both encoder and decoder use only causal
  convolutions. The encoder has strides [2,2,2,4,6,10] producing a 128-dim
  latent at 25 Hz from 48 kHz raw audio. The decoder upsamples from this
  25 Hz latent back to 48 kHz waveform with zero look-ahead.
- **Causal BigVGAN-v2 decoder**: The AMP upsampling blocks use causal
  convolutions, and the SnakeBeta activations are applied in a causal manner.
  This means the decoder can generate each output sample using only past
  latent frames — **zero algorithmic latency**. As soon as a new latent frame
  arrives (every 40ms at 25 Hz), the decoder can produce the next chunk of
  waveform immediately.
- **Continuous autoregressive (CAR) generation**: The LLM backbone generates
  continuous VAE latents (not discrete tokens) autoregressively at 6.25 Hz,
  and the causal decoder streams them to waveform in real time.
- **48 kHz full-band output**: The decoder outputs directly at 48 kHz, making
  dots.tts a full-band streaming TTS system.

### Problem It Solved

HoliTok proved that a BigVGAN decoder could work inside a VAE tokenizer, but
its decoder was not fully causal — it still had non-causal convolutions in the
upsampling path, requiring future latent frames to generate waveform for the
current frame. This introduced latency and prevented true frame-by-frame
streaming synthesis. For a streaming TTS system where the LLM generates
latents autoregressively, a non-causal decoder would need to buffer multiple
future latent frames before producing audio, defeating the purpose of
low-latency generation.

dots.tts's fully causal adaptation means every component — LLM to semantic
encoder to AudioVAE encoder to AudioVAE decoder — can operate in lockstep:
the LLM produces a latent frame, and the decoder instantly produces the
corresponding waveform chunk with zero look-ahead. This enables **zero-latency
streaming** where audio playback can begin as soon as the first latent frame
is generated.

---

## Summary Table: The Lineage

| Step | Model          | Year | Institution              | Key Innovation                                    | Problem Solved                                        |
|------|----------------|------|--------------------------|---------------------------------------------------|-------------------------------------------------------|
| 1    | WaveNet        | 2016 | DeepMind                 | Dilated causal convs, autoregressive waveform     | First neural model to surpass concatenative TTS       |
| 2    | WaveGlow       | 2018 | NVIDIA                   | Invertible flow, one-pass generation               | WaveNet's speed bottleneck (non-autoregressive)       |
| 3    | MelGAN         | 2019 | MILA/Lyrebird            | GAN training, MSD, fully convolutional             | Flow model complexity; real-time CPU synthesis        |
| 4    | HiFi-GAN       | 2020 | Kakao/SNU                | MPD + MRF generator                                | MelGAN's poor harmonic modeling and artifacts         |
| 5    | BigVGAN        | 2023 | NVIDIA/SNU               | Snake/SnakeBeta activations, AMP anti-aliasing     | Aliasing at high SR; inefficient periodic modeling    |
| 6    | BigVGAN-v2     | 2024 | NVIDIA                   | CQT discriminator, multi-scale mel loss            | Harmonic fidelity; perceptually-tuned spectral loss   |
| 7    | HoliTok        | 2026 | SJTU                     | BigVGAN decoder in VAE tokenizer, causal encoder    | Bridged discrete tokens + SOTA waveform generation    |
| 8    | dots.tts       | 2026 | rednote-hilab            | Fully causal BigVGAN-v2 AudioVAE for streaming     | Zero-latency streaming synthesis from AR latents      |

---

## Architectural Inheritance in dots.tts

The dots.tts AudioVAE decoder inherits directly from this chain:

```
WaveNet gated conv blocks (conceptual ancestor)
  -> WaveGlow WN affine coupling blocks
    -> MelGAN conv-transpose GAN generator
      -> HiFi-GAN MRF upsampling + MPD
        -> BigVGAN SnakeBeta activations + AMP anti-aliased upsampling
          -> BigVGAN-v2 CQT discriminator training + multi-scale mel loss
            -> HoliTok VAE tokenizer integration with causal encoder
              -> dots.tts fully causal decoder adaptation
```

The dots.tts decoder uses BigVGAN-v2 architecture (SnakeBeta + AMP blocks +
CQT training) but with **causal convolutions throughout** for zero-look-ahead
streaming, and is trained as the decoder of a VAE (following HoliTok's approach)
rather than as a standalone mel-spectrogram-to-waveform vocoder.

Specific dots.tts decoder details from the paper (Section 2.2):

- **Input**: 128-dim VAE latent at 25 Hz (from the causal conv encoder)
- **Output**: 48 kHz full-band waveform
- **Encoder strides**: [2,2,2,4,6,10] — total 960x compression (48000 to 50 Hz
  after strided convs, then further down to 25 Hz via the LSTM/VQ bottleneck)
- **Decoder**: BigVGAN-v2 style with causal convolutions in all AMP upsampling
  blocks, trained jointly with the encoder in a VAE framework
