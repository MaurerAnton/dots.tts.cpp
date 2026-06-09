// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// Standalone BigVGAN vocoder test — loads latents.bin + vocoder.safetensors, outputs WAV
#include "dots_tts.h"
#include "bigvgan_cpp.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    // Load latents from file
    FILE * lf = fopen("latents.bin", "rb");
    if (!lf) { printf("No latents.bin\n"); return 1; }
    fseek(lf, 0, SEEK_END);
    long sz = ftell(lf);
    fseek(lf, 0, SEEK_SET);
    int n_frames = sz / (128 * sizeof(float));
    float * latent = new float[n_frames * 128];
    fread(latent, sizeof(float), n_frames * 128, lf);
    fclose(lf);
    { float rms=0; for(int i=0;i<n_frames*128;i++) rms+=latent[i]*latent[i];
      printf("Latents: %d frames, RMS=%.4f\n", n_frames, sqrtf(rms/(n_frames*128))); }

    BigVGANDecoder dec;
    const char * vp = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/vocoder.safetensors";
    if (!bigvgan_load(vp, dec)) { printf("Load failed\n"); return 1; }

    int n_samples;
    float * wav = new float[n_frames * VAE_HOP_SAMPLES];
    bigvgan_decode(dec, latent, n_frames, wav, &n_samples);

    // Write WAV
    FILE * wf = fopen("output_vocoder.wav", "wb");
    if (wf) {
        int ds = n_samples * 2, fs = 36 + ds;
        fwrite("RIFF",1,4,wf); fwrite(&fs,4,1,wf); fwrite("WAVE",1,4,wf);
        fwrite("fmt ",1,4,wf);
        int fz=16; fwrite(&fz,4,1,wf);
        short af=1; fwrite(&af,2,1,wf); short nc=1; fwrite(&nc,2,1,wf);
        int sr=VAE_SAMPLE_RATE; fwrite(&sr,4,1,wf);
        int br=sr*2; fwrite(&br,4,1,wf); short ba=2; fwrite(&ba,2,1,wf); short bp=16; fwrite(&bp,2,1,wf);
        fwrite("data",1,4,wf); fwrite(&ds,4,1,wf);
        for (int i=0;i<n_samples;i++) {
            float s=wav[i]*32767.0f;
            if(s>32767)s=32767; if(s<-32768)s=-32768;
            short si=(short)s; fwrite(&si,2,1,wf);
        }
        fclose(wf);
        printf("Wrote output_vocoder.wav: %d samples\n", n_samples);
    }

    delete[] wav; delete[] latent;
    bigvgan_free(dec);
    return 0;
}
