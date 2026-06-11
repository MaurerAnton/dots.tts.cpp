// Test: decode Python latents with C++ BigVGAN
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "bigvgan.h"
#include <cstdio>
#include <cstring>
#include <cmath>

int main() {
    const char * latent_path = "latents_py.bin";
    const char * model_path = "models/vocoder_eff.safetensors";
    const char * out_path = "output_bv_test.wav";
    
    // Load latents
    FILE * f = fopen(latent_path, "rb");
    if (!f) { fprintf(stderr, "MISSING %s\n", latent_path); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    int n_frames = sz / (128 * sizeof(float));
    float * latents = new float[n_frames * 128];
    fread(latents, sizeof(float), n_frames * 128, f);
    fclose(f);
    printf("Loaded %d frames, RMS=%.4f\n", n_frames, 
           [&]{float r=0;for(int i=0;i<n_frames*128;i++)r+=latents[i]*latents[i];return sqrtf(r/(n_frames*128));}());
    
    // Load BigVGAN
    BigVGANDecoder bv;
    if (!bigvgan_load(model_path, bv)) { fprintf(stderr, "FAILED BigVGAN\n"); return 1; }
    
    // Decode
    int n_samples = n_frames * 4 * 48000 / PATCHENC_PATCH_SIZE / 4;  // rough estimate
    float * audio = new float[n_samples];
    bigvgan_decode(bv, latents, n_frames, audio, &n_samples);
    
    // Write WAV
    FILE * wf = fopen(out_path, "wb");
    if (wf) {
        int ds = n_samples * 2, fs = 36 + ds;
        fwrite("RIFF",1,4,wf); fwrite(&fs,4,1,wf); fwrite("WAVE",1,4,wf);
        fwrite("fmt ",1,4,wf); int fz=16; fwrite(&fz,4,1,wf);
        short af=1; fwrite(&af,2,1,wf); short nc=1; fwrite(&nc,2,1,wf);
        int sr=48000; fwrite(&sr,4,1,wf); int br=sr*2; fwrite(&br,4,1,wf);
        short ba=2; fwrite(&ba,2,1,wf); short bp=16; fwrite(&bp,2,1,wf);
        fwrite("data",1,4,wf); fwrite(&ds,4,1,wf);
        for(int i=0;i<n_samples;i++){float s=audio[i]*32767.0f;if(s>32767)s=32767;if(s<-32768)s=-32768;short si=(short)s;fwrite(&si,2,1,wf);}
        fclose(wf);
        printf("Written %s: %d samples\n", out_path, n_samples);
    }
    
    delete[] latents; delete[] audio; bigvgan_free(bv);
    return 0;
}
