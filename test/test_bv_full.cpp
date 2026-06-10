// Run full BigVGAN decoder on stored latents and print per-stage RMS
#include "bigvgan_cpp.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

void conv1d_causal(float*,const float*,int,int,const float*,const float*,int,int,int=1);
void lstm_forward(const float*,int,int,int,const float**,const float**,const float**,const float**,float*,bool);

int main() {
    BigVGANDecoder dec;
    if (!bigvgan_load("models/vocoder_eff.safetensors", dec)) return 1;
    
    float latents[128*128]; int nf=0;
    FILE* f = fopen("build/latents.bin","rb");
    while(fread(latents+nf*128,sizeof(float),128,f)==128) nf++;
    fclose(f);
    
    float* audio = (float*)malloc(nf * 1920 * sizeof(float));
    int nsamp = nf * 1920;
    bigvgan_decode(dec, latents, nf, audio, &nsamp);
    // Write WAV
    FILE* wf = fopen("debug/cpp_bv_audio.wav", "wb");
    if (wf) {
        int ds = nsamp * 2, fs = 36 + ds;
        fwrite("RIFF",1,4,wf); fwrite(&fs,4,1,wf); fwrite("WAVE",1,4,wf);
        fwrite("fmt ",1,4,wf); int fz=16; fwrite(&fz,4,1,wf);
        short af=1; fwrite(&af,2,1,wf); short nc=1; fwrite(&nc,2,1,wf);
        int sr=48000; fwrite(&sr,4,1,wf); int br=sr*2; fwrite(&br,4,1,wf);
        short ba=2; fwrite(&ba,2,1,wf); short bp=16; fwrite(&bp,2,1,wf);
        fwrite("data",1,4,wf); fwrite(&ds,4,1,wf);
        for(int i=0;i<nsamp;i++){float s=audio[i]*32767.0f;if(s>32767)s=32767;if(s<-32768)s=-32768;short si=(short)s;fwrite(&si,2,1,wf);}
        fclose(wf);
    }
    printf("Wrote %d samples to debug/cpp_bv_audio.wav\n", nsamp);
    
    free(audio);
    return 0;
}
