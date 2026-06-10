// Quick dump: post_proj output from C++ for Python comparison
#include "dots_tts.h"
#include "bigvgan_cpp.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

// Declare conv1d_causal from bigvgan_cpp.cpp
void conv1d_causal(float*,const float*,int,int,const float*,const float*,int,int,int=1);

int main() {
    BigVGANDecoder dec;
    if (!bigvgan_load("models/vocoder_eff.safetensors", dec)) return 1;
    
    float latents[128*128]; int nf=0;
    FILE* f = fopen("build/latents.bin","rb");
    while(fread(latents+nf*128,sizeof(float),128,f)==128) nf++;
    fclose(f);
    
    float* buf = (float*)malloc(nf*256*sizeof(float));
    conv1d_causal(buf, latents, 128, nf, dec.post_proj_w.ptr(), dec.post_proj_b.ptr(), 128, 1);
    float r=0; for(int i=0;i<nf*128;i++) r+=buf[i]*buf[i];
    printf("C++ post_proj RMS: %.4f\n", sqrtf(r/(nf*128)));
    
    f=fopen("debug/cpp_post_proj.bin","wb"); fwrite(buf,sizeof(float),nf*128,f); fclose(f);
    free(buf);
    return 0;
}
