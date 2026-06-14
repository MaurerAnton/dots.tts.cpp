#include "bigvgan_cpp.h"
#include <cstdio>
#include <cstring>
int main() {
    BigVGANDecoder bv;
    if (!bigvgan_load("models/vocoder_eff.safetensors", bv)) return 1;
    float lat[4*128];
    FILE* f=fopen("debug/lat_s9.bin","rb"); fread(lat,sizeof(float),4*128,f); fclose(f);
    int ns; float audio[4*1920];
    bigvgan_decode(bv, lat, 4, audio, &ns);
    printf("rb0_rms=%.4f post_proj=...\n",0.0);
    bigvgan_free(bv); return 0;
}
