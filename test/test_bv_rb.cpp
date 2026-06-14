// Quick BigVGAN resblock RMS test (4 frames)
#include "bigvgan_cpp.h"
#include <cstdio>
#include <cmath>
#include <cstring>

int main() {
    BigVGANDecoder bv;
    if (!bigvgan_load("models/vocoder_eff.safetensors", bv)) {
        fprintf(stderr, "FAILED load\n"); return 1;
    }
    float lat[4*128];
    for(int i=0;i<4*128;i++) lat[i] = (float)(i%13-6)/7.0f * 2.6f;
    int n_samples; float audio[4*1920];
    bigvgan_decode(bv, lat, 4, audio, &n_samples);
    printf("Done: %d samples\n", n_samples);
    bigvgan_free(bv);
    return 0;
}
