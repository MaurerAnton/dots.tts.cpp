// SPDX-License-Identifier: GPL-3.0-or-later
// Quick BigVGAN test: 4 frames
#include "bigvgan_cpp.h"
#include <cstdio>
#include <cmath>
#include <cstring>

int main() {
    BigVGANDecoder bv;
    if (!bigvgan_load("models/vocoder_eff.safetensors", bv)) {
        fprintf(stderr, "FAILED load\n"); return 1;
    }
    printf("Loaded. Testing...\n");
    
    // 4 frames of deterministic test latents
    float lat[4*128];
    for(int i=0;i<4*128;i++) lat[i] = (float)(i%13-6)/7.0f * 2.6f;
    // Save input for Python
    { FILE * f = fopen("/tmp/cpp_bv_input.bin", "wb"); fwrite(lat, sizeof(float), 4*128, f); fclose(f); }
    
    int n_samples;
    float audio[4*1920];
    bigvgan_decode(bv, lat, 4, audio, &n_samples);
    
    printf("Output: %d samples, first10: ", n_samples);
    for(int i=0;i<10;i++) printf("%.4f ", audio[i]);
    printf("\nDone.\n");
    
    bigvgan_free(bv);
    return 0;
}
