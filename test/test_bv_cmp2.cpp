// SPDX-License-Identifier: GPL-3.0-or-later
// BigVGAN comparison: C++ vs Python reference
#include "bigvgan_cpp.h"
#include <cstdio>
#include <cmath>
#include <cstring>

int main() {
    BigVGANDecoder bv;
    if (!bigvgan_load("models/vocoder_eff.safetensors", bv)) {
        fprintf(stderr, "FAILED load\n"); return 1;
    }
    
    // Load Python reference input (28 frames x 128)
    float lat[28*128];
    FILE * f = fopen("/tmp/bv_ref_input.bin", "rb");
    if (!f) { fprintf(stderr, "No ref input\n"); return 1; }
    fread(lat, sizeof(float), 28*128, f); fclose(f);
    
    // Run C++ BigVGAN
    int n_samples;
    float audio[28*1920];
    bigvgan_decode(bv, lat, 28, audio, &n_samples);
    
    // Load Python reference output
    float ref[28*1920];
    f = fopen("/tmp/bv_ref_output.bin", "rb");
    if (!f) { fprintf(stderr, "No ref output\n"); return 1; }
    fread(ref, sizeof(float), n_samples, f); fclose(f);
    
    // Compare
    float maxd=0, rms_c=0, rms_p=0, cross=0;
    for(int i=0;i<n_samples;i++) {
        float d = fabsf(audio[i] - ref[i]);
        if(d > maxd) maxd = d;
        rms_c += audio[i]*audio[i];
        rms_p += ref[i]*ref[i];
        cross += audio[i]*ref[i];
    }
    rms_c = sqrtf(rms_c/n_samples);
    rms_p = sqrtf(rms_p/n_samples);
    float corr = cross / n_samples / (rms_c * rms_p);
    float gain = rms_p / rms_c;
    
    printf("C++ RMS=%.6f Py RMS=%.6f gain=tanh*%.3f corr=%.6f max_diff=%.6f\n",
           rms_c, rms_p, gain, corr, maxd);
    
    if(corr > 0.999f) printf("BIGVGAN MATCHES PYTHON (corr=%.4f)\n", corr);
    else printf("BIGVGAN DIFFERS: need gain=%.3f at tanh\n", gain);
    
    bigvgan_free(bv);
    return 0;
}
