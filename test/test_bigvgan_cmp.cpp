// Test BigVGAN decoder against Python reference
#include "bigvgan_cpp.h"
#include "dots_tts.h"
#include <cstdio>
#include <cmath>

int main() {
    const char * sf_path = "/home/bym/dots.tts.cpp/models/vocoder_eff.safetensors";
    
    // Load reference latents
    float * latents = new float[16 * 128];
    FILE * f = fopen("/tmp/ref_latents.bin", "rb");
    if (!f) { printf("No latents file\n"); return 1; }
    fread(latents, sizeof(float), 16*128, f);
    fclose(f);
    printf("Loaded 16x128 latents\n");
    
    // Load Python reference audio
    float * ref_audio = new float[30720];
    f = fopen("/tmp/ref_audio_py.bin", "rb");
    if (!f) { printf("No ref audio\n"); return 1; }
    int ref_n = fread(ref_audio, sizeof(float), 30720, f);
    fclose(f);
    float ref_rms = 0;
    for (int i = 0; i < ref_n; i++) ref_rms += ref_audio[i]*ref_audio[i];
    ref_rms = sqrtf(ref_rms/ref_n);
    printf("Python ref: %d samples, RMS=%.6f\n", ref_n, ref_rms);
    
    // Load BigVGAN
    BigVGANDecoder dec;
    if (!bigvgan_load(sf_path, dec)) { printf("Load failed\n"); return 1; }
    
    // Decode
    int n_samples;
    float * wav = new float[16 * VAE_HOP_SAMPLES];
    bigvgan_decode(dec, latents, 16, wav, &n_samples);
    
    float cpp_rms = 0;
    for (int i = 0; i < n_samples; i++) cpp_rms += wav[i]*wav[i];
    cpp_rms = sqrtf(cpp_rms/n_samples);
    printf("C++ BigVGAN: %d samples, RMS=%.6f\n", n_samples, cpp_rms);
    
    // Compare
    int n_cmp = n_samples < ref_n ? n_samples : ref_n;
    float max_diff = 0, avg_diff = 0;
    for (int i = 0; i < n_cmp; i++) {
        float d = fabsf(wav[i] - ref_audio[i]);
        avg_diff += d;
        if (d > max_diff) max_diff = d;
    }
    avg_diff /= n_cmp;
    
    // Correlation
    float dot = 0, norm_cpp = 0, norm_ref = 0;
    for (int i = 0; i < n_cmp; i++) {
        dot += wav[i] * ref_audio[i];
        norm_cpp += wav[i] * wav[i];
        norm_ref += ref_audio[i] * ref_audio[i];
    }
    float corr = dot / sqrtf(norm_cpp * norm_ref);
    
    printf("\n=== Comparison ===\n");
    printf("  Samples: C++=%d Py=%d\n", n_samples, ref_n);
    printf("  RMS:     C++=%.6f Py=%.6f\n", cpp_rms, ref_rms);
    printf("  Max diff: %.6f\n", max_diff);
    printf("  Avg diff: %.6f\n", avg_diff);
    printf("  Correlation: %.6f\n", corr);
    printf("  Match: %s\n", corr > 0.99 ? "EXCELLENT" : corr > 0.9 ? "GOOD" : corr > 0.5 ? "FAIR" : "POOR");
    
    // Save C++ output for listening
    f = fopen("/tmp/cpp_bigvgan_out.bin", "wb");
    fwrite(wav, sizeof(float), n_samples, f);
    fclose(f);
    
    bigvgan_free(dec);
    delete[] latents; delete[] wav; delete[] ref_audio;
    return 0;
}
