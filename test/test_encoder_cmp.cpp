// Test AudioVAE encoder vs Python reference
#include "audiovae_encoder.h"
#include <cstdio>
#include <cmath>

int main() {
    const char * sf_path = "/home/bym/dots.tts.cpp/models/encoder_eff.safetensors";
    
    // Load test audio
    float * audio = new float[48000];
    FILE * f = fopen("/tmp/enc_test_audio.bin", "rb");
    if (!f) { printf("No audio\n"); return 1; }
    fread(audio, sizeof(float), 48000, f);
    fclose(f);
    
    // Load Python reference latents
    float * ref_latents = new float[25 * 128];
    f = fopen("/tmp/enc_ref_latents.bin", "rb");
    if (!f) { printf("No ref latents\n"); return 1; }
    fread(ref_latents, sizeof(float), 25 * 128, f);
    fclose(f);
    float ref_rms = 0;
    for (int i = 0; i < 25*128; i++) ref_rms += ref_latents[i]*ref_latents[i];
    ref_rms = sqrtf(ref_rms/(25*128));
    printf("Python ref: 25 frames, RMS=%.6f\n", ref_rms);
    
    // Load encoder
    AudioVAEEncoderWeights w;
    if (!audiovae_encoder_load(sf_path, w)) { printf("Load fail\n"); return 1; }
    
    // Encode
    float * latents = new float[50 * 128]; // overallocate
    int n_frames;
    audiovae_encode(w, audio, 48000, latents, &n_frames);
    
    float cpp_rms = 0;
    for (int i = 0; i < n_frames*128; i++) cpp_rms += latents[i]*latents[i];
    cpp_rms = sqrtf(cpp_rms/(n_frames*128));
    printf("C++ encoder: %d frames, RMS=%.6f\n", n_frames, cpp_rms);
    
    // Compare
    int n_cmp = n_frames < 25 ? n_frames : 25;
    float max_diff = 0, avg_diff = 0;
    for (int i = 0; i < n_cmp * 128; i++) {
        float d = fabsf(latents[i] - ref_latents[i]);
        avg_diff += d;
        if (d > max_diff) max_diff = d;
    }
    avg_diff /= (n_cmp * 128);
    
    // Per-channel correlation
    float dot = 0, nc=0, nr=0;
    for (int i = 0; i < n_cmp*128; i++) {
        dot += latents[i] * ref_latents[i];
        nc += latents[i] * latents[i];
        nr += ref_latents[i] * ref_latents[i];
    }
    float corr = dot / sqrtf(nc * nr);
    
    printf("\n=== Encoder Comparison ===\n");
    printf("  Frames: C++=%d Py=25\n", n_frames);
    printf("  RMS:    C++=%.6f Py=%.6f\n", cpp_rms, ref_rms);
    printf("  Max diff: %.6f\n", max_diff);
    printf("  Avg diff: %.6f\n", avg_diff);
    printf("  Correlation: %.6f\n", corr);
    printf("  Match: %s\n", corr > 0.99 ? "EXCELLENT" : corr > 0.9 ? "GOOD" : corr > 0.5 ? "FAIR" : "POOR");
    
    audiovae_encoder_free(w);
    delete[] audio; delete[] latents; delete[] ref_latents;
    return 0;
}
