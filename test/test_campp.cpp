// Test CAM++ speaker encoder
#include "campp.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    // Read test audio
    FILE * f = fopen("/tmp/test_speaker.pcm", "rb");
    if (!f) { fprintf(stderr, "No test file\n"); return 1; }
    fseek(f, 0, SEEK_END);
    int n_samples = ftell(f) / sizeof(float);
    fseek(f, 0, SEEK_SET);
    float * audio = new float[n_samples];
    fread(audio, sizeof(float), n_samples, f);
    fclose(f);
    printf("Loaded %d samples\n", n_samples);
    
    // Load CAM++ weights
    const char * sf_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/speaker_encoder.safetensors";
    
    CAMPPWeights wp;
    if (!campp_load(sf_path, wp)) {
        fprintf(stderr, "Failed to load CAM++ weights\n");
        return 1;
    }
    
    // Extract embedding
    float embedding[512];
    if (!campp_extract(wp, audio, n_samples, embedding)) {
        fprintf(stderr, "Failed to extract embedding (audio too short?)\n");
        return 1;
    }
    
    // Print stats
    float mean = 0, std = 0;
    for (int i = 0; i < 512; i++) mean += embedding[i];
    mean /= 512;
    for (int i = 0; i < 512; i++) std += (embedding[i] - mean) * (embedding[i] - mean);
    std = sqrtf(std / 512);
    
    printf("Speaker embedding (512-dim):\n");
    printf("  Mean: %.6f\n", mean);
    printf("  Std:  %.6f\n", std);
    printf("  Min:  %.6f\n", embedding[0]); // just show first few
    printf("  Max:  %.6f\n", embedding[0]);
    for (int i = 0; i < 512; i++) {
        if (embedding[i] < embedding[0]) ((float*)&embedding[0])[0] = embedding[i]; // hack: find min
        // actually just dump first 8
    }
    printf("  First 8: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", embedding[i]);
    printf("\n");
    
    // Find actual min/max
    float mn = embedding[0], mx = embedding[0];
    for (int i = 0; i < 512; i++) {
        if (embedding[i] < mn) mn = embedding[i];
        if (embedding[i] > mx) mx = embedding[i];
    }
    printf("  Range: [%.4f, %.4f]\n", mn, mx);
    
    // Compare with Python reference if available
    printf("\nDone. Embedding saved to /tmp/campp_embedding.bin\n");
    f = fopen("/tmp/campp_embedding.bin", "wb");
    fwrite(embedding, sizeof(float), 512, f);
    fclose(f);
    
    campp_free(wp);
    delete[] audio;
    return 0;
}
