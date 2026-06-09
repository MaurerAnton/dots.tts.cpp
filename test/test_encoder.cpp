// Test AudioVAE encoder
#include "audiovae_encoder.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    // Generate test audio: 2 seconds of 48kHz sine + noise
    int sr = 48000;
    int n_samples = 2 * sr;
    float * audio = new float[n_samples];
    for (int i = 0; i < n_samples; i++) {
        float t = (float)i / sr;
        audio[i] = 0.3f * sinf(2.0f * M_PI * 440.0f * t) + 0.02f * ((float)rand()/RAND_MAX - 0.5f);
    }
    printf("Generated %d samples (2s @ 48kHz), RMS=%.4f\n", n_samples, 
           sqrtf([&]{float r=0;for(int i=0;i<n_samples;i++)r+=audio[i]*audio[i];return r/n_samples;}()));
    
    // Load encoder
    const char * sf_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/vocoder.safetensors";
    AudioVAEEncoderWeights w;
    if (!audiovae_encoder_load(sf_path, w)) {
        fprintf(stderr, "Failed to load encoder weights\n");
        return 1;
    }
    
    // Encode
    int max_frames = n_samples / 1920 + 10;
    float * latents = new float[128 * max_frames];
    int n_frames = 0;
    if (!audiovae_encode(w, audio, n_samples, latents, &n_frames)) {
        fprintf(stderr, "Encode failed\n");
        return 1;
    }
    
    // Show stats
    float mean = 0, std = 0;
    for (int i = 0; i < 128 * n_frames; i++) mean += latents[i];
    mean /= (128 * n_frames);
    for (int i = 0; i < 128 * n_frames; i++) std += (latents[i] - mean) * (latents[i] - mean);
    std = sqrtf(std / (128 * n_frames));
    printf("Latents: %d frames x 128, mean=%.4f std=%.4f\n", n_frames, mean, std);
    printf("First 8 dims of frame 0: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", latents[i]);
    printf("\n");
    
    // Check for NaN
    int nan_count = 0;
    for (int i = 0; i < 128 * n_frames; i++) if (std::isnan(latents[i])) nan_count++;
    printf("NaN count: %d / %d\n", nan_count, 128 * n_frames);
    
    // Expected: ~2s * 25Hz = 50 frames
    printf("Expected frames: ~%d\n", n_samples / 1920);
    
    // Save for Python comparison
    FILE * f = fopen("/tmp/encoder_latents.bin", "wb");
    fwrite(latents, sizeof(float), 128 * n_frames, f);
    fclose(f);
    printf("Saved to /tmp/encoder_latents.bin\n");
    
    audiovae_encoder_free(w);
    delete[] audio; delete[] latents;
    return nan_count > 0 ? 1 : 0;
}
