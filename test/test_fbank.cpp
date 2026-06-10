// Debug CAM++ FBank extraction
#include "campp.h"
#include <cstdio>
#include <cmath>

int main() {
    FILE * f = fopen("/tmp/test_speaker.pcm", "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    int n_samples = ftell(f) / sizeof(float);
    fseek(f, 0, SEEK_SET);
    float * audio = new float[n_samples];
    fread(audio, sizeof(float), n_samples, f);
    fclose(f);
    printf("Loaded %d samples\n", n_samples);
    
    // Show audio stats
    float amean = 0, astd = 0;
    for (int i = 0; i < n_samples; i++) amean += audio[i];
    amean /= n_samples;
    for (int i = 0; i < n_samples; i++) astd += (audio[i]-amean)*(audio[i]-amean);
    astd = sqrtf(astd/n_samples);
    printf("Audio: mean=%.6f std=%.6f\n", amean, astd);
    
    // Manual FBank test
    int n_mels = 80, n_fft = 512;
    int frame_len = 400; // 25ms @ 16kHz
    int frame_shift = 160; // 10ms
    int n_frames = (n_samples - frame_len) / frame_shift + 1;
    printf("Frames: %d\n", n_frames);
    
    // Hamming window
    float window[400];
    for (int i = 0; i < frame_len; i++)
        window[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (frame_len - 1));
    
    // Extract first frame FBank to test
    float frame[400], mag[257];
    for (int i = 0; i < frame_len; i++) frame[i] = audio[i] * window[i];
    
    // DFT
    for (int k = 0; k <= n_fft/2; k++) {
        float re = 0, im = 0;
        for (int n = 0; n < frame_len; n++) {
            float angle = -2.0f * M_PI * k * n / n_fft;
            re += frame[n] * cosf(angle);
            im += frame[n] * sinf(angle);
        }
        mag[k] = sqrtf(re*re + im*im) / sqrtf((float)n_fft);
    }
    
    printf("First frame mag: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", mag[i]);
    printf("... rms=%.6f\n", sqrtf(mag[10]*mag[10]));
    
    // Check if any NaN in mag
    for (int i = 0; i <= n_fft/2; i++)
        if (std::isnan(mag[i])) { printf("NaN in mag[%d]!\n", i); break; }
    
    delete[] audio;
    return 0;
}
