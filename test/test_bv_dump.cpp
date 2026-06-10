// Dump BigVGAN stage intermediates for Python comparison
#include "dots_tts.h"
#include "bigvgan_cpp.h"
#include "safetensors.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>

bool load_bigvgan_cpp(SafeTensorsFile & sf, BigVGANDecoder & dec);

static void write_float_bin(const char* path, const float* data, int n) {
    FILE* f = fopen(path, "wb"); fwrite(data, sizeof(float), n, f); fclose(f);
}

int main() {
    const char* model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";
    
    SafeTensorsFile sf;
    if (!sf.open(model_path)) { fprintf(stderr, "Failed to open %s\n", model_path); return 1; }
    
    BigVGANDecoder dec;
    if (!load_bigvgan_cpp(sf, dec)) { fprintf(stderr, "Failed to load BigVGAN\n"); return 1; }
    sf.close();
    
    // Load C++ latents
    float* latents = (float*)malloc(128 * 128 * sizeof(float));
    int n_frames = 0;
    {
        FILE* f = fopen("build/latents.bin", "rb");
        if (!f) f = fopen("latents.bin", "rb");
        if (!f) { fprintf(stderr, "No latents file\n"); return 1; }
        n_frames = 0;
        while (fread(latents + n_frames * 128, sizeof(float), 128, f) == 128) n_frames++;
        fclose(f);
    }
    fprintf(stderr, "Loaded %d latent frames\n", n_frames);
    
    // Allocate output buffers
    int max_len = n_frames * 1920;
    float* audio = (float*)malloc(max_len * sizeof(float));
    int n_samples = max_len;
    
    // Call modified decoder that dumps intermediates
    // Actually, the decoder already prints RMS. Let me just capture those.
    bigvgan_decode(dec, latents, n_frames, audio, &n_samples);
    
    // Write final audio
    write_float_bin("debug/cpp_bv_audio.bin", audio, n_samples);
    fprintf(stderr, "Wrote %d audio samples to debug/cpp_bv_audio.bin\n", n_samples);
    
    free(latents); free(audio);
    return 0;
}
