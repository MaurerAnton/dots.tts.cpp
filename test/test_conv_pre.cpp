// Test: compare C++ conv1d_causal vs Python for conv_pre
#include "dots_tts.h"
#include "bigvgan_cpp.h"
#include "safetensors.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>

bool load_bigvgan_cpp(SafeTensorsFile & sf, BigVGANDecoder & dec);

int main() {
    const char* model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";
    
    // Actually, bigvgan_load uses vocoder_eff.safetensors
    BigVGANDecoder dec;
    if (!bigvgan_load("models/vocoder_eff.safetensors", dec)) return 1;
    
    // Generate dummy input matching bottleneck output shape
    int ic = 128, ilen = 4;
    float * in = (float*)malloc(ic * ilen * sizeof(float));
    // Use known values for reproducibility
    for (int i = 0; i < ic*ilen; i++) in[i] = ((float)(i % 100) / 100.0f - 0.5f) * 5.38f;  // RMS ~2.69
    
    float * out = (float*)malloc(1536 * ilen * sizeof(float));
    
    // C++ conv1d_causal
    conv1d_causal(out, in, ic, ilen, dec.conv_pre_w.ptr(), dec.conv_pre_b.ptr(), 1536, 5);
    
    // Print output RMS and first few values
    float rms = 0;
    for (int i = 0; i < 1536*ilen; i++) rms += out[i]*out[i];
    rms = sqrtf(rms / (1536*ilen));
    printf("C++ conv_pre output RMS: %.4f\n", rms);
    printf("first5: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", out[0], out[1], out[2], out[3], out[4]);
    
    // Write input and output for Python comparison
    FILE* f = fopen("debug/cpp_conv_pre_in.bin", "wb");
    fwrite(in, sizeof(float), ic*ilen, f); fclose(f);
    f = fopen("debug/cpp_conv_pre_out.bin", "wb");
    fwrite(out, sizeof(float), 1536*ilen, f); fclose(f);
    printf("Dumped to debug/cpp_conv_pre_{in,out}.bin\n");
    
    free(in); free(out);
    return 0;
}
