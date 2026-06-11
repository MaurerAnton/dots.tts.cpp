// SPDX-License-Identifier: GPL-3.0-or-later
// Test: C++ PE vs Python PE on same input
#include "patchenc.h"
#include "safetensors.h"
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "ggml.h"
#include "ggml-cpu.h"
bool load_patchenc_weights(SafeTensorsFile & sf, ggml_context * w_ctx, patch_encoder & enc);
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/model.safetensors";
    
    SafeTensorsFile sf;
    if (!sf.open(model_path)) { fprintf(stderr, "FAILED open\n"); return 1; }
    
    ggml_init_params gparams = { .mem_size = 2048ULL*1024*1024 };
    ggml_context * w_ctx = ggml_init(gparams);
    
    patch_encoder pe;
    if (!load_patchenc_weights(sf, w_ctx, pe)) { fprintf(stderr, "FAILED pe load\n"); return 1; }
    sf.close();
    
    // Load Python's PE input
    float input[512];
    FILE * f = fopen("/tmp/pe_python_input.bin", "rb");
    if (!f) { fprintf(stderr, "No input file\n"); return 1; }
    fread(input, sizeof(float), 512, f); fclose(f);
    
    ggml_init_params cparams = { .mem_size = 256ULL*1024*1024 };
    ggml_context * ctx = ggml_init(cparams);
    
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PATCHENC_LATENT_DIM, PATCHENC_PATCH_SIZE);
    memcpy(tensor_data(x), input, 512*sizeof(float));
    
    // Run C++ PE and compute graph
    ggml_tensor * out = patchenc_forward(pe, ctx, x, 1);
    ggml_cgraph * cg = ggml_new_graph(ctx);
    ggml_build_forward_expand(cg, out);
    ggml_graph_compute_with_ctx(ctx, cg, 8);
    
    float * od = tensor_data(out);
    float rms=0; for(int i=0;i<1536;i++) rms += od[i]*od[i];
    rms = sqrtf(rms/1536);
    printf("C++ PE output RMS=%.6f  first10:", rms);
    for(int i=0;i<10;i++) printf(" %.6f", od[i]);
    printf("\n");
    
    // Compare with Python
    f = fopen("/tmp/pe_python_output.bin", "rb");
    if (f) {
        float py_out[1536];
        fread(py_out, sizeof(float), 1536, f); fclose(f);
        float py_rms=0, cross=0, diff_rms=0;
        for(int i=0;i<1536;i++) {
            py_rms += py_out[i]*py_out[i];
            cross += od[i] * py_out[i];
            float d = od[i] - py_out[i];
            diff_rms += d*d;
        }
        py_rms = sqrtf(py_rms/1536); diff_rms = sqrtf(diff_rms/1536);
        float corr = cross/1536 / rms / py_rms;
        printf("Python PE output RMS=%.6f  first10:", py_rms);
        for(int i=0;i<10;i++) printf(" %.6f", py_out[i]);
        printf("\n");
        printf("Corr=%.6f  Diff RMS=%.6f  Ratio=%.4f\n", corr, diff_rms, rms/py_rms);
        
        if (corr > 0.999f) printf("C++ PE MATCHES PYTHON!\n");
        else printf("MISMATCH (corr=%.6f)\n", corr);
    }
    
    ggml_free(ctx); ggml_free(w_ctx);
    return 0;
}
