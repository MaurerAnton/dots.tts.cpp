// SPDX-License-Identifier: GPL-3.0-or-later
// Component test: coord_proj (Linear 128->1024) C++ vs Python
#include "safetensors.h"
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <cmath>

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/model.safetensors";
    
    SafeTensorsFile sf;
    if (!sf.open(model_path)) { fprintf(stderr, "FAILED open\n"); return 1; }
    ggml_init_params wp = { .mem_size = 3ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model dit;
    if (!load_dit_weights(sf, w_ctx, dit)) { fprintf(stderr, "FAILED dit\n"); return 1; }
    sf.close();
    
    // Load Python noise for call 0
    float noise[128];
    FILE * f = fopen("py_noise_call0.bin", "rb");
    if (!f) { fprintf(stderr, "No noise file\n"); return 1; }
    fread(noise, sizeof(float), 128, f);
    fclose(f);
    
    // Run coord_proj via ggml_mul_mat
    ggml_tensor * nv = ggml_new_tensor_2d(w_ctx, GGML_TYPE_F32, 128, 1);
    memcpy(tensor_data(nv), noise, 128 * sizeof(float));
    ggml_tensor * proj = ggml_mul_mat(w_ctx, dit.coord_proj_w, nv);
    if (dit.coord_proj_b) proj = ggml_add(w_ctx, proj, dit.coord_proj_b);
    { ggml_cgraph * cg = ggml_new_graph(w_ctx); ggml_build_forward_expand(cg, proj);
      ggml_graph_compute_with_ctx(w_ctx, cg, 1); }
    
    float * out = tensor_data(proj);
    float rms=0; for(int i=0;i<1024;i++) rms+=out[i]*out[i];
    printf("C++ coord_proj output: RMS=%.6f first5=%.6f %.6f %.6f %.6f %.6f\n",
           sqrtf(rms/1024), out[0],out[1],out[2],out[3],out[4]);
    
    // Save for Python comparison
    f = fopen("/tmp/cpp_coord_proj.bin", "wb");
    fwrite(out, sizeof(float), 1024, f); fclose(f);
    f = fopen("/tmp/coord_noise.bin", "wb");
    fwrite(noise, sizeof(float), 128, f); fclose(f);
    
    ggml_free(w_ctx);
    return 0;
}
