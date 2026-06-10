// Test: call manual_dit_block 18 times
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "dit_manual.h"
#include "safetensors.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cmath>

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";
    SafeTensorsFile sf; sf.open(model_path);
    ggml_init_params wp = { .mem_size = 2ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model m; load_dit_weights(sf, w_ctx, m); sf.close();

    // All-zero input like the test
    int n_tokens = 8, hidden = 1024;
    float * x_in = new float[n_tokens * hidden];
    memset(x_in, 0, n_tokens * hidden * sizeof(float));
    // Compute input_layer manually
    float * h = new float[n_tokens * hidden];
    { float * iw = tensor_data(m.input_layer_w); float * ib = m.input_layer_b ? tensor_data(m.input_layer_b) : nullptr;
      for (int t = 0; t < n_tokens; t++) manual_linear(h + t*hidden, x_in + t*hidden, iw, ib, hidden, hidden); }

    // Compute cond
    float cond[1024];
    { float t_val = 0.0f; int half = 128; float * se = new float[256];
      for(int i=0;i<half;i++){ float f=expf(-logf(10000.0f)*(float)i/(float)half); se[i]=cosf(t_val*f); se[half+i]=sinf(t_val*f); }
      float * h1 = new float[1024];
      { float * w1 = tensor_data(m.t_embed_w1); float * b1 = m.t_embed_b1 ? tensor_data(m.t_embed_b1) : nullptr;
        for(int o=0;o<1024;o++){ float s=b1?b1[o]:0.0f; for(int i=0;i<256;i++) s+=w1[o*256+i]*se[i]; h1[o]=s/(1.0f+expf(-s)); } }
      { float * w2 = tensor_data(m.t_embed_w2); float * b2 = m.t_embed_b2 ? tensor_data(m.t_embed_b2) : nullptr;
        for(int o=0;o<1024;o++){ float s=b2?b2[o]:0.0f; for(int i=0;i<1024;i++) s+=w2[o*1024+i]*h1[i]; cond[o]=s; } }
      delete[] se; delete[] h1; }

    printf("Running 18 blocks...\n");
    float * block_out = new float[n_tokens * hidden];
    for (int i = 0; i < 18; i++) {
        manual_dit_block(h, cond, m.layers[i], block_out, n_tokens);
        float * tmp = h; h = block_out; block_out = tmp;
        if (i == 0) { float r=0; for(int j=0;j<n_tokens*hidden;j++) r+=h[j]*h[j];
            printf("  block %d: rms=%.4f\n", i, sqrtf(r/(n_tokens*hidden))); }
    }
    float r=0; for(int j=0;j<n_tokens*hidden;j++) r+=h[j]*h[j];
    printf("  final: rms=%.4f (expected ~21.2 for Python)\n", sqrtf(r/(n_tokens*hidden)));

    delete[] x_in; delete[] h; delete[] block_out;
    ggml_free(w_ctx);
    return 0;
}
