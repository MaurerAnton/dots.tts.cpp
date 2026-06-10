// Test: manual blocks with DiT + PatchEncoder loaded (full pipeline sim)
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "patchenc.h"
#include "dit_manual.h"
#include "safetensors.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <cmath>

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);
bool load_patchenc_weights(SafeTensorsFile & sf, ggml_context * w_ctx, patch_encoder & enc);
void dit_forward_raw(dit_model & model, const float * x, int seq_len, float t_val,
    const float * speaker_emb, float * out);

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";
    SafeTensorsFile sf; sf.open(model_path);
    ggml_init_params wp = { .mem_size = 6ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model m; load_dit_weights(sf, w_ctx, m);
    fprintf(stderr, "DiT loaded, w_ctx=%zu MB\n", ggml_used_mem(w_ctx)/(1024*1024));
    patch_encoder pe; load_patchenc_weights(sf, w_ctx, pe);
    fprintf(stderr, "PE loaded, w_ctx=%zu MB\n", ggml_used_mem(w_ctx)/(1024*1024));
    
    ggml_init_params gp = { .mem_size = 1ULL*1024*1024*1024 };
    ggml_context * gctx = ggml_init(gp);
    fprintf(stderr, "gctx created\n");
    ggml_tensor * ht = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 1536, 2);
    fprintf(stderr, "conditioning...\n"); fflush(stderr);
    for (int i = 0; i < 1536*2; i++) ((float*)ht->data)[i] = 0.1f;
    ggml_tensor * cl = ggml_mul_mat(gctx, m.hidden_proj_w, ht);
    if (m.hidden_proj_b) cl = ggml_add(gctx, cl, m.hidden_proj_b);
    { ggml_cgraph * cgf = ggml_new_graph(gctx); ggml_build_forward_expand(cgf, cl);
      ggml_graph_compute_with_ctx(gctx, cgf, 1); }
    sf.close();

    // Actually call dit_forward_raw with the full pipeline input format
    // like the real pipeline: 8 positions, text + noise, t=0, speaker
    float spk_emb[512] = {0};
    { FILE * f = fopen("speaker_emb.bin", "rb"); if(f){fread(spk_emb,sizeof(float),512,f);fclose(f);} }

    int n_tokens = 8;
    float * x_in = new float[n_tokens * 1024];
    // Fill with same pattern as pipeline: text conditioning + zeros
    float * cond_llm_data = new float[1024*2];
    { float * cd = (float*)((char*)cl->data);
      memcpy(cond_llm_data, cd, 1024*2*sizeof(float)); }
    memset(x_in, 0, n_tokens * 1024 * sizeof(float));
    memcpy(x_in, cond_llm_data, 2 * 1024 * sizeof(float)); // first 2 tokens = text

    printf("Running dit_forward_raw with real data...\n");
    float * out_data = new float[n_tokens * VAE_LATENT_DIM];
    for (int i = 0; i < 10; i++) {  // 10 ODE steps like pipeline
        float t_val = (float)i / 10.0f;
        dit_forward_raw(m, x_in, n_tokens, t_val, spk_emb, out_data);
        float r=0; for(int j=0;j<n_tokens*VAE_LATENT_DIM;j++) r+=out_data[j]*out_data[j];
        printf("  step %d: rms=%.4f\n", i, sqrtf(r/(n_tokens*VAE_LATENT_DIM)));
    }
    printf("SUCCESS\n");

    delete[] x_in; delete[] cond_llm_data; delete[] out_data;
    ggml_free(gctx); ggml_free(w_ctx);
    return 0;
}
