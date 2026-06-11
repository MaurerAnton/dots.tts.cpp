// Quick test: compare manual LM layer vs PyTorch on embeddings only
// (too big for 28 layers, test with just embed+first layer RMS)
#include "dots_tts.h"
#include "dots_tts_util.h"  
#include "dit_manual.h"
#include "safetensors.h"
#include <cstdio>
#include <cstring>
#include <cmath>

int main() {
    // Token IDs from Python test
    int seq_len = 2, hidden = 1536;
    int token_ids[2] = {14990, 1879};  // hello, world

    // Load embeddings from safetensors
    const char * path = getenv("DOTS_TTS_MODEL");
    if (!path) path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/model.safetensors";
    
    SafeTensorsFile sf;
    sf.open(path);
    const st_tensor_info * info = sf.find("llm.model.embed_tokens.weight");
    if (!info) { fprintf(stderr, "No embed_tokens\n"); return 1; }
    
    // Read raw bytes from safetensors file
    // Need file offset: sf has data_start_ + info->data_offsets[0]
    // Let me just use ggml to load
    ggml_init_params wp = { .mem_size = 2ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    ggml_tensor * embed_t = sf.load_tensor(w_ctx, *info);
    if (!embed_t) { fprintf(stderr, "FAILED load embed\n"); return 1; }
    sf.close();
    
    float * emb_w = (float*)((char*)embed_t->data + embed_t->view_offs);
    int vocab = (int)embed_t->ne[0];  // ggml dims: ne[0] is fastest
    
    // Get embeddings for hello (14990), world (1879)
    float h_emb[1536], w_emb[1536];
    memcpy(h_emb, emb_w + 14990 * hidden, hidden*sizeof(float));
    memcpy(w_emb, emb_w + 1879 * hidden, hidden*sizeof(float));
    
    float r1=0, r2=0;
    for(int i=0;i<hidden;i++) { r1+=h_emb[i]*h_emb[i]; r2+=w_emb[i]*w_emb[i]; }
    fprintf(stderr, "hello embed RMS=%.4f world embed RMS=%.4f\n", sqrtf(r1/hidden), sqrtf(r2/hidden));
    fprintf(stderr, "hello first5: %.6f %.6f %.6f %.6f %.6f\n", h_emb[0], h_emb[1], h_emb[2], h_emb[3], h_emb[4]);
    fprintf(stderr, "world first5: %.6f %.6f %.6f %.6f %.6f\n", w_emb[0], w_emb[1], w_emb[2], w_emb[3], w_emb[4]);
    
    // Compare with Python
    // Python: hello first5 = [0.00973777, 0.00577415, -0.00011356, 0.00601332, 0.00381825]
    // We should get the same
    
    ggml_free(w_ctx);
    return 0;
}
