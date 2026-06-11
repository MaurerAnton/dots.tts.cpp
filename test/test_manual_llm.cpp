// Manual LLM forward pass — compare byte-by-byte with PyTorch
// Loads all 28 layers of LM weights as raw float arrays
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit_manual.h"
#include "safetensors.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>

static float* load_sf_tensor(SafeTensorsFile & sf, const char * name, size_t & n_elems) {
    const st_tensor_info * info = sf.find(name);
    if (!info) { n_elems = 0; return nullptr; }
    n_elems = 1;
    for (size_t d : info->shape) n_elems *= d;
    float * buf = (float*)malloc(n_elems * sizeof(float));
    if (!buf) { fprintf(stderr, "OOM loading %s (%zu floats)\n", name, n_elems); return nullptr; }
    sf.load_raw(*info, buf, n_elems);
    return buf;
}

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/model.safetensors";
    
    SafeTensorsFile sf;
    if (!sf.open(model_path)) { fprintf(stderr, "FAILED open\n"); return 1; }
    
    fprintf(stderr, "Loading LLM weights...\n");
    const int N = 28, H = 1536, F = 8960, NH = 12, NKV = 2, HD = 128;
    
    // Embeddings
    size_t ne;
    float * embed = load_sf_tensor(sf, "llm.model.embed_tokens.weight", ne);
    if (!embed) { fprintf(stderr, "FAILED embed\n"); return 1; }
    // embed shape: [151672, 1536] row-major
    
    // Layer weights
    struct LayerW { float *an, *fn, *qw, *qb, *kw, *kb, *vw, *vb, *ow, *ob, *gw, *uw, *dw; };
    std::vector<LayerW> layers(N);
    for (int i = 0; i < N; i++) {
        char name[512]; LayerW & l = layers[i];
        snprintf(name, sizeof(name), "llm.model.layers.%d.input_layernorm.weight", i);
        l.an = load_sf_tensor(sf, name, ne);
        snprintf(name, sizeof(name), "llm.model.layers.%d.post_attention_layernorm.weight", i);
        l.fn = load_sf_tensor(sf, name, ne);
        snprintf(name, sizeof(name), "llm.model.layers.%d.self_attn.q_proj.weight", i);
        l.qw = load_sf_tensor(sf, name, ne);
        snprintf(name, sizeof(name), "llm.model.layers.%d.self_attn.q_proj.bias", i);
        l.qb = load_sf_tensor(sf, name, ne);
        snprintf(name, sizeof(name), "llm.model.layers.%d.self_attn.k_proj.weight", i);
        l.kw = load_sf_tensor(sf, name, ne);
        snprintf(name, sizeof(name), "llm.model.layers.%d.self_attn.k_proj.bias", i);
        l.kb = load_sf_tensor(sf, name, ne);
        snprintf(name, sizeof(name), "llm.model.layers.%d.self_attn.v_proj.weight", i);
        l.vw = load_sf_tensor(sf, name, ne);
        snprintf(name, sizeof(name), "llm.model.layers.%d.self_attn.v_proj.bias", i);
        l.vb = load_sf_tensor(sf, name, ne);
        snprintf(name, sizeof(name), "llm.model.layers.%d.self_attn.o_proj.weight", i);
        l.ow = load_sf_tensor(sf, name, ne);
        snprintf(name, sizeof(name), "llm.model.layers.%d.self_attn.o_proj.bias", i);
        l.ob = load_sf_tensor(sf, name, ne);
        snprintf(name, sizeof(name), "llm.model.layers.%d.mlp.gate_proj.weight", i);
        l.gw = load_sf_tensor(sf, name, ne);
        snprintf(name, sizeof(name), "llm.model.layers.%d.mlp.up_proj.weight", i);
        l.uw = load_sf_tensor(sf, name, ne);
        snprintf(name, sizeof(name), "llm.model.layers.%d.mlp.down_proj.weight", i);
        l.dw = load_sf_tensor(sf, name, ne);
    }
    float * final_norm = load_sf_tensor(sf, "llm.model.norm.weight", ne);
    sf.close();
    fprintf(stderr, "Loaded %d layers\n", N);
    
    // Token IDs (same as Python test)
    int seq_len = 11, token_ids[] = {58, 108704, 60, 14990, 1879, 58, 108704, 103124, 105761, 60, 151668};
    
    // Embeddings
    float * h = (float*)malloc(seq_len * H * sizeof(float));
    float * h2 = (float*)malloc(seq_len * H * sizeof(float));
    for (int t = 0; t < seq_len; t++)
        memcpy(h + t*H, embed + token_ids[t]*H, H*sizeof(float));
    { float r=0; for(int i=0;i<seq_len*H;i++) r+=h[i]*h[i];
      fprintf(stderr, "embed RMS=%.4f\n", sqrtf(r/(seq_len*H))); }
    
    // Buffers (reused across layers)
    float * normed = (float*)malloc(seq_len * H * sizeof(float));
    int kvdim = NKV * HD;  // 256 for 2 KV heads
    float * q = (float*)malloc(seq_len * H * sizeof(float));
    float * k = (float*)malloc(seq_len * kvdim * sizeof(float));
    float * v = (float*)malloc(seq_len * kvdim * sizeof(float));
    float * ao = (float*)malloc(seq_len * H * sizeof(float));
    float * attn_out = (float*)malloc(seq_len * H * sizeof(float));
    float * gate = (float*)malloc(seq_len * F * sizeof(float));
    float * up = (float*)malloc(seq_len * F * sizeof(float));
    float * mlp_out = (float*)malloc(seq_len * H * sizeof(float));
    float * scores = (float*)malloc(seq_len * seq_len * sizeof(float));
    float * qh = (float*)malloc(seq_len * HD * sizeof(float));
    float * kh = (float*)malloc(seq_len * HD * sizeof(float));
    float * vh = (float*)malloc(seq_len * HD * sizeof(float));
    
    float scale = 1.0f / sqrtf((float)HD);
    
    for (int i = 0; i < N; i++) {
        LayerW & l = layers[i];
        
        // RMSNorm
        for (int t = 0; t < seq_len; t++)
            manual_rms_norm(normed + t*H, h + t*H, l.an, H);
        
        // QKV projections (K/V: 256 output, Q: 1536 output)
        static int once=0; if(!once){fprintf(stderr,"GQA_FIX_ACTIVE kvdim=%d\n",kvdim);once=1;}
        for (int t = 0; t < seq_len; t++) {
            manual_linear(q + t*H, normed + t*H, l.qw, l.qb, H, H);
            manual_linear(k + t*kvdim, normed + t*H, l.kw, l.kb, H, kvdim);
            manual_linear(v + t*kvdim, normed + t*H, l.vw, l.vb, H, kvdim);
        }
        
        // Multi-head attention
        memset(ao, 0, seq_len * H * sizeof(float));
        for (int head = 0; head < NH; head++) {
            int kv_h = head * NKV / NH;
            for (int t = 0; t < seq_len; t++) for (int d = 0; d < HD; d++) {
                qh[t*HD+d] = q[t*H + head*HD + d];
                kh[t*HD+d] = k[t*kvdim + kv_h*HD + d];
                vh[t*HD+d] = v[t*kvdim + kv_h*HD + d];
            }
            manual_rope_theta(qh, kh, seq_len, HD, 1000000.0f);  // Mistral/Qwen2 RoPE theta
            
            // Scores + causal mask + softmax
            for (int ti = 0; ti < seq_len; ti++) {
                for (int tj = 0; tj < seq_len; tj++) {
                    float s = 0; for (int d = 0; d < HD; d++) s += qh[ti*HD+d] * kh[tj*HD+d];
                    scores[ti*seq_len+tj] = s * scale;
                }
                for (int tj = ti+1; tj < seq_len; tj++) scores[ti*seq_len+tj] = -INFINITY;
                manual_softmax(scores + ti*seq_len, seq_len);
            }
            // Weighted sum of V
            for (int ti = 0; ti < seq_len; ti++)
                for (int d = 0; d < HD; d++) {
                    float s = 0; for (int tj = 0; tj < seq_len; tj++) s += scores[ti*seq_len+tj] * vh[tj*HD+d];
                    ao[ti*H + head*HD + d] = s;
                }
        }
        
        // O projection
        for (int t = 0; t < seq_len; t++)
            manual_linear(attn_out + t*H, ao + t*H, l.ow, l.ob, H, H);
        
        // Residual
        for (int j = 0; j < seq_len*H; j++) h2[j] = h[j] + attn_out[j];
        
        // RMSNorm before MLP
        for (int t = 0; t < seq_len; t++)
            manual_rms_norm(normed + t*H, h2 + t*H, l.fn, H);
        
        // MLP: gate_proj → SiLU, up_proj → multiply, down_proj
        for (int t = 0; t < seq_len; t++) {
            manual_linear(gate + t*F, normed + t*H, l.gw, nullptr, H, F);
            manual_linear(up + t*F, normed + t*H, l.uw, nullptr, H, F);
            for (int j = 0; j < F; j++)
                gate[t*F+j] = gate[t*F+j] / (1.0f + expf(-gate[t*F+j])) * up[t*F+j];
        }
        for (int t = 0; t < seq_len; t++)
            manual_linear(mlp_out + t*H, gate + t*F, l.dw, nullptr, F, H);
        
        // Final residual + swap buffers
        for (int j = 0; j < seq_len*H; j++) h[j] = h2[j] + mlp_out[j];
        
        if (i < 3 || i > 24) {
            float r = 0; for (int j = 0; j < seq_len*H; j++) r += h[j]*h[j];
            fprintf(stderr, "layer[%d] RMS=%.4f\n", i, sqrtf(r/(seq_len*H)));
        }
    }
    
    // Final norm
    for (int t = 0; t < seq_len; t++)
        manual_rms_norm(h2 + t*H, h + t*H, final_norm, H);
    
    float * last = h2 + (seq_len-1)*H;
    { float r = 0; for (int i = 0; i < H; i++) r += last[i]*last[i];
      fprintf(stderr, "last hidden RMS=%.4f first5=%.6f %.6f %.6f %.6f %.6f\n",
              sqrtf(r/H), last[0], last[1], last[2], last[3], last[4]); }
    
    // Compare with Python
    { FILE * f = fopen("py_lm_all_hiddens.bin", "rb");
      if (f) {
          float * py_all = (float*)malloc(11 * H * sizeof(float));
          fread(py_all, sizeof(float), 11*H, f); fclose(f);
          float * py_last = py_all + (11-1)*H;
          float corr = 0, cpp_sq = 0, py_sq = 0, cross = 0;
          for (int i = 0; i < H; i++) {
              cpp_sq += last[i]*last[i]; py_sq += py_last[i]*py_last[i]; cross += last[i]*py_last[i];
          }
          float cpp_rms = sqrtf(cpp_sq/H), py_rms = sqrtf(py_sq/H);
          corr = cross / H / (cpp_rms * py_rms);
          fprintf(stderr, "Py  last RMS=%.4f first5=%.6f %.6f %.6f %.6f %.6f\n",
                  py_rms, py_last[0], py_last[1], py_last[2], py_last[3], py_last[4]);
          fprintf(stderr, "Correlation: %.6f\n", corr);
          float maxd = 0; for (int i = 0; i < H; i++) { float d = fabsf(last[i]-py_last[i]); if (d>maxd) maxd=d; }
          fprintf(stderr, "Max diff: %.6f\n", maxd);
          if (corr > 0.99999f) fprintf(stderr, "MANUAL LLM MATCHES PYTHON!\n");
          else fprintf(stderr, "DIFFERS (corr=%.4f)\n", corr);
          free(py_all);
      }
    }
    
    // Save for later
    { FILE * f = fopen("manual_llm_last.bin", "wb");
      if (f) { fwrite(last, sizeof(float), H, f); fclose(f); } }
    
    // Free all
    free(embed); free(final_norm); free(scores); free(qh); free(kh); free(vh);
    free(h); free(h2); free(normed); free(q); free(k); free(v);
    free(ao); free(attn_out); free(gate); free(up); free(mlp_out);
    for (auto & l : layers) { free(l.an); free(l.fn); free(l.qw); free(l.qb); free(l.kw); free(l.kb); free(l.vw); free(l.vb); free(l.ow); free(l.ob); free(l.gw); free(l.uw); free(l.dw); }
    return 0;
}
