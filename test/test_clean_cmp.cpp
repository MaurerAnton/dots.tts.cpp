// Dump attention scores for head 0
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

// Modified manual_attention that dumps scores for head 0
static void manual_attention_dump(float * out, const float * x,
    const float * qw, const float * kw, const float * vw, const float * ow,
    const float * qb, const float * kb, const float * vb, const float * ob,
    const float * qnw, const float * knw,
    int n_tokens, int hidden, int n_heads, int head_dim) {
    int n = n_tokens * hidden;
    float * q = new float[n], * k = new float[n], * v = new float[n];
    for (int t = 0; t < n_tokens; t++) {
        manual_linear(q + t*hidden, x + t*hidden, qw, qb, hidden, hidden);
        manual_linear(k + t*hidden, x + t*hidden, kw, kb, hidden, hidden);
        manual_linear(v + t*hidden, x + t*hidden, vw, vb, hidden, hidden);
    }
    float * scores = new float[n_tokens * n_tokens];
    float * ao_flat = new float[n];
    for (int h = 0; h < n_heads; h++) {
        float * qh = new float[n_tokens*head_dim], * kh = new float[n_tokens*head_dim], * vh = new float[n_tokens*head_dim];
        for (int t=0;t<n_tokens;t++) for (int d=0;d<head_dim;d++) {
            qh[t*head_dim+d]=q[t*hidden+h*head_dim+d]; kh[t*head_dim+d]=k[t*hidden+h*head_dim+d];
            vh[t*head_dim+d]=v[t*hidden+h*head_dim+d]; }
        for (int t=0;t<n_tokens;t++) { manual_rms_norm(qh+t*head_dim, qh+t*head_dim, qnw, head_dim);
            manual_rms_norm(kh+t*head_dim, kh+t*head_dim, knw, head_dim); }
        manual_rope(qh, kh, n_tokens, head_dim);
        float scale=1.0f/sqrtf((float)head_dim);
        for(int i=0;i<n_tokens;i++){ for(int j=0;j<n_tokens;j++){ float s=0;
            for(int d=0;d<head_dim;d++) s+=qh[i*head_dim+d]*kh[j*head_dim+d]; scores[i*n_tokens+j]=s*scale; }
            for(int j=i+1;j<n_tokens;j++) scores[i*n_tokens+j]=-INFINITY;
            manual_softmax(scores+i*n_tokens,n_tokens); }
        if(h==0){ // Dump scores for head 0
            FILE * f=fopen("debug/cpp_scores_h0.bin","wb");
            fwrite(scores,sizeof(float),n_tokens*n_tokens,f); fclose(f);
            fprintf(stderr,"Head 0 scores row 3: ");
            for(int j=0;j<8;j++) fprintf(stderr,"%.4f ", scores[3*8+j]);
            fprintf(stderr,"\n");
        }
        for(int i=0;i<n_tokens;i++) for(int d=0;d<head_dim;d++){ float s=0;
            for(int j=0;j<n_tokens;j++) s+=scores[i*n_tokens+j]*vh[j*head_dim+d]; ao_flat[i*hidden+h*head_dim+d]=s; }
        delete[] qh; delete[] kh; delete[] vh;
    }
    delete[] scores;
    for (int t = 0; t < n_tokens; t++) manual_linear(out + t*hidden, ao_flat + t*hidden, ow, ob, hidden, hidden);
    delete[] q; delete[] k; delete[] v; delete[] ao_flat;
}

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";
    SafeTensorsFile sf; sf.open(model_path);
    ggml_init_params wp = { .mem_size = 3ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model m; load_dit_weights(sf, w_ctx, m); sf.close();

    float mod_in[8192];
    { FILE * f = fopen("debug/py_modulated.bin","rb"); fread(mod_in,sizeof(float),8192,f); fclose(f); }

    float cpp_out[8192];
    manual_attention_dump(cpp_out, mod_in,
        tensor_data(m.layers[0].attn_q_weight), tensor_data(m.layers[0].attn_k_weight),
        tensor_data(m.layers[0].attn_v_weight), tensor_data(m.layers[0].attn_o_weight),
        nullptr, nullptr, nullptr,
        m.layers[0].attn_o_bias ? tensor_data(m.layers[0].attn_o_bias) : nullptr,
        m.layers[0].q_norm_w ? tensor_data(m.layers[0].q_norm_w) : nullptr,
        m.layers[0].k_norm_w ? tensor_data(m.layers[0].k_norm_w) : nullptr,
        8, 1024, 16, 64);

    ggml_free(w_ctx);
    return 0;
}
