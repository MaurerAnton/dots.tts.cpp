// Test: dump block 0 FFN intermediates from C++
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

static void write_bin(const char * name, const float * data, int n) {
    char path[256]; snprintf(path, sizeof(path), "debug/cpp_%s.bin", name);
    FILE * f = fopen(path, "wb"); fwrite(data, sizeof(float), n, f); fclose(f);
}

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "models/model.safetensors";
    SafeTensorsFile sf; sf.open(model_path);
    ggml_init_params wp = { .mem_size = 2ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model m; load_dit_weights(sf, w_ctx, m); sf.close();

    int n_tokens = 8, hidden = DIT_HIDDEN_SIZE;
    float * x_in = new float[n_tokens * hidden];
    FILE * f_in = fopen("debug/cpp_dit_input.bin", "rb");
    fread(x_in, sizeof(float), n_tokens * hidden, f_in); fclose(f_in);

    // input_layer
    float * h = new float[n_tokens * hidden];
    { float * iw = tensor_data(m.input_layer_w);
      float * ib = m.input_layer_b ? tensor_data(m.input_layer_b) : nullptr;
      for (int t = 0; t < n_tokens; t++) manual_linear(h + t*hidden, x_in + t*hidden, iw, ib, hidden, hidden); }

    // cond
    float cond[1024];
    { float t_val = 0.0f; int half = 128; float se[256], h1[1024];
      for(int i=0;i<half;i++){ float f=expf(-logf(10000.0f)*(float)i/(float)half); se[i]=cosf(t_val*f); se[half+i]=sinf(t_val*f); }
      float * w1 = tensor_data(m.t_embed_w1); float * b1 = m.t_embed_b1 ? tensor_data(m.t_embed_b1) : nullptr;
      for(int o=0;o<1024;o++){ float s=b1?b1[o]:0.0f; for(int i=0;i<256;i++) s+=w1[o*256+i]*se[i]; h1[o]=s/(1.0f+expf(-s)); }
      float * w2 = tensor_data(m.t_embed_w2); float * b2 = m.t_embed_b2 ? tensor_data(m.t_embed_b2) : nullptr;
      for(int o=0;o<1024;o++){ float s=b2?b2[o]:0.0f; for(int i=0;i<1024;i++) s+=w2[o*1024+i]*h1[i]; cond[o]=s; }
      float temp[1024]; float * sw1=tensor_data(m.spk_proj_w1); float * sb1=m.spk_proj_b1?tensor_data(m.spk_proj_b1):nullptr;
      for(int o=0;o<1024;o++){ float s=sb1?sb1[o]:0.0f; for(int i=0;i<512;i++) s+=sw1[o*512+i]*0.0f; temp[o]=s; }
      float mean=0; for(int i=0;i<1024;i++) mean+=temp[i]; mean/=1024;
      float var=0; for(int i=0;i<1024;i++){ float d=temp[i]-mean; var+=d*d; } var=var/1024+1e-5f;
      float istd=1.0f/sqrtf(var); float *lw=m.spk_ln_w?tensor_data(m.spk_ln_w):nullptr, *lb=m.spk_ln_b?tensor_data(m.spk_ln_b):nullptr;
      for(int i=0;i<1024;i++){ float x=(temp[i]-mean)*istd; if(lw)x*=lw[i]; if(lb)x+=lb[i]; cond[i]+=x; } }

    // adaLN
    float cs[1024]; for(int i=0;i<1024;i++) cs[i]=cond[i]/(1.0f+expf(-cond[i]));
    float adaln_raw[6*DIT_HIDDEN_SIZE];
    { float * aw = tensor_data(m.layers[0].adaln_linear_w); float * ab = m.layers[0].adaln_linear_b ? tensor_data(m.layers[0].adaln_linear_b) : nullptr;
      for(int o=0;o<6*hidden;o++){ float s=ab?ab[o]:0.0f; for(int i=0;i<hidden;i++) s+=aw[o*hidden+i]*cs[i]; adaln_raw[o]=s; } }
    float * sm=adaln_raw, * scm=adaln_raw+hidden, * gm=adaln_raw+2*hidden;
    float * sml=adaln_raw+3*hidden, * scl=adaln_raw+4*hidden, * gml=adaln_raw+5*hidden;

    write_bin("b0_ffn_shift", sml, hidden);
    write_bin("b0_ffn_scale", scl, hidden);
    write_bin("b0_ffn_gate", gml, hidden);

    // Attention (matching test_attn_calib approach)
    float * normed = new float[n_tokens*hidden], * mod = new float[n_tokens*hidden];
    for(int t=0;t<n_tokens;t++){ manual_layernorm(normed+t*hidden,h+t*hidden,hidden);
        for(int i=0;i<hidden;i++) mod[t*hidden+i]=normed[t*hidden+i]*(1.0f+scm[i])+sm[i]; }
    float * ao = new float[n_tokens*hidden];
    manual_attention(ao, mod, tensor_data(m.layers[0].attn_q_weight),
        tensor_data(m.layers[0].attn_k_weight), tensor_data(m.layers[0].attn_v_weight),
        tensor_data(m.layers[0].attn_o_weight),
        nullptr,nullptr,nullptr, m.layers[0].attn_o_bias?tensor_data(m.layers[0].attn_o_bias):nullptr,
        m.layers[0].q_norm_w?tensor_data(m.layers[0].q_norm_w):nullptr,
        m.layers[0].k_norm_w?tensor_data(m.layers[0].k_norm_w):nullptr,
        n_tokens,hidden,DIT_NUM_HEADS,DIT_HEAD_SIZE);
    float * after_attn = new float[n_tokens*hidden];
    for(int i=0;i<n_tokens*hidden;i++) after_attn[i]=h[i]+gm[i%hidden]*ao[i];
    write_bin("b0_ffn_input", after_attn, n_tokens*hidden);

    // FFN
    for(int t=0;t<n_tokens;t++){ manual_layernorm(normed+t*hidden,after_attn+t*hidden,hidden);
        for(int i=0;i<hidden;i++) mod[t*hidden+i]=normed[t*hidden+i]*(1.0f+scl[i])+sml[i]; }
    write_bin("b0_ffn_normed", normed, n_tokens*hidden);
    write_bin("b0_ffn_modulated", mod, n_tokens*hidden);

    float * fh1 = new float[n_tokens*DIT_FFN_SIZE], * fh2 = new float[n_tokens*hidden];
    float * fw1=tensor_data(m.layers[0].ffn_w1), * fw2=tensor_data(m.layers[0].ffn_w2);
    float * fb1=m.layers[0].ffn_b1?tensor_data(m.layers[0].ffn_b1):nullptr, * fb2=m.layers[0].ffn_b2?tensor_data(m.layers[0].ffn_b2):nullptr;
    for(int t=0;t<n_tokens;t++){ manual_linear(fh1+t*DIT_FFN_SIZE, mod+t*hidden, fw1, fb1, hidden, DIT_FFN_SIZE);
        for(int i=0;i<DIT_FFN_SIZE;i++){ float xv=fh1[t*DIT_FFN_SIZE+i]; float x2=xv*xv;
            fh1[t*DIT_FFN_SIZE+i]=0.5f*xv*(1.0f+tanhf(0.7978845608f*(xv+0.044715f*x2*xv))); }
        manual_linear(fh2+t*hidden, fh1+t*DIT_FFN_SIZE, fw2, fb2, DIT_FFN_SIZE, hidden); }
    write_bin("b0_ffn_fc1", fh1, n_tokens*DIT_FFN_SIZE);
    write_bin("b0_ffn_gelu", fh1, n_tokens*DIT_FFN_SIZE);
    write_bin("b0_ffn_fc2", fh2, n_tokens*hidden);

    float * ffn_out = new float[n_tokens*hidden];
    for(int i=0;i<n_tokens*hidden;i++) ffn_out[i]=after_attn[i]+gml[i%hidden]*fh2[i];
    write_bin("b0_ffn_output", ffn_out, n_tokens*hidden);

    float rms=0; for(int i=0;i<n_tokens*hidden;i++) rms+=ffn_out[i]*ffn_out[i];
    fprintf(stderr, "Block 0 FFN output RMS: %.4f\n", sqrtf(rms/(n_tokens*hidden)));

    // Verify fc1 weight dimensions
    fprintf(stderr, "ffn_w1 ne: [%ld,%ld,%ld,%ld]  ffn_w2 ne: [%ld,%ld,%ld,%ld]\n",
        m.layers[0].ffn_w1->ne[0], m.layers[0].ffn_w1->ne[1], m.layers[0].ffn_w1->ne[2], m.layers[0].ffn_w1->ne[3],
        m.layers[0].ffn_w2->ne[0], m.layers[0].ffn_w2->ne[1], m.layers[0].ffn_w2->ne[2], m.layers[0].ffn_w2->ne[3]);

    delete[] x_in; delete[] h; delete[] normed; delete[] mod; delete[] ao;
    delete[] after_attn; delete[] fh1; delete[] fh2; delete[] ffn_out;
    ggml_free(w_ctx);
    return 0;
}
