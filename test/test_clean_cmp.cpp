// Isolate manual_attention: dump Q/K/V and compare with Python
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
    ggml_init_params wp = { .mem_size = 3ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    dit_model m; load_dit_weights(sf, w_ctx, m); sf.close();

    // Load Python modulated input
    float mod_in[8192];
    { FILE * f = fopen("debug/py_modulated.bin","rb"); fread(mod_in,sizeof(float),8192,f); fclose(f); }

    // Load Python Q/K/V for comparison
    float py_q[8192], py_k[8192], py_v[8192];
    { FILE * f = fopen("debug/py_q.bin","rb"); fread(py_q,sizeof(float),8192,f); fclose(f); }
    { FILE * f = fopen("debug/py_k.bin","rb"); fread(py_k,sizeof(float),8192,f); fclose(f); }
    { FILE * f = fopen("debug/py_v.bin","rb"); fread(py_v,sizeof(float),8192,f); fclose(f); }

    // Run C++ manual_attention
    float cpp_out[8192];
    manual_attention(cpp_out, mod_in,
        tensor_data(m.layers[0].attn_q_weight), tensor_data(m.layers[0].attn_k_weight),
        tensor_data(m.layers[0].attn_v_weight), tensor_data(m.layers[0].attn_o_weight),
        nullptr, nullptr, nullptr, 
        m.layers[0].attn_o_bias ? tensor_data(m.layers[0].attn_o_bias) : nullptr,
        m.layers[0].q_norm_w ? tensor_data(m.layers[0].q_norm_w) : nullptr,
        m.layers[0].k_norm_w ? tensor_data(m.layers[0].k_norm_w) : nullptr,
        8, 1024, 16, 64);

    // Actually we can't get Q/K/V from manual_attention easily. Let's just compare final output.
    // Load Python attn output
    float py_attn[8192];
    { FILE * f = fopen("debug/py_attn_out.bin","rb"); fread(py_attn,sizeof(float),8192,f); fclose(f); }

    float r_py=0, r_cpp=0, max_d=0; int max_i=0;
    for(int i=0;i<8192;i++){
        r_py+=py_attn[i]*py_attn[i]; r_cpp+=cpp_out[i]*cpp_out[i];
        float d=fabsf(py_attn[i]-cpp_out[i]); if(d>max_d){max_d=d; max_i=i;}
    }
    printf("Attention output: Py RMS=%.4f C++ RMS=%.4f max_diff=%.6f at idx %d\n",
        sqrtf(r_py/8192), sqrtf(r_cpp/8192), max_d, max_i);
    printf("  Py[%d]=%.4f C++[%d]=%.4f\n", max_i, py_attn[max_i], max_i, cpp_out[max_i]);
    printf("%s\n", max_d < 0.01 ? "MATCH!" : "DIFFERS");

    // Dump C++ output
    FILE * f = fopen("debug/cpp_attn_out.bin","wb");
    fwrite(cpp_out, sizeof(float), 8192, f); fclose(f);

    ggml_free(w_ctx);
    return 0;
}
