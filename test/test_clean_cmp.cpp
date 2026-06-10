// Dump attention intermediates from manual_dit_block for comparison
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

    float py_il[8192], py_cond[1024], py_b0[8192];
    { FILE * f = fopen("debug/py_il_out.bin","rb"); fread(py_il, sizeof(float), 8192, f); fclose(f); }
    { FILE * f = fopen("debug/py_cond.bin","rb"); fread(py_cond, sizeof(float), 1024, f); fclose(f); }
    { FILE * f = fopen("debug/py_b0_out.bin","rb"); fread(py_b0, sizeof(float), 8192, f); fclose(f); }

    // Run manual block and dump intermediates
    float cpp_b0[8192];
    manual_dit_block(py_il, py_cond, m.layers[0], cpp_b0, 8);

    // Compare with Python
    float max_d=0; int max_i=0;
    for(int i=0;i<8192;i++){
        float d=fabsf(py_b0[i]-cpp_b0[i]);
        if(d>max_d){max_d=d; max_i=i;}
    }
    int token = max_i / 1024;
    int feat = max_i % 1024;
    printf("Max diff: %.6f at idx %d (token %d, feat %d)\n", max_d, max_i, token, feat);
    printf("Py[%d]=%.4f  C++[%d]=%.4f\n", max_i, py_b0[max_i], max_i, cpp_b0[max_i]);
    
    // Dump C++ output for external comparison
    FILE * f = fopen("debug/cpp_b0_out.bin","wb");
    fwrite(cpp_b0, sizeof(float), 8192, f); fclose(f);

    ggml_free(w_ctx);
    return 0;
}
