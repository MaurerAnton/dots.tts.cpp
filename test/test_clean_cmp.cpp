// Clean comparison: C++ manual_dit_block vs Python for pipeline input
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

    // Load Python data
    float py_il[8192], py_cond[1024], py_b0[8192];
    { FILE * f = fopen("debug/py_il_out.bin","rb"); fread(py_il, sizeof(float), 8192, f); fclose(f); }
    { FILE * f = fopen("debug/py_cond.bin","rb"); fread(py_cond, sizeof(float), 1024, f); fclose(f); }
    { FILE * f = fopen("debug/py_b0_out.bin","rb"); fread(py_b0, sizeof(float), 8192, f); fclose(f); }

    float cpp_b0[8192];
    manual_dit_block(py_il, py_cond, m.layers[0], cpp_b0, 8);

    // Compare
    float r_py=0, r_cpp=0, max_d=0;
    int first_diff = -1;
    for(int i=0;i<8192;i++){
        r_py+=py_b0[i]*py_b0[i]; r_cpp+=cpp_b0[i]*cpp_b0[i];
        float d=fabsf(py_b0[i]-cpp_b0[i]);
        if(d>max_d){max_d=d; if(first_diff<0) first_diff=i;}
    }
    printf("Py RMS=%.4f  C++ RMS=%.4f  max_diff=%.6f at idx=%d\n",
        sqrtf(r_py/8192), sqrtf(r_cpp/8192), max_d, first_diff);
    printf("Py first5: "); for(int i=0;i<5;i++) printf("%.4f ", py_b0[i]); printf("\n");
    printf("C++ first5: "); for(int i=0;i<5;i++) printf("%.4f ", cpp_b0[i]); printf("\n");
    printf("Py at %d: %.4f  C++: %.4f\n", first_diff, py_b0[first_diff], cpp_b0[first_diff]);
    printf("%s\n", max_d < 0.01 ? "MATCH!" : "DIFFERS");

    ggml_free(w_ctx);
    return 0;
}
