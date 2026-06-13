// Test: BLAS-based linear layer vs Python MKL
// Compares C++ OpenBLAS output with Python reference
#include "dots_tts.h"
#include "dit.h"
#include "dit_manual.h"
#include "safetensors.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cblas.h>
#include <cstdio>
#include <cstring>
#include <cmath>

bool load_dit_weights(SafeTensorsFile & sf, ggml_context * w_ctx, dit_model & m);

// BLAS-based linear: out = input @ weight^T + bias
static void linear_blas(float * out, const float * in, const float * w, const float * b,
    int in_dim, int out_dim) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        1, out_dim, in_dim, 1.0f, in, in_dim, w, out_dim, 0.0f, out, out_dim);
    if (b) for (int o = 0; o < out_dim; o++) out[o] += b[o];
}

static float compute_rms(const float * d, int n) { double r=0; for(int i=0;i<n;i++)r+=(double)d[i]*d[i]; return sqrtf(r/n); }

int main() {
    setbuf(stdout, NULL);
    const char * sf_path = "models/model.safetensors";
    { FILE * f = fopen(sf_path, "rb"); if (!f) sf_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/e3520f75254d0020a0406db31c51a79d00d22d55/model.safetensors"; }
    
    ggml_init_params wp = { .mem_size = 3ULL*1024*1024*1024 };
    ggml_context * w_ctx = ggml_init(wp);
    SafeTensorsFile sf; sf.open(sf_path);
    dit_model dit; load_dit_weights(sf, w_ctx, dit); sf.close();
    
    // Load the test input from Python (debug_v4/dit_input_test.bin)
    const int total_len = 5, hidden = 1024;
    float * dit_input = new float[total_len * hidden];
    { FILE * f = fopen("debug_v4/dit_input_test.bin", "rb");
      if (!f) { printf("Need dit_input_test.bin — run: python3 gen_test.py\n"); return 1; }
      fread(dit_input, sizeof(float), total_len * hidden, f); fclose(f); }
    
    // Test: compare BLAS linear vs manual_linear on first layer
    float * iw = tensor_data(dit.input_layer_w);
    float * ib = dit.input_layer_b ? tensor_data(dit.input_layer_b) : nullptr;
    
    float out_blas[1024], out_manual[1024];
    linear_blas(out_blas, dit_input, iw, ib, hidden, hidden);
    manual_linear(out_manual, dit_input, iw, ib, hidden, hidden);
    
    printf("BLAS vs Manual on input_layer:\n");
    printf("  BLAS RMS=%.6f  Manual RMS=%.6f\n", compute_rms(out_blas, hidden), compute_rms(out_manual, hidden));
    float corr=0; {double sx=0,sy=0,sxy=0,sx2=0,sy2=0; for(int i=0;i<hidden;i++){sx+=out_blas[i];sy+=out_manual[i];sxy+=(double)out_blas[i]*out_manual[i];sx2+=(double)out_blas[i]*out_blas[i];sy2+=(double)out_manual[i]*out_manual[i];} corr=(float)((hidden*sxy-sx*sy)/sqrt((hidden*sx2-sx*sx)*(hidden*sy2-sy*sy)));}
    printf("  Corr=%.8f  Max|diff|=%.6f\n", corr, [&]{float m=0;for(int i=0;i<hidden;i++){float d=fabsf(out_blas[i]-out_manual[i]);if(d>m)m=d;}return m;}());
    
    // Now compare BLAS output with Python reference
    float py_out[1024];
    { FILE * f = fopen("debug_v4/py_input_layer_out.bin", "rb");
      if (f) { fread(py_out, sizeof(float), hidden, f); fclose(f);
        printf("\nBLAS vs Python input_layer:\n");
        printf("  BLAS RMS=%.6f  Py RMS=%.6f\n", compute_rms(out_blas, hidden), compute_rms(py_out, hidden));
        corr=0; {double sx=0,sy=0,sxy=0,sx2=0,sy2=0; for(int i=0;i<hidden;i++){sx+=out_blas[i];sy+=py_out[i];sxy+=(double)out_blas[i]*py_out[i];sx2+=(double)out_blas[i]*out_blas[i];sy2+=(double)py_out[i]*py_out[i];} corr=(float)((hidden*sxy-sx*sy)/sqrt((hidden*sx2-sx*sx)*(hidden*sy2-sy*sy)));}
        printf("  Corr=%.8f  Max|diff|=%.6f\n", corr, [&]{float m=0;for(int i=0;i<hidden;i++){float d=fabsf(out_blas[i]-py_out[i]);if(d>m)m=d;}return m;}());
      } else printf("No Python reference — generate with Python first\n");
    }
    
    delete[] dit_input;
    ggml_free(w_ctx);
    return 0;
}
