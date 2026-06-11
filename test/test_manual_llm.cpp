// Verify: shared llm_manual.cpp matches PyTorch
#include "llm_manual.h"
#include "safetensors.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/model.safetensors";
    
    SafeTensorsFile sf;
    if (!sf.open(model_path)) { fprintf(stderr, "FAILED open\n"); return 1; }
    
    LLMWeights w;
    if (!load_llm_weights(sf, w, 28)) { fprintf(stderr, "FAILED load\n"); return 1; }
    sf.close();
    
    int token_ids[] = {58, 108704, 60, 14990, 1879, 58, 108704, 103124, 105761, 60, 151668};
    int n_tok = 11;
    float * hiddens = (float*)malloc(n_tok * 1536 * sizeof(float));
    
    llm_manual_forward(w, token_ids, n_tok, hiddens);
    free_llm_weights(w);
    
    float * last = hiddens + (n_tok-1)*1536;
    float r = 0; for (int i = 0; i < 1536; i++) r += last[i]*last[i];
    fprintf(stderr, "last hidden RMS=%.4f first5=%.6f %.6f %.6f %.6f %.6f\n",
            sqrtf(r/1536), last[0], last[1], last[2], last[3], last[4]);
    
    FILE * f = fopen("py_lm_all_hiddens.bin", "rb");
    if (f) {
        float * py_all = (float*)malloc(11 * 1536 * sizeof(float));
        fread(py_all, sizeof(float), 11*1536, f); fclose(f);
        float * py_last = py_all + 10*1536;
        
        float py_r = 0; for (int i = 0; i < 1536; i++) py_r += py_last[i]*py_last[i];
        fprintf(stderr, "Py  last RMS=%.4f first5=%.6f %.6f %.6f %.6f %.6f\n",
                sqrtf(py_r/1536), py_last[0], py_last[1], py_last[2], py_last[3], py_last[4]);
        
        float cpp_sq=0, py_sq=0, cross=0;
        for (int i = 0; i < 1536; i++) { cpp_sq+=last[i]*last[i]; py_sq+=py_last[i]*py_last[i]; cross+=last[i]*py_last[i]; }
        float corr = cross/1536/sqrtf(cpp_sq/1536)/sqrtf(py_sq/1536);
        
        if (corr > 0.99999f) fprintf(stderr, "MANUAL LLM MATCHES PYTHON!\n");
        else fprintf(stderr, "DIFFERS (corr=%.6f)\n", corr);
        free(py_all);
    }
    
    free(hiddens);
    return 0;
}
