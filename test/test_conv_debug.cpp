// Debug: dump ds_proj weight
#include "patchenc.h"
#include "safetensors.h"
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "ggml.h"
bool load_patchenc_weights(SafeTensorsFile & sf, ggml_context * w_ctx, patch_encoder & enc);
#include <cstdio>
#include <cmath>

int main() {
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/model.safetensors";
    
    SafeTensorsFile sf;
    if (!sf.open(model_path)) { fprintf(stderr, "FAILED\n"); return 1; }
    
    ggml_init_params gp = { .mem_size = 2048ULL*1024*1024 };
    ggml_context * w_ctx = ggml_init(gp);
    
    patch_encoder pe;
    if (!load_patchenc_weights(sf, w_ctx, pe)) { fprintf(stderr, "FAILED pe\n"); return 1; }
    sf.close();
    
    float * w = tensor_data(pe.conv_weight);
    float * b = pe.conv_bias ? tensor_data(pe.conv_bias) : nullptr;
    
    printf("conv_weight ne: %lld %lld %lld %lld\n", 
           pe.conv_weight->ne[0], pe.conv_weight->ne[1], pe.conv_weight->ne[2], pe.conv_weight->ne[3]);
    printf("First 10 weights: ");
    for(int i=0;i<10;i++) printf("%.6f ", w[i]);
    printf("\n");
    
    // Access as Python would: w[oc, ic, k] for some channels
    int oc=0, ic=0;
    printf("w[0,0,0]=%.6f w[0,0,1]=%.6f w[1,0,0]=%.6f\n",
           w[(oc*128 + ic)*2 + 0], w[(oc*128 + ic)*2 + 1], w[(1*128 + 0)*2 + 0]);
    
    if(b) { printf("bias first5: %.6f %.6f %.6f %.6f %.6f\n", b[0],b[1],b[2],b[3],b[4]); }
    
    // Also dump input  
    float input[512];
    FILE * f = fopen("/tmp/pe_python_input.bin","rb");
    if(f){ fread(input,sizeof(float),512,f); fclose(f); }
    printf("Input first10: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
           input[0],input[1],input[2],input[3],input[4],input[5],input[6],input[7],input[8],input[9]);
    
    // Manual conv
    int seq=4, ch=128;
    float out[256]={0};
    for(int o=0;o<2;o++){
        for(int oc=0;oc<ch;oc++){
            float s = b?b[oc]:0;
            for(int ic=0;ic<ch;ic++){
                int wb = (oc*ch + ic)*2;
                if(o==0){
                    s += input[0*ch + ic] * w[wb + 1];
                } else {
                    s += input[1*ch + ic] * w[wb + 0] + input[2*ch + ic] * w[wb + 1];
                }
            }
            out[o*ch+oc] = s;
        }
    }
    printf("Manual conv pos0 first5: %.6f %.6f %.6f %.6f %.6f\n",
           out[0],out[1],out[2],out[3],out[4]);
    printf("Manual conv pos1 first5: %.6f %.6f %.6f %.6f %.6f\n",
           out[128],out[129],out[130],out[131],out[132]);
    
    ggml_free(w_ctx);
    return 0;
}
