// SPDX-License-Identifier: GPL-3.0-or-later
// Test: PatchEncoder output for known input
#include "patchenc.h"
#include "safetensors.h"
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "ggml.h"
bool load_patchenc_weights(SafeTensorsFile & sf, ggml_context * w_ctx, patch_encoder & enc);
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

int main() {
    // Load PE weights
    const char * model_path = getenv("DOTS_TTS_MODEL");
    if (!model_path) model_path = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35/model.safetensors";
    
    SafeTensorsFile sf;
    if (!sf.open(model_path)) { fprintf(stderr, "FAILED open\n"); return 1; }
    
    ggml_init_params gparams = { .mem_size = 1536ULL*1024*1024 };
    ggml_context * w_ctx = ggml_init(gparams);
    
    patch_encoder pe;
    if (!load_patchenc_weights(sf, w_ctx, pe)) { fprintf(stderr, "FAILED pe load\n"); return 1; }
    sf.close();
    
    // Create a test input: 1 patch (4 frames * 128 dims)
    ggml_init_params cparams = { .mem_size = 128ULL*1024*1024 };
    ggml_context * ctx = ggml_init(cparams);
    
    // Load reference input from Python's first call latents
    float test_input[512];
    FILE * f = fopen("py_call0_latents.bin", "rb");
    if (!f) {
        // Fallback: random
        srand(42);
        for(int i=0;i<512;i++) test_input[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    } else {
        fread(test_input, sizeof(float), 512, f);
        fclose(f);
    }
    
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PATCHENC_LATENT_DIM, PATCHENC_PATCH_SIZE);
    memcpy(tensor_data(x), test_input, 512*sizeof(float));
    
    ggml_tensor * out = patchenc_forward(pe, ctx, x, 1);
    
    float * od = tensor_data(out);
    // out shape: [1536, 1] in ggml layout = [1536]
    float rms=0; for(int i=0;i<1536;i++) rms += od[i]*od[i];
    rms = sqrtf(rms/1536);
    printf("PE output RMS=%.4f  first10:", rms);
    for(int i=0;i<10;i++) printf(" %.6f", od[i]);
    printf("\n");
    
    // Also test with denormalized input
    static const float VAE_MEAN[128] = {-0.9025,-0.6183,-0.0263,-0.0685,-0.1783,-0.5570,-0.6110,-0.3879,-0.3249,-0.2662,0.3478,0.0372,-0.5750,0.0344,-0.5378,0.2272,-0.2407,0.2857,-0.4126,0.0592,-0.3421,-0.5783,0.0266,-0.0334,-0.0829,-0.3520,0.1200,-0.4391,0.3389,-0.4224,-0.3515,-0.0558,0.1107,-0.0028,0.6533,-0.0096,0.3998,0.0979,0.0710,0.1544,-0.1871,0.1712,0.2767,-0.3803,-0.2228,-0.3248,0.0888,0.0382,-0.2751,-0.1028,-0.3058,0.2343,0.0244,-0.2566,-0.0230,0.7069,-0.4051,0.5665,-0.1977,-0.4928,0.1987,-0.3744,0.2669,-0.5638,-0.0391,0.1564,-0.1035,-0.4483,0.6975,0.2761,-0.2020,0.2640,0.0682,0.1176,0.7706,-1.3600,0.0722,-0.1375,0.0337,0.1764,-0.2927,0.3615,-0.1335,0.3452,0.0590,-0.2773,0.0653,0.3692,-0.3961,-0.0325,0.1038,-0.2431,-0.0402,0.3810,-0.0239,-0.3426,-0.0924,-0.2611,0.1166,0.4047,0.1759,-0.6634,-0.0081,2.0526,0.0439,-0.1364,0.1873,-0.0263,-0.7545,0.5526,-0.0752,-0.3717,0.1637,0.7791,0.7361,-0.3620,-0.3784,0.4654,-0.2694,-0.2115,0.3204,0.2995,-0.7347,0.4701,0.1141,0.6925,0.7140,-0.0032};
    static const float VAE_STD[128] = {3.0316,2.8055,1.7751,1.7465,2.0646,1.9869,2.2711,2.0302,1.8684,2.3117,2.0107,1.9570,2.8696,2.3088,2.0507,2.1352,2.6940,1.7395,1.9994,1.8537,2.0860,2.1602,1.9088,2.6236,1.9415,3.6039,1.7381,1.8368,2.7282,2.0870,2.4967,2.4496,1.6383,1.6716,4.6338,2.4076,3.9610,1.8117,1.8016,1.8020,2.0124,1.7533,1.9128,2.2415,2.0514,1.9496,1.8469,2.1641,1.9519,1.9785,1.7530,1.6008,1.9471,2.6634,2.2622,2.4504,1.9333,2.8137,2.0679,2.1296,1.8197,2.4677,2.2560,2.2627,1.9336,1.6912,2.1193,1.7280,3.0617,1.5928,1.7773,1.8976,1.6118,1.5717,2.4272,3.1342,2.3934,1.8039,1.8262,1.7740,3.5085,1.9350,2.2693,1.8149,1.5992,1.8857,2.5523,2.2707,1.8558,1.6974,1.9261,1.8115,1.5260,1.8547,1.4196,1.7291,1.6253,2.0238,2.5385,2.1883,1.6951,2.5633,2.0287,13.0302,2.1418,1.6957,4.7318,2.3544,2.6665,2.8945,1.7005,6.3084,2.2875,2.7594,2.3732,1.7797,1.9751,2.6090,1.6278,2.5487,2.0741,2.5441,2.0957,1.6018,1.4397,3.2050,1.8041,2.3601};
    
    ggml_reset(ctx);
    float denorm_input[512];
    for(int i=0;i<4;i++) for(int c=0;c<128;c++)
        denorm_input[i*128+c] = test_input[i*128+c] * VAE_STD[c] + VAE_MEAN[c];
    
    ggml_tensor * xd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PATCHENC_LATENT_DIM, PATCHENC_PATCH_SIZE);
    memcpy(tensor_data(xd), denorm_input, 512*sizeof(float));
    ggml_tensor * outd = patchenc_forward(pe, ctx, xd, 1);
    float * odd = tensor_data(outd);
    float rmsd=0; for(int i=0;i<1536;i++) rmsd += odd[i]*odd[i];
    rmsd = sqrtf(rmsd/1536);
    printf("PE denorm out RMS=%.4f  first10:", rmsd);
    for(int i=0;i<10;i++) printf(" %.6f", odd[i]);
    printf("\n");
    
    // Compare
    float cross=0, d2=0;
    for(int i=0;i<1536;i++) { cross+=od[i]*odd[i]; float d=od[i]-odd[i]; d2+=d*d; }
    float corr = cross/1536/rms/rmsd;
    printf("Corr denorm vs raw: %.6f  RMS diff: %.6f\n", corr, sqrtf(d2/1536));
    
    ggml_free(ctx); ggml_free(w_ctx);
    return 0;
}
