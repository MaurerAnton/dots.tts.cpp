// Dump C++ bottleneck output (latents after post_proj+LSTM) for Python comparison
#include "dots_tts.h"
#include "bigvgan_cpp.h"
#include "safetensors.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>

int main() {
    BigVGANDecoder dec;
    if (!bigvgan_load("models/vocoder_eff.safetensors", dec)) return 1;
    
    // Load latents
    float latents[128*128];
    int n_frames = 0;
    FILE* f = fopen("build/latents.bin", "rb");
    if (!f) f = fopen("latents.bin", "rb");
    while (fread(latents + n_frames*128, sizeof(float), 128, f) == 128) n_frames++;
    fclose(f);
    fprintf(stderr, "Loaded %d latent frames (RMS)\n", n_frames);
    
    // Run post_proj
    float* tmp = (float*)malloc(n_frames * 1536 * sizeof(float));
    conv1d_causal(tmp, latents, 128, n_frames, dec.post_proj_w.ptr(), dec.post_proj_b.ptr(), 128, 1);
    { float r=0; for(int i=0;i<n_frames*128;i++) r+=tmp[i]*tmp[i];
      printf("post_proj RMS: %.4f\n", sqrtf(r/(n_frames*128))); }
    
    // Save post_proj output
    f = fopen("debug/cpp_post_proj.bin", "wb");
    fwrite(tmp, sizeof(float), n_frames*128, f); fclose(f);
    
    // Run LSTM bottleneck
    float* mi_buf = (float*)malloc(n_frames * 512 * sizeof(float));
    for (int t = 0; t < n_frames; t++)
        for (int o = 0; o < 512; o++) {
            float s = dec.mi_b1.ptr()[o];
            for (int i = 0; i < 128; i++)
                s += tmp[t*128+i] * dec.mi_w1.ptr()[o*128+i];
            mi_buf[t*512+o] = s;
        }
    // LSTM with 4 layers
    float *hx[4]={0}, *cx[4]={0};
    float* lstm_out = (float*)malloc(n_frames * 512 * sizeof(float));
    for (int t = 0; t < n_frames; t++) {
        float* xt = mi_buf + t*512;
        float* ht = lstm_out + t*512;
        for (int layer = 0; layer < 4; layer++) {
            float* w_ih = dec.mi_lstm_w_ih[layer].ptr();
            float* w_hh = dec.mi_lstm_w_hh[layer].ptr();
            float* b_ih = dec.mi_lstm_b_ih[layer].ptr();
            float* b_hh = dec.mi_lstm_b_hh[layer].ptr();
            // Allocate hx/cx per layer if not done
            static float hx_mem[4][512], cx_mem[4][512];
            if (t == 0) { memset(hx_mem[layer], 0, 512*4); memset(cx_mem[layer], 0, 512*4); }
            float* h = hx_mem[layer], *c = cx_mem[layer];
            float* out_t = (layer == 3) ? ht : xt; // last layer writes to ht, others use xt as scratch
            float scratch[512];
            if (layer < 3) memcpy(scratch, xt, 512*4);
            
            for (int g = 0; g < 4; g++) {
                float* gate_out = (layer < 3) ? scratch : ht;
                for (int o = 0; o < 512; o++) {
                    float s = b_ih[g*512+o] + b_hh[g*512+o];
                    for (int i = 0; i < 512; i++) {
                        s += xt[i] * w_ih[(g*512+o)*512 + i];
                        s += h[i] * w_hh[(g*512+o)*512 + i];
                    }
                    if (g == 0) { // input gate
                        float ig = 1.0f/(1.0f+expf(-s));
                        float cg = tanhf(s); // cell gate shares same weights? No — gates 0,1,2,3 are i,f,g,o
                        c[o] = (1.0f/(1.0f+expf(-(b_ih[2*512+o] + b_hh[2*512+o] + 
                            [&](){float ss=0;for(int i=0;i<512;i++)ss+=xt[i]*w_ih[(2*512+o)*512+i]+h[i]*w_hh[(2*512+o)*512+i];return ss;}())))) * tanhf(
                            [&](){float ss=b_ih[1*512+o]+b_hh[1*512+o];for(int i=0;i<512;i++)ss+=xt[i]*w_ih[(1*512+o)*512+i]+h[i]*w_hh[(1*512+o)*512+i];return ss;}())) * 
                            c[o] + ig * tanhf(
                            [&](){float ss=b_ih[3*512+o]+b_hh[3*512+o];for(int i=0;i<512;i++)ss+=xt[i]*w_ih[(3*512+o)*512+i]+h[i]*w_hh[(3*512+o)*512+i];return ss;}()));
                    }
                }
            }
        }
    }
    abort(); // Too complex, let me just use the existing lstm_forward
    
    return 0;
}
