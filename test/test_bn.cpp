// Dump C++ bottleneck output for Python comparison
#include "bigvgan_cpp.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

void conv1d_causal(float*,const float*,int,int,const float*,const float*,int,int,int=1);
void lstm_forward(const float* input, int seq_len, int input_size, int num_layers,
    const float** w_ih, const float** w_hh, const float** b_ih, const float** b_hh,
    float* output, bool skip_mode);

int main() {
    BigVGANDecoder dec;
    if (!bigvgan_load("models/vocoder_eff.safetensors", dec)) return 1;
    
    float latents[128*128]; int nf=0;
    FILE* f = fopen("build/latents.bin","rb");
    while(fread(latents+nf*128,sizeof(float),128,f)==128) nf++;
    fclose(f);
    
    float* buf = (float*)malloc(nf*512*sizeof(float));
    
    // post_proj
    conv1d_causal(buf, latents, 128, nf, dec.post_proj_w.ptr(), dec.post_proj_b.ptr(), 128, 1);
    
    // Linear 128->512
    float* mi_buf = (float*)malloc(nf*512*sizeof(float));
    for(int t=0;t<nf;t++) for(int o=0;o<512;o++){float s=dec.mi_b1.ptr()[o];
        for(int i=0;i<128;i++) s+=buf[t*128+i]*dec.mi_w1.ptr()[o*128+i]; mi_buf[t*512+o]=s;}
    
    // LSTM
    const float* w_ih[4]={dec.mi_lstm_w_ih[0].ptr(),dec.mi_lstm_w_ih[1].ptr(),dec.mi_lstm_w_ih[2].ptr(),dec.mi_lstm_w_ih[3].ptr()};
    const float* w_hh[4]={dec.mi_lstm_w_hh[0].ptr(),dec.mi_lstm_w_hh[1].ptr(),dec.mi_lstm_w_hh[2].ptr(),dec.mi_lstm_w_hh[3].ptr()};
    const float* b_ih[4]={dec.mi_lstm_b_ih[0].ptr(),dec.mi_lstm_b_ih[1].ptr(),dec.mi_lstm_b_ih[2].ptr(),dec.mi_lstm_b_ih[3].ptr()};
    const float* b_hh[4]={dec.mi_lstm_b_hh[0].ptr(),dec.mi_lstm_b_hh[1].ptr(),dec.mi_lstm_b_hh[2].ptr(),dec.mi_lstm_b_hh[3].ptr()};
    float* lstm_out = (float*)malloc(nf*512*sizeof(float));
    lstm_forward(mi_buf, nf, 512, 4, w_ih, w_hh, b_ih, b_hh, lstm_out, true);
    
    // Linear 512->128
    float* bn_out = (float*)malloc(nf*128*sizeof(float));
    for(int t=0;t<nf;t++) for(int o=0;o<128;o++){float s=dec.mi_b2.ptr()[o];
        for(int i=0;i<512;i++) s+=lstm_out[t*512+i]*dec.mi_w2.ptr()[o*512+i]; bn_out[t*128+o]=s;}
    
    // RMS
    float r=0; for(int i=0;i<nf*128;i++) r+=bn_out[i]*bn_out[i];
    printf("C++ bottleneck RMS: %.4f\n", sqrtf(r/(nf*128)));
    
    // Dump for Python comparison (time-major layout: [T, C])
    f=fopen("debug/cpp_bottleneck.bin","wb"); fwrite(bn_out,sizeof(float),nf*128,f); fclose(f);
    printf("Dumped to debug/cpp_bottleneck.bin\n");
    
    free(buf); free(mi_buf); free(lstm_out); free(bn_out);
    return 0;
}
