// Dump C++ conv_pre with channel-major layout for byte comparison
#include "bigvgan_cpp.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

void conv1d_causal(float*,const float*,int,int,const float*,const float*,int,int,int=1);
void lstm_forward(const float*,int,int,int,const float**,const float**,const float**,const float**,float*,bool);

int main() {
    BigVGANDecoder dec;
    if (!bigvgan_load("models/vocoder_eff.safetensors", dec)) return 1;
    
    float latents[128*128]; int nf=0;
    FILE* f = fopen("build/latents.bin","rb");
    while(fread(latents+nf*128,sizeof(float),128,f)==128) nf++;
    fclose(f);
    
    float* buf=(float*)malloc(nf*512*sizeof(float));
    conv1d_causal(buf,latents,128,nf,dec.post_proj_w.ptr(),dec.post_proj_b.ptr(),128,1);
    
    float* mi_buf=(float*)malloc(nf*512*sizeof(float));
    for(int t=0;t<nf;t++) for(int o=0;o<512;o++){float s=dec.mi_b1.ptr()[o];
        for(int i=0;i<128;i++) s+=buf[t*128+i]*dec.mi_w1.ptr()[o*128+i]; mi_buf[t*512+o]=s;}
    
    const float* w_ih[4]={dec.mi_lstm_w_ih[0].ptr(),dec.mi_lstm_w_ih[1].ptr(),dec.mi_lstm_w_ih[2].ptr(),dec.mi_lstm_w_ih[3].ptr()};
    const float* w_hh[4]={dec.mi_lstm_w_hh[0].ptr(),dec.mi_lstm_w_hh[1].ptr(),dec.mi_lstm_w_hh[2].ptr(),dec.mi_lstm_w_hh[3].ptr()};
    const float* b_ih[4]={dec.mi_lstm_b_ih[0].ptr(),dec.mi_lstm_b_ih[1].ptr(),dec.mi_lstm_b_ih[2].ptr(),dec.mi_lstm_b_ih[3].ptr()};
    const float* b_hh[4]={dec.mi_lstm_b_hh[0].ptr(),dec.mi_lstm_b_hh[1].ptr(),dec.mi_lstm_b_hh[2].ptr(),dec.mi_lstm_b_hh[3].ptr()};
    float* lstm_out=(float*)malloc(nf*512*sizeof(float));
    lstm_forward(mi_buf,nf,512,4,w_ih,w_hh,b_ih,b_hh,lstm_out,true);
    
    float* tmp=(float*)malloc(nf*128*sizeof(float));
    for(int t=0;t<nf;t++) for(int o=0;o<128;o++){float s=dec.mi_b2.ptr()[o];
        for(int i=0;i<512;i++) s+=lstm_out[t*512+i]*dec.mi_w2.ptr()[o*512+i]; tmp[t*128+o]=s;}
    
    // conv_pre with channel-major layout
    int ic=128,oc=1536,K=5,pad=2,padded_len=nf+4;
    float* padded=(float*)calloc(ic*padded_len,sizeof(float));
    for(int c=0;c<ic;c++) for(int t=0;t<nf;t++) padded[c*padded_len+(t+pad)]=tmp[t*ic+c];
    float* cp_out=(float*)malloc(nf*oc*sizeof(float));
    for(int o=0;o<oc;o++){float b=dec.conv_pre_b.ptr()?dec.conv_pre_b.ptr()[o]:0;
        for(int t=0;t<nf;t++){float s=b;
            for(int k=0;k<K;k++){int pt=t+k;
                for(int c=0;c<ic;c++) s+=padded[c*padded_len+pt]*dec.conv_pre_w.ptr()[(o*ic+c)*K+k];}
            cp_out[t*oc+o]=s;}}
    
    float r=0; for(int i=0;i<nf*oc;i++) r+=cp_out[i]*cp_out[i];
    printf("C++ conv_pre RMS: %.4f\n", sqrtf(r/(nf*oc)));
    
    // Dump as time-major for Python comparison
    f=fopen("debug/cpp_conv_pre.bin","wb"); fwrite(cp_out,sizeof(float),nf*oc,f); fclose(f);
    printf("Dumped debug/cpp_conv_pre.bin (time-major [T,C])\n");
    
    free(buf);free(mi_buf);free(lstm_out);free(tmp);free(padded);free(cp_out);
    return 0;
}
