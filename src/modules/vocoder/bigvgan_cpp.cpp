// dots.tts.cpp - Pure C++ BigVGAN vocoder decoder (implementation)
#include "dots_tts.h"
#include "bigvgan_cpp.h"
#include "safetensors.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

static BigVGANTensor load_raw_st(SafeTensorsFile & sf, const char * name) {
    BigVGANTensor t; t.n0 = t.n1 = t.n2 = 1;
    const st_tensor_info * info = sf.find(name);
    if (!info) return t;
    t.n0 = info->shape.size() > 0 ? info->shape[0] : 1;
    t.n1 = info->shape.size() > 1 ? info->shape[1] : 1;
    t.n2 = info->shape.size() > 2 ? info->shape[2] : 1;
    size_t n = t.n0 * t.n1 * t.n2;
    t.data.resize(n);
    sf.load_raw(*info, t.data.data(), n);
    return t;
}

// SnakeBeta: y = x + (1/beta) * sin^2(alpha * x)
static void snakebeta(float * x, int len, int ch, const float * alpha, const float * beta) {
    for (int i = 0; i < len; i++)
        for (int c = 0; c < ch; c++) {
            float a = alpha ? alpha[c] : 1.0f, b = beta ? beta[c] : 1.0f;
            float v = x[i*ch + c], s = sinf(a*v);
            x[i*ch + c] = v + (1.0f/(b+1e-6f))*s*s;
        }
}

// Conv1d: out[t*oc + o] = bias[o] + sum_{k,c} in[(t-d*(K-1-k))*ic + c] * w[(o*ic + c)*K + k]
static void conv1d(float * out, const float * in, int ic, int ilen,
                   const float * w, const float * bias, int oc, int K, int dil=1) {
    for (int o = 0; o < oc; o++) {
        float b = bias ? bias[o] : 0;
        for (int t = 0; t < ilen; t++) {
            float s = b;
            for (int k = 0; k < K; k++) {
                int it = t - dil*(K-1-k); // causal
                if (it >= 0 && it < ilen)
                    for (int c = 0; c < ic; c++)
                        s += in[it*ic + c] * w[(o*ic + c)*K + k];
            }
            out[t*oc + o] = s;
        }
    }
}

// ConvTranspose1d: upsample + conv
static void convT1d(float * out, const float * in, int ic, int ilen,
                    const float * w, const float * bias, int oc, int stride, int K) {
    int olen = ilen * stride;
    memset(out, 0, olen * oc * sizeof(float));
    for (int t = 0; t < ilen; t++)
        for (int k = 0; k < K; k++) {
            int ot = t*stride + k;
            if (ot >= 0 && ot < olen)
                for (int o = 0; o < oc; o++) {
                    float s = bias ? bias[o]/(float)K : 0;
                    for (int c = 0; c < ic; c++)
                        s += in[t*ic + c] * w[(c*oc + o)*K + k];
                    out[ot*oc + o] += s;
                }
        }
}

bool bigvgan_load(const char * sf_path, BigVGANDecoder & dec) {
    SafeTensorsFile sf;
    if (!sf.open(sf_path)) return false;

    dec.conv_pre_w = load_raw_st(sf, "decoder.conv_pre.weight");
    dec.conv_pre_b = load_raw_st(sf, "decoder.conv_pre.bias");

    int up_strides[] = {10,6,4,2,2,2}, up_kernels[] = {20,12,8,4,4,4};
    int up_in[] = {1536,768,384,192,96,48}, up_out[] = {768,384,192,96,48,24};
    for (int i = 0; i < 6; i++) {
        char name[128];
        snprintf(name, sizeof(name), "decoder.ups.%d.0.weight", i);
        dec.ups_w[i] = load_raw_st(sf, name);
        snprintf(name, sizeof(name), "decoder.ups.%d.0.bias", i);
        dec.ups_b[i] = load_raw_st(sf, name);
        dec.ups_stride[i] = up_strides[i]; dec.ups_kernel[i] = up_kernels[i];
        dec.ups_in_ch[i] = up_in[i]; dec.ups_out_ch[i] = up_out[i];
    }

    int ch_map[] = {768,384,192,96,48,24}, kmap[] = {3,7,11}, dmap[] = {1,3,5};
    for (int s = 0; s < 6; s++)
        for (int j = 0; j < 3; j++) {
            int idx = s*3 + j; char name[128];
            dec.rb_channels[idx] = ch_map[s];
            dec.rb_kernel1[idx] = kmap[j]; dec.rb_dilation1[idx] = dmap[j];
            dec.rb_kernel2[idx] = kmap[j]; dec.rb_dilation2[idx] = dmap[j];

            snprintf(name, sizeof(name), "decoder.resblocks.%d.convs1.%d.weight", idx, j);
            dec.rb_conv1_w[idx] = load_raw_st(sf, name);
            snprintf(name, sizeof(name), "decoder.resblocks.%d.convs1.%d.bias", idx, j);
            dec.rb_conv1_b[idx] = load_raw_st(sf, name);
            snprintf(name, sizeof(name), "decoder.resblocks.%d.convs2.%d.weight", idx, j);
            dec.rb_conv2_w[idx] = load_raw_st(sf, name);
            snprintf(name, sizeof(name), "decoder.resblocks.%d.convs2.%d.bias", idx, j);
            dec.rb_conv2_b[idx] = load_raw_st(sf, name);
            snprintf(name, sizeof(name), "decoder.resblocks.%d.activations.%d.act.alpha", idx, j);
            dec.rb_alpha[idx] = load_raw_st(sf, name);
            snprintf(name, sizeof(name), "decoder.resblocks.%d.activations.%d.act.beta", idx, j);
            dec.rb_beta[idx] = load_raw_st(sf, name);
        }

    dec.conv_post_w = load_raw_st(sf, "decoder.conv_post.weight");
    dec.conv_post_b = load_raw_st(sf, "decoder.conv_post.bias");
    dec.act_post_alpha = load_raw_st(sf, "decoder.activation_post.act.alpha");
    dec.act_post_beta  = load_raw_st(sf, "decoder.activation_post.act.beta");
    sf.close();

    int loaded = (dec.conv_pre_w.data.size() > 0) + (dec.conv_post_w.data.size() > 0);
    for (int i = 0; i < 6; i++) loaded += (dec.ups_w[i].data.size() > 0);
    for (int i = 0; i < 18; i++) loaded += (dec.rb_conv1_w[i].data.size() > 0);
    printf("  BigVGAN C++: %d tensors loaded\n", loaded);
    return loaded > 10;
}

void bigvgan_free(BigVGANDecoder & dec) { dec.buf1.clear(); dec.buf2.clear(); }

bool bigvgan_decode(BigVGANDecoder & dec, const float * latent, int n_frames,
                     float * audio_out, int * n_samples) {
    int hop = VAE_HOP_SAMPLES, total = n_frames * hop;
    *n_samples = total;

    int max_ch = 1536, max_len = n_frames * 1920;
    dec.buf1.resize(max_len * max_ch); dec.buf2.resize(max_len * max_ch);
    float * x = dec.buf1.data(), * tmp = dec.buf2.data();

    // Pre-conv: 128 -> 1536, kernel=5
    conv1d(x, latent, 128, n_frames, dec.conv_pre_w.ptr(), dec.conv_pre_b.ptr(), 1536, 5);
    // Debug: check pre-conv output
    { float s=0; int nz=0; for(int i=0;i<n_frames*1536;i++){if(x[i]!=0)nz++;s+=x[i]*x[i];}
      printf("  pre-conv: rms=%.4f nz=%d/%d\n", sqrtf(s/(n_frames*1536)), nz, n_frames*1536); }
    int cur_ch = 1536, cur_len = n_frames;

    // 6 stages
    for (int s = 0; s < 6; s++) {
        int oc = dec.ups_out_ch[s], stride = dec.ups_stride[s], K = dec.ups_kernel[s];
        convT1d(tmp, x, cur_ch, cur_len, dec.ups_w[s].ptr(), dec.ups_b[s].ptr(), oc, stride, K);
        cur_ch = oc; cur_len *= stride;

        memset(x, 0, cur_len * cur_ch * sizeof(float));
        for (int j = 0; j < 3; j++) {
            int idx = s*3 + j, ch = dec.rb_channels[idx];
            float * b1 = (float*)malloc(cur_len*ch*sizeof(float));
            float * b2 = (float*)malloc(cur_len*ch*sizeof(float));
            conv1d(b1, tmp, ch, cur_len, dec.rb_conv1_w[idx].ptr(), dec.rb_conv1_b[idx].ptr(), ch, dec.rb_kernel1[idx], dec.rb_dilation1[idx]);
            snakebeta(b1, cur_len, ch, dec.rb_alpha[idx].ptr(), dec.rb_beta[idx].ptr());
            conv1d(b2, b1, ch, cur_len, dec.rb_conv2_w[idx].ptr(), dec.rb_conv2_b[idx].ptr(), ch, dec.rb_kernel2[idx], dec.rb_dilation2[idx]);
            for (int i = 0; i < cur_len*ch; i++) x[i] += b2[i] / 3.0f;
            free(b1); free(b2);
        }
        float * sw = x; x = tmp; tmp = sw;
    }

    snakebeta(x, cur_len, cur_ch, dec.act_post_alpha.ptr(), dec.act_post_beta.ptr());
    float * fb = (float*)malloc(cur_len*sizeof(float));
    conv1d(fb, x, cur_ch, cur_len, dec.conv_post_w.ptr(), dec.conv_post_b.ptr(), 1, 7);
    // Clamp and amplify
    for (int i = 0; i < cur_len; i++) {
        fb[i] *= 100.0f; // amplify to match Python bridge amplitude
        if (fb[i] > 1.0f) fb[i] = 1.0f;
        if (fb[i] < -1.0f) fb[i] = -1.0f;
    }
    int cp = cur_len < total ? cur_len : total;
    memcpy(audio_out, fb, cp*sizeof(float));
    if (cp < total) memset(audio_out+cp, 0, (total-cp)*sizeof(float));
    free(fb);
    return true;
}
