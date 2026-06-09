// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// Manual DiT operations: RMS norm, linear, RoPE, softmax, attention, block
// Pure C++ — no ggml dependency. Used by both test and production code.
#pragma once
#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

// LayerNorm (NOT RMS norm!): out = (x - mean) / sqrt(var + eps)
// elementwise_affine=False for norm1/norm2, so no weight/bias
inline void manual_layernorm(float * out, const float * x, int n) {
    float mean = 0; for (int i = 0; i < n; i++) mean += x[i]; mean /= n;
    float var = 0; for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    float inv_std = 1.0f / sqrtf(var / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv_std;
}

// RMS Norm (used for qk_norm inside attention)
inline void manual_rms_norm(float * out, const float * x, const float * weight, int n) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) { float v = x[i] * inv; if (weight) v *= weight[i]; out[i] = v; }
}

// Linear: out[o] = sum_i W[o,i] * x[i] + bias[o]  (PyTorch-direct access)
inline void manual_linear(float * out, const float * x, const float * w, const float * bias,
                          int in_feat, int out_feat) {
    for (int o = 0; o < out_feat; o++) { float s = bias ? bias[o] : 0.0f;
        for (int i = 0; i < in_feat; i++) s += w[o * in_feat + i] * x[i]; out[o] = s; }
}

// RoPE
inline void manual_rope(float * q, float * k, int seq_len, int head_dim) {
    for (int s = 0; s < seq_len; s++) { float theta = 1.0f; float ts = powf(10000.0f, -2.0f/(float)head_dim);
        for (int d = 0; d < head_dim; d += 2) { float cs = cosf((float)s*theta), sn = sinf((float)s*theta);
            int idx = s*head_dim+d; float q0=q[idx], q1=q[idx+1], k0=k[idx], k1=k[idx+1];
            q[idx]=q0*cs-q1*sn; q[idx+1]=q1*cs+q0*sn; k[idx]=k0*cs-k1*sn; k[idx+1]=k1*cs+k0*sn; theta*=ts; } }
}

// Softmax in-place
inline void manual_softmax(float * x, int n) {
    float mx = x[0]; for (int i=1;i<n;i++) if(x[i]>mx) mx=x[i]; float sum=0;
    for (int i=0;i<n;i++) { x[i]=expf(x[i]-mx); sum+=x[i]; } for (int i=0;i<n;i++) x[i]/=sum;
}

// Full multi-head attention (scaled dot-product, no flash)
inline void manual_attention(float * out, const float * x,
    const float * qw, const float * kw, const float * vw, const float * ow,
    const float * qb, const float * kb, const float * vb, const float * ob,
    const float * qnw, const float * knw,
    int n_tokens, int hidden, int n_heads, int head_dim) {
    int n = n_tokens * hidden;
    float * q = new float[n], * k = new float[n], * v = new float[n];
    // QKV projections (per-token)
    for (int t = 0; t < n_tokens; t++) {
        manual_linear(q + t*hidden, x + t*hidden, qw, qb, hidden, hidden);
        manual_linear(k + t*hidden, x + t*hidden, kw, kb, hidden, hidden);
        manual_linear(v + t*hidden, x + t*hidden, vw, vb, hidden, hidden);
    }

    // DEBUG first call
    { static int cnt=0; if(cnt==0){
        float r=0; for(int i=0;i<n_tokens*hidden;i++) r+=q[i]*q[i];
        fprintf(stderr,"  attn_q: rms=%.4f first3=[%.4f,%.4f,%.4f]\n", sqrtf(r/(n_tokens*hidden)), q[0], q[1], q[2]);
        r=0; for(int i=0;i<n_tokens*hidden;i++) r+=k[i]*k[i];
        fprintf(stderr,"  attn_k: rms=%.4f first3=[%.4f,%.4f,%.4f]\n", sqrtf(r/(n_tokens*hidden)), k[0], k[1], k[2]);
        r=0; for(int i=0;i<n_tokens*hidden;i++) r+=v[i]*v[i];
        fprintf(stderr,"  attn_v: rms=%.4f first3=[%.4f,%.4f,%.4f]\n", sqrtf(r/(n_tokens*hidden)), v[0], v[1], v[2]);
    } cnt++; }
    float * scores = new float[n_tokens * n_tokens];
    float * ao_flat = new float[n];
    for (int h = 0; h < n_heads; h++) {
        float * qh = new float[n_tokens*head_dim], * kh = new float[n_tokens*head_dim], * vh = new float[n_tokens*head_dim];
        for (int t=0;t<n_tokens;t++) for (int d=0;d<head_dim;d++) {
            qh[t*head_dim+d]=q[t*hidden+h*head_dim+d]; kh[t*head_dim+d]=k[t*hidden+h*head_dim+d];
            vh[t*head_dim+d]=v[t*hidden+h*head_dim+d]; }
        for (int t=0;t<n_tokens;t++) { manual_rms_norm(qh+t*head_dim, qh+t*head_dim, qnw, head_dim);
            manual_rms_norm(kh+t*head_dim, kh+t*head_dim, knw, head_dim); }
        static int dcnt2=0; if(dcnt2==0 && h==0){
            float r=0; for(int i=0;i<n_tokens*head_dim;i++) r+=qh[i]*qh[i];
            fprintf(stderr,"  qh0_norm: rms=%.4f first3=[%.4f,%.4f,%.4f]\n", sqrtf(r/(n_tokens*head_dim)), qh[0], qh[1], qh[2]);
        } dcnt2++;
        manual_rope(qh, kh, n_tokens, head_dim);
        static int dcnt=0; if(dcnt==0 && h==0){
            float r=0; for(int i=0;i<n_tokens*head_dim;i++) r+=qh[i]*qh[i];
            fprintf(stderr,"  qh0_rope: rms=%.4f first3=[%.4f,%.4f,%.4f]\n", sqrtf(r/(n_tokens*head_dim)), qh[0], qh[1], qh[2]);
        } dcnt++;
        float scale=1.0f/sqrtf((float)head_dim);
        for(int i=0;i<n_tokens;i++){ for(int j=0;j<n_tokens;j++){ float s=0;
            for(int d=0;d<head_dim;d++) s+=qh[i*head_dim+d]*kh[j*head_dim+d]; scores[i*n_tokens+j]=s*scale; }
            for(int j=i+1;j<n_tokens;j++) scores[i*n_tokens+j]=-INFINITY;
            manual_softmax(scores+i*n_tokens,n_tokens); }
        static int dcnt3=0; if(dcnt3==0 && h==0){
            fprintf(stderr,"  scores_h0 row0: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                scores[0],scores[1],scores[2],scores[3],scores[4],scores[5],scores[6],scores[7]);
        } dcnt3++;
        for(int i=0;i<n_tokens;i++) for(int d=0;d<head_dim;d++){ float s=0;
            for(int j=0;j<n_tokens;j++) s+=scores[i*n_tokens+j]*vh[j*head_dim+d]; ao_flat[i*hidden+h*head_dim+d]=s; }
        delete[] qh; delete[] kh; delete[] vh;
    }
    delete[] scores;
    // O projection (per-token)
    for (int t = 0; t < n_tokens; t++)
        manual_linear(out + t*hidden, ao_flat + t*hidden, ow, ob, hidden, hidden);
    { static int dcnt4=0; if(dcnt4==0){
        float r=0; for(int i=0;i<n_tokens*hidden;i++) r+=out[i]*out[i];
        fprintf(stderr,"  attn_out: rms=%.4f first3=[%.4f,%.4f,%.4f]\n", sqrtf(r/(n_tokens*hidden)), out[0], out[1], out[2]);
    } dcnt4++; }
    delete[] q; delete[] k; delete[] v; delete[] ao_flat;
}

// Full manual DiT block
inline void manual_dit_block(const float * x_in, const float * cond, const dit_block & block,
    float * out, int n_tokens) {
    int hidden = DIT_HIDDEN_SIZE;
    float * cs = new float[1024]; for(int i=0;i<1024;i++) cs[i]=cond[i]/(1.0f+expf(-cond[i]));
    float * adaln_raw = new float[6*DIT_HIDDEN_SIZE];
    { float * aw = tensor_data(block.adaln_linear_w); float * ab = block.adaln_linear_b?tensor_data(block.adaln_linear_b):nullptr;
      for(int o=0;o<6*hidden;o++){ float s=ab?ab[o]:0.0f; for(int i=0;i<DIT_HIDDEN_SIZE;i++) s+=aw[o*DIT_HIDDEN_SIZE+i]*cs[i]; adaln_raw[o]=s; } }
    float * sm=adaln_raw, * scm=adaln_raw+hidden, * gm=adaln_raw+2*hidden;
    float * sml=adaln_raw+3*hidden, * scl=adaln_raw+4*hidden, * gml=adaln_raw+5*hidden;
    float * normed = new float[n_tokens*hidden], * mod = new float[n_tokens*hidden], * ao = new float[n_tokens*hidden];
    for(int t=0;t<n_tokens;t++){ manual_layernorm(normed+t*hidden, x_in+t*hidden, hidden);
        if(t==0){ float r=0; for(int i=0;i<hidden;i++){ float v=normed[i]*(1.0f+scm[i])+sm[i]; r+=v*v; mod[t*hidden+i]=v; }
          fprintf(stderr,"  manual_mod t0: rms=%.4f first3=[%.4f,%.4f,%.4f]\n", sqrtf(r/hidden), mod[0], mod[1], mod[2]); }
        else for(int i=0;i<hidden;i++) mod[t*hidden+i]=normed[t*hidden+i]*(1.0f+scm[i])+sm[i]; }
    manual_attention(ao, mod, tensor_data(block.attn_q_weight), tensor_data(block.attn_k_weight),
        tensor_data(block.attn_v_weight), tensor_data(block.attn_o_weight),
        nullptr,nullptr,nullptr, block.attn_o_bias?tensor_data(block.attn_o_bias):nullptr,
        block.q_norm_w?tensor_data(block.q_norm_w):nullptr, block.k_norm_w?tensor_data(block.k_norm_w):nullptr,
        n_tokens,hidden,DIT_NUM_HEADS,DIT_HEAD_SIZE);
    float * h = new float[n_tokens*hidden];
    for(int i=0;i<n_tokens*hidden;i++) h[i]=x_in[i]+gm[i%hidden]*ao[i];
    { static int dcnt6=0; if(dcnt6==0){
        float r=0; for(int i=0;i<n_tokens*hidden;i++) r+=h[i]*h[i];
        fprintf(stderr,"  after_attn: rms=%.4f first3=[%.4f,%.4f,%.4f]\n", sqrtf(r/(n_tokens*hidden)), h[0], h[1], h[2]);
    } dcnt6++; }
    for(int t=0;t<n_tokens;t++){ manual_layernorm(normed+t*hidden, h+t*hidden, hidden);
        for(int i=0;i<hidden;i++) mod[t*hidden+i]=normed[t*hidden+i]*(1.0f+scl[i])+sml[i]; }
    float * fh1 = new float[n_tokens*DIT_FFN_SIZE], * fh2 = new float[n_tokens*hidden];
    float * fw1=tensor_data(block.ffn_w1), * fw2=tensor_data(block.ffn_w2);
    float * fb1=block.ffn_b1?tensor_data(block.ffn_b1):nullptr, * fb2=block.ffn_b2?tensor_data(block.ffn_b2):nullptr;
    for(int t=0;t<n_tokens;t++){ manual_linear(fh1+t*DIT_FFN_SIZE, mod+t*hidden, fw1, fb1, hidden, DIT_FFN_SIZE);
        static int dc7=0; if(dc7==0){
            float r=0; for(int i=0;i<DIT_FFN_SIZE;i++) r+=fh1[i]*fh1[i];
            fprintf(stderr,"  fc1_out: rms=%.4f first3=[%.4f,%.4f,%.4f]\n", sqrtf(r/DIT_FFN_SIZE), fh1[0], fh1[1], fh1[2]);
        } dc7++;
        for(int i=0;i<DIT_FFN_SIZE;i++){ float xv=fh1[t*DIT_FFN_SIZE+i];
            // GELU: x * Phi(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float x2=xv*xv, x3=x2*xv;
            fh1[t*DIT_FFN_SIZE+i]=0.5f*xv*(1.0f+tanhf(0.7978845608f*(xv+0.044715f*x3))); }
        static int dc8=0; if(dc8==0){
            float r=0; for(int i=0;i<DIT_FFN_SIZE;i++) r+=fh1[i]*fh1[i];
            fprintf(stderr,"  silu_out: rms=%.4f first3=[%.4f,%.4f,%.4f]\n", sqrtf(r/DIT_FFN_SIZE), fh1[0], fh1[1], fh1[2]);
        } dc8++;
        manual_linear(fh2+t*hidden, fh1+t*DIT_FFN_SIZE, fw2, fb2, DIT_FFN_SIZE, hidden);
        static int dc9=0; if(dc9==0){
            float r=0; for(int i=0;i<hidden;i++) r+=fh2[i]*fh2[i];
            fprintf(stderr,"  fc2_out: rms=%.4f first3=[%.4f,%.4f,%.4f]\n", sqrtf(r/hidden), fh2[0], fh2[1], fh2[2]);
        } dc9++; }
    for(int i=0;i<n_tokens*hidden;i++) out[i]=h[i]+gml[i%hidden]*fh2[i];
    { static int dcnt5=0; if(dcnt5==0){
        float r=0; for(int i=0;i<n_tokens*hidden;i++) r+=fh2[i]*fh2[i];
            fprintf(stderr,"  ffn_out: rms=%.4f first3=[%.4f,%.4f,%.4f]\n", sqrtf(r/(n_tokens*hidden)), fh2[0], fh2[1], fh2[2]);
    } dcnt5++; }
    // CLEANUP
    delete[] cs; delete[] adaln_raw;
    delete[] normed; delete[] mod; delete[] ao; delete[] h; delete[] fh1; delete[] fh2;
}
