// BLAS-accelerated linear layer matching PyTorch's matmul precision
#pragma once
#ifdef USE_OPENBLAS
#include <cblas.h>
inline void blas_linear(float * out, const float * x, const float * w, const float * bias, int in_feat, int out_feat, int batch) {
    // out [batch, out_feat] = x [batch, in_feat] @ w^T [out_feat, in_feat] + bias
    // C = A @ B^T with A=[batch, in_feat], B=[out_feat, in_feat]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch, out_feat, in_feat,
                1.0f, x, in_feat, w, in_feat,
                0.0f, out, out_feat);
    if (bias) {
        for (int b = 0; b < batch; b++)
            for (int o = 0; o < out_feat; o++)
                out[b * out_feat + o] += bias[o];
    }
}
#else
// Fallback: manual linear (triple loop, row-major)
inline void blas_linear(float * out, const float * x, const float * w, const float * bias, int in_feat, int out_feat, int batch) {
    if (bias) {
        for (int b = 0; b < batch; b++)
            for (int o = 0; o < out_feat; o++) {
                float s = bias[o];
                for (int i = 0; i < in_feat; i++) s += w[o * in_feat + i] * x[b * in_feat + i];
                out[b * out_feat + o] = s;
            }
    } else {
        for (int b = 0; b < batch; b++)
            for (int o = 0; o < out_feat; o++) {
                float s = 0;
                for (int i = 0; i < in_feat; i++) s += w[o * in_feat + i] * x[b * in_feat + i];
                out[b * out_feat + o] = s;
            }
    }
}
#endif
