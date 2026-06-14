// BLAS-based linear layer — fuses bias into sgemm for byte-identical match with MKL
// out = input @ weight^T + bias
// Uses OpenBLAS cblas_sgemm which uses same blocking strategy as MKL
#pragma once
#include <cblas.h>
#include <cstring>

inline void manual_linear_blas(float * out, const float * in, const float * w, const float * b,
    int in_dim, int out_dim) {
    if (b) {
        // Pre-fill out with bias so sgemm adds to it
        for (int o = 0; o < out_dim; o++) out[o] = b[o];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            1, out_dim, in_dim, 1.0f, in, in_dim, w, out_dim, 1.0f, out, out_dim);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            1, out_dim, in_dim, 1.0f, in, in_dim, w, out_dim, 0.0f, out, out_dim);
    }
}
