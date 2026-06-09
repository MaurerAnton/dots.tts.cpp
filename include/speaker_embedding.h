// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// speaker_embedding.h — Deterministic pseudo-speaker embeddings
// Without the full ECAPA-TDNN CNN, we generate embeddings from speaker name hash.
// Different names → different consistent voices.
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

// Generate a deterministic pseudo-speaker embedding from a name string.
// Uses djb2 hash → PCG random → normal distribution → 512-dim embedding.
// Same name always produces the same voice.
inline void speaker_embedding_from_name(const char * name, float * emb_out, int dim = 512) {
    // djb2 hash
    uint32_t hash = 5381;
    for (const char * p = name; *p; p++) hash = ((hash << 5) + hash) + (unsigned char)*p;

    // PCG-like pseudo-random generator
    uint64_t state = hash;
    auto randf = [&]() -> float {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float)((state >> 33) & 0x3FFFFFFF) / (float)0x3FFFFFFF;
    };

    // Box-Muller: uniform → normal
    for (int i = 0; i < dim; i += 2) {
        float u1 = randf() * 0.999f + 0.001f;
        float u2 = randf() * 0.999f + 0.001f;
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * 3.1415926535f * u2;
        emb_out[i] = r * cosf(theta);
        if (i + 1 < dim) emb_out[i + 1] = r * sinf(theta);
    }

    // Normalize to unit length
    float norm = 0;
    for (int i = 0; i < dim; i++) norm += emb_out[i] * emb_out[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (int i = 0; i < dim; i++) emb_out[i] /= norm;
    }
}
