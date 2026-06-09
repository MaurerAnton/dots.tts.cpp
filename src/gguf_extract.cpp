// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// gguf_extract — Extract one tensor from GGUF to binary file
// Uses ggml's built-in GGUF reader. Zero Python dependency.
#include "gguf.h"
#include "ggml.h"
#include <cstdio>

bool extract_embeddings(const char * gguf_path, const char * out_path) {
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context * ctx = gguf_init_from_file(gguf_path, params);
    if (!ctx) return false;

    int64_t tidx = gguf_find_tensor(ctx, "token_embd.weight");
    if (tidx < 0) { gguf_free(ctx); return false; }

    const char * name = gguf_get_tensor_name(ctx, tidx);
    uint64_t offset = gguf_get_tensor_offset(ctx, tidx);
    size_t nbytes = gguf_get_tensor_size(ctx, tidx);

    printf("  Extracting '%s': %zu bytes...\n", name, nbytes);

    FILE * in = fopen(gguf_path, "rb");
    if (!in) { gguf_free(ctx); return false; }
    fseek(in, offset, SEEK_SET);

    FILE * out = fopen(out_path, "wb");
    if (!out) { fclose(in); gguf_free(ctx); return false; }

    char buf[262144];
    for (size_t rem = nbytes; rem > 0;) {
        size_t chunk = rem < sizeof(buf) ? rem : sizeof(buf);
        fread(buf, 1, chunk, in);
        fwrite(buf, 1, chunk, out);
        rem -= chunk;
    }
    fclose(out); fclose(in); gguf_free(ctx);
    printf("  Wrote %s (%.1f MB)\n", out_path, nbytes/(1024.0*1024.0));
    return true;
}
