#pragma once

// dots.tts.cpp - Minimal safetensors file loader
// Format: [u64 header_len][JSON header][raw tensor data...]
// No external deps — just fopen/fread + simple JSON parser

#include "ggml.h"
#include <string>
#include <vector>
#include <cstdint>

struct st_tensor_info {
    std::string name;
    std::string dtype;       // "F32" or "F16"
    std::vector<int64_t> shape;
    uint64_t data_offset;    // byte offset in file
    uint64_t data_length;    // length in bytes
};

class SafeTensorsFile {
public:
    bool open(const char * path);
    void close();

    // List all tensors
    const std::vector<st_tensor_info> & tensors() const { return tensors_; }

    // Find a tensor by name
    const st_tensor_info * find(const char * name) const;

    // Load tensor data into a ggml tensor (must be pre-allocated with correct shape)
    // Automatically converts F16 -> F32 if needed
    bool load_tensor(const st_tensor_info & info, ggml_tensor * dst);

    // Load and create a new ggml tensor
    ggml_tensor * load_tensor(ggml_context * ctx, const st_tensor_info & info);

private:
    FILE * fp_ = nullptr;
    std::vector<st_tensor_info> tensors_;
    uint64_t data_start_ = 0;  // file offset where tensor data begins

    // Minimal JSON parser (no library needed)
    static std::string json_get_string(const char * & p);
    static int64_t json_get_int(const char * & p);
    static void json_skip_whitespace(const char * & p);
    static void json_expect(const char * & p, char c);
};
