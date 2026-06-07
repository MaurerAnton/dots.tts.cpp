// dots.tts.cpp - Safetensors loader implementation
// Minimalist: no external JSON lib, no Python, just C++17 + stdio

#include "safetensors.h"
#include "dots_tts_util.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>

// ===========================================================================
// JSON micro-parser (just enough for safetensors headers)
// ===========================================================================

void SafeTensorsFile::json_skip_whitespace(const char * & p) {
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
}

void SafeTensorsFile::json_expect(const char * & p, char c) {
    json_skip_whitespace(p);
    if (*p != c) {
        fprintf(stderr, "JSON: expected '%c' at offset %td, got '%c'\n", c, p - p, *p);
        abort();
    }
    p++;
}

std::string SafeTensorsFile::json_get_string(const char * & p) {
    json_expect(p, '"');
    const char * start = p;
    while (*p && *p != '"') p++;
    std::string s(start, p - start);
    json_expect(p, '"');
    return s;
}

int64_t SafeTensorsFile::json_get_int(const char * & p) {
    json_skip_whitespace(p);
    char * end;
    int64_t v = strtoll(p, &end, 10);
    p = end;
    return v;
}

// ===========================================================================
// File operations
// ===========================================================================

bool SafeTensorsFile::open(const char * path) {
    fp_ = fopen(path, "rb");
    if (!fp_) {
        fprintf(stderr, "SafeTensors: cannot open %s\n", path);
        return false;
    }

    // Read header length (u64 LE)
    uint64_t header_len;
    if (fread(&header_len, 8, 1, fp_) != 1) {
        fprintf(stderr, "SafeTensors: failed to read header length\n");
        return false;
    }

    // Read JSON header
    char * header_buf = new char[header_len + 1];
    if (fread(header_buf, 1, header_len, fp_) != header_len) {
        fprintf(stderr, "SafeTensors: failed to read header (%llu bytes)\n",
                (unsigned long long)header_len);
        delete[] header_buf;
        return false;
    }
    header_buf[header_len] = '\0';

    // Parse JSON header
    const char * p = header_buf;
    json_expect(p, '{');

    // Skip first key (usually "__metadata__") and its value
    // We iterate through all top-level keys
    while (*p) {
        json_skip_whitespace(p);
        if (*p == '}') break;
        if (*p == ',') { p++; continue; }

        std::string key = json_get_string(p);
        json_expect(p, ':');
        json_expect(p, '{');

        // Parse tensor info
        st_tensor_info info;
        info.name = key;

        while (*p) {
            json_skip_whitespace(p);
            if (*p == '}') break;
            if (*p == ',') { p++; continue; }

            std::string field = json_get_string(p);
            json_expect(p, ':');

            if (field == "dtype") {
                info.dtype = json_get_string(p);
            } else if (field == "shape") {
                json_expect(p, '[');
                while (*p) {
                    json_skip_whitespace(p);
                    if (*p == ']') break;
                    if (*p == ',') { p++; continue; }
                    info.shape.push_back(json_get_int(p));
                }
                json_expect(p, ']');
            } else if (field == "data_offsets") {
                json_expect(p, '[');
                info.data_offset = json_get_int(p);
                json_expect(p, ',');
                info.data_length = json_get_int(p) - info.data_offset;
                json_expect(p, ']');
            } else {
                // Skip unknown field value
                json_skip_whitespace(p);
                if (*p == '"') { json_get_string(p); }
                else if (*p == '[') {
                    p++; int depth = 1;
                    while (*p && depth) {
                        if (*p == '[') depth++;
                        if (*p == ']') depth--;
                        p++;
                    }
                } else if (*p == '{') {
                    p++; int depth = 1;
                    while (*p && depth) {
                        if (*p == '{') depth++;
                        if (*p == '}') depth--;
                        p++;
                    }
                } else while (*p && *p != ',' && *p != '}') p++;
            }
        }
        json_expect(p, '}');

        if (!info.name.empty() && info.name != "__metadata__") {
            tensors_.push_back(info);
        }
    }

    delete[] header_buf;

    // Data starts after 8-byte header + JSON header
    data_start_ = 8 + header_len;

    printf("SafeTensors: %zu tensors found\n", tensors_.size());
    return true;
}

void SafeTensorsFile::close() {
    if (fp_) { fclose(fp_); fp_ = nullptr; }
    tensors_.clear();
}

const st_tensor_info * SafeTensorsFile::find(const char * name) const {
    for (auto & t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

// ===========================================================================
// Tensor loading
// ===========================================================================

bool SafeTensorsFile::load_tensor(const st_tensor_info & info, ggml_tensor * dst) {
    if (!fp_) return false;

    // Calculate expected element count
    size_t expected_elems = 1;
    for (auto d : info.shape) expected_elems *= d;

    size_t ggml_elems = 1;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (dst->ne[i] > 0) ggml_elems *= dst->ne[i];
    }

    if (expected_elems != ggml_elems) {
        fprintf(stderr, "SafeTensors: shape mismatch for %s: file has %zu, ggml has %zu\n",
                info.name.c_str(), expected_elems, ggml_elems);
        return false;
    }

    // Seek to tensor data
    uint64_t offset = data_start_ + info.data_offset;
    if (fseeko(fp_, (off_t)offset, SEEK_SET) != 0) {
        perror("fseeko");
        return false;
    }

    float * dst_data = tensor_data(dst);

    if (info.dtype == "F32") {
        // Direct read
        size_t nread = fread(dst_data, sizeof(float), expected_elems, fp_);
        if (nread != expected_elems) {
            fprintf(stderr, "SafeTensors: short read for %s\n", info.name.c_str());
            return false;
        }
    } else if (info.dtype == "F16") {
        // Read as f16, convert to f32
        uint16_t * buf16 = new uint16_t[expected_elems];
        size_t nread = fread(buf16, sizeof(uint16_t), expected_elems, fp_);
        if (nread != expected_elems) {
            delete[] buf16;
            return false;
        }
        for (size_t i = 0; i < expected_elems; i++) {
            dst_data[i] = ggml_fp16_to_fp32(buf16[i]);
        }
        delete[] buf16;
    } else {
        fprintf(stderr, "SafeTensors: unsupported dtype %s for %s\n",
                info.dtype.c_str(), info.name.c_str());
        return false;
    }

    return true;
}

ggml_tensor * SafeTensorsFile::load_tensor(ggml_context * ctx, const st_tensor_info & info) {
    if (!fp_) return nullptr;

    // Create ggml tensor with matching shape
    int n_dims = info.shape.size();
    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < n_dims && i < 4; i++) {
        ne[i] = info.shape[n_dims - 1 - i]; // Reverse: safetensors is row-major, ggml is column-major
    }

    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
        ne[0], ne[1], ne[2], ne[3]);

    if (!load_tensor(info, t)) {
        return nullptr;
    }

    return t;
}
