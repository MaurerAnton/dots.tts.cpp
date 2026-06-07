#pragma once
// dots.tts.cpp - internal utilities

#include "ggml.h"

// Get writable pointer to tensor data (accounting for view offset)
inline float * tensor_data(ggml_tensor * t) {
    return (float *)((char *)t->data + t->view_offs);
}

inline const float * tensor_data_const(const ggml_tensor * t) {
    return (const float *)((const char *)t->data + t->view_offs);
}
