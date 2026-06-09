// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// DiT intermediate tensor dumping for calibration

#pragma once
#include "ggml.h"
#include <cstdio>
#include <cmath>
#include <cstring>

#define DIT_DUMP_MAX 128

struct dit_dump_slot {
    char name[64];
    ggml_tensor * tensor;
    int n_elements;
    float * copy;  // captured copy of tensor data
    dit_dump_slot() : tensor(nullptr), n_elements(0), copy(nullptr) { name[0] = '\0'; }
    ~dit_dump_slot() { delete[] copy; }
};

struct dit_dump_ctx {
    dit_dump_slot slots[DIT_DUMP_MAX];
    int count;
    bool enabled;
    
    dit_dump_ctx() : count(0), enabled(false) {}
    
    void add(const char * name, ggml_tensor * t, int n = 0) {
        if (!enabled || count >= DIT_DUMP_MAX) return;
        strncpy(slots[count].name, name, sizeof(slots[count].name) - 1);
        slots[count].name[sizeof(slots[count].name) - 1] = '\0';
        slots[count].tensor = t;
        slots[count].n_elements = n;
        count++;
    }
    
    void capture() {
        // Call AFTER ggml_graph_compute, copies tensor data before it gets overwritten
        if (!enabled || count == 0) return;
        for (int i = 0; i < count; i++) {
            if (!slots[i].tensor || !slots[i].tensor->data) continue;
            if (slots[i].copy) continue; // already captured
            int n = slots[i].n_elements;
            if (n == 0) { n = 1; for (int d = 0; d < 4; d++) if (slots[i].tensor->ne[d] > 0) n *= slots[i].tensor->ne[d]; }
            slots[i].copy = new float[n];
            memcpy(slots[i].copy, slots[i].tensor->data, n * sizeof(float));
        }
    }
    
    void write_all(const char * dir = "debug") {
        if (!enabled || count == 0) return;
        for (int i = 0; i < count; i++) {
            if (!slots[i].tensor || !slots[i].tensor->data) continue;
            char path[256];
            snprintf(path, sizeof(path), "%s/%s.bin", dir, slots[i].name);
            FILE * f = fopen(path, "wb");
            if (!f) continue;
            int n = slots[i].n_elements;
            if (n == 0) {
                n = 1;
                for (int d = 0; d < 4; d++)
                    if (slots[i].tensor->ne[d] > 0) n *= slots[i].tensor->ne[d];
            }
            float * data = slots[i].copy ? slots[i].copy : (float*)slots[i].tensor->data;
            fwrite(data, sizeof(float), n, f);
            fclose(f);
            float rms = 0;
            for (int j = 0; j < n; j++) rms += data[j] * data[j];
            printf("  DiT dump %s: %d elems, rms=%.4f\n", slots[i].name, n, sqrtf(rms/n));
        }
        count = 0; // reset after write
    }
};
