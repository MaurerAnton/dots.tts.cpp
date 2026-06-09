// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer
// DiT intermediate tensor dumping for calibration

#pragma once
#include "ggml.h"
#include <cstdio>
#include <cmath>

#define DIT_DUMP_MAX 64

struct dit_dump_slot {
    const char * name;
    ggml_tensor * tensor;
    int n_elements;  // 0 = auto-detect from tensor shape
};

struct dit_dump_ctx {
    dit_dump_slot slots[DIT_DUMP_MAX];
    int count;
    bool enabled;
    
    dit_dump_ctx() : count(0), enabled(false) {}
    
    void add(const char * name, ggml_tensor * t, int n = 0) {
        if (!enabled || count >= DIT_DUMP_MAX) return;
        slots[count].name = name;
        slots[count].tensor = t;
        slots[count].n_elements = n;
        count++;
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
            fwrite(slots[i].tensor->data, sizeof(float), n, f);
            fclose(f);
            float rms = 0;
            float * d = (float*)slots[i].tensor->data;
            for (int j = 0; j < n; j++) rms += d[j] * d[j];
            printf("  DiT dump %s: %d elems, rms=%.4f\n", slots[i].name, n, sqrtf(rms/n));
        }
        count = 0; // reset after write
    }
};
