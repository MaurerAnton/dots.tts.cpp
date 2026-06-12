// Minimal MT19937 matching Python's torch.manual_seed RNG
#pragma once
#include <cstdint>
#include <cmath>

struct MT19937 {
    uint32_t mt[624];
    int mti;
    void seed(uint32_t s) {
        mt[0] = s;
        for (mti = 1; mti < 624; mti++)
            mt[mti] = 1812433253U * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti;
    }
    uint32_t next() {
        if (mti >= 624) {
            for (int i = 0; i < 624; i++) {
                uint32_t y = (mt[i] & 0x80000000U) + (mt[(i+1)%624] & 0x7fffffffU);
                mt[i] = mt[(i+397)%624] ^ (y >> 1);
                if (y & 1) mt[i] ^= 0x9908b0dfU;
            }
            mti = 0;
        }
        uint32_t y = mt[mti++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }
    float uniform() { return (next() >> 5) * (1.0f / 134217728.0f); }
    float normal() {
        float u1 = uniform(), u2 = uniform();
        if (u1 < 1e-6f) u1 = 1e-6f;
        if (u2 < 1e-6f) u2 = 1e-6f;
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    }
};
