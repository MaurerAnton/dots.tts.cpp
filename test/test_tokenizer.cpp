// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// test_tokenizer.cpp — verify C++ BPE tokenizer matches Python
#include "gpt2_bpe_tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    const char * model_dir = "/home/bym/.cache/huggingface/hub/models--rednote-hilab--dots.tts-soar/snapshots/1fd9452e55c2c9f38fe1a8ee09eaf7448c222d35";

    GPT2BPETokenizer tok;
    if (!tok.load(model_dir)) {
        fprintf(stderr, "FAILED to load tokenizer from %s\n", model_dir);
        return 1;
    }
    printf("Loaded tokenizer: %d vocab, OK\n", tok.vocab_size());

    // Test cases that must match Python output
    struct { const char * text; int expected_ids[20]; int n_expected; } tests[] = {
        {"Hello world", {9707, 1879}, 2},
        {"Hello", {9707}, 1},
        {"world", {14615}, 1},
        {"this is a test", {574, 374, 264, 1273}, 4},
        {"Goodbye everyone", {15216, 28374, 5019}, 3},
        {"Hello world, this is dots TTS", {9707, 1879, 11, 419, 374, 30994, 350, 9951}, 8},
        {"", {}, 0},
    };

    int passed = 0, failed = 0;
    for (const auto & t : tests) {
        auto ids = tok.encode(t.text);

        printf("\"%s\" -> [", t.text);
        for (size_t i = 0; i < ids.size(); i++) {
            if (i > 0) printf(", ");
            printf("%d", ids[i]);
        }
        printf("]\n  expected: [");
        for (int i = 0; i < t.n_expected; i++) {
            if (i > 0) printf(", ");
            printf("%d", t.expected_ids[i]);
        }
        printf("] ");

        bool ok = ids.size() == (size_t)t.n_expected;
        if (ok) {
            for (int i = 0; i < t.n_expected; i++) {
                if ((int)ids[i] != t.expected_ids[i]) { ok = false; break; }
            }
        }
        printf("%s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else failed++;
    }

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
