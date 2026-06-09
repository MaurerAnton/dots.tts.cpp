// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// gpt2_bpe_tokenizer.h — Pure C++ GPT-2 BPE tokenizer
// Reads vocab.json and merges.txt directly at runtime.
// Matches HuggingFace GPT-2/Mistral BPE tokenization exactly.
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

struct GPT2BPETokenizer {
    // Load from directory containing vocab.json and merges.txt
    bool load(const char * model_dir);

    // Tokenize text, returns token IDs
    std::vector<int32_t> encode(const std::string & text) const;

    // Number of tokens in vocabulary
    int vocab_size() const { return (int)vocab.size(); }

private:
    // bytes_to_unicode table (standard GPT-2 mapping)
    static const char * byte_to_unicode();

    // Vocabulary: string → token_id
    std::unordered_map<std::string, int32_t> vocab;

    // BPE merges: pair → rank (lower = higher priority)
    // Pair is concatenated string: first + " " + second
    std::unordered_map<std::string, int> bpe_ranks;

    // Cache for BPE operations
    std::string apply_bpe(const std::string & word) const;
    std::string basic_clean(const std::string & text) const;
};
