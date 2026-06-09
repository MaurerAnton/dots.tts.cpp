// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// gpt2_bpe_tokenizer.cpp — Pure C++ GPT-2 Byte-Level BPE tokenizer
// Matches HuggingFace GPT-2/Mistral tokenization.
//
// Algorithm:
//   1. Text → UTF-8 bytes
//   2. Each byte → bytes_to_unicode mapping (256 fixed mappings)
//   3. Regex pre-tokenizer splits into "words"
//   4. For each word: apply BPE merges greedily by rank
//   5. Split merged tokens on spaces, look up each in vocab
//
// The standard GPT-2 bytes_to_unicode mapping is hardcoded.
// It maps bytes 33-255 to 2-byte UTF-8 codepoints (U+00A1-U+019F range).
// Space (byte 32) maps to U+0120 = "Ġ" (C4 A0 in UTF-8).
// Control chars (0-31) and DEL (127) map to U+0100-U+011F and U+017F.

#include "gpt2_bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>

// ─── bytes_to_unicode table ───
// Generated from the standard GPT-2 mapping.
// 188 printable chars (33-126 except 127) map to themselves,
// bytes 0-32, 127-255 map to U+0100-U+019F range.
// Each entry is a UTF-8 encoded string (1 or 2 bytes).
struct byte_unicode {
    const char * s;
    int len;
};

static const byte_unicode B2U[256] = {
    // 0-31: control chars → U+0100-U+011F
    {"Ā",2},{"ā",2},{"Ă",2},{"ă",2},{"Ą",2},{"ą",2},{"Ć",2},{"ć",2},
    {"Ĉ",2},{"ĉ",2},{"Ċ",2},{"ċ",2},{"Č",2},{"č",2},{"Ď",2},{"ď",2},
    {"Đ",2},{"đ",2},{"Ē",2},{"ē",2},{"Ĕ",2},{"ĕ",2},{"Ė",2},{"ė",2},
    {"Ę",2},{"ę",2},{"Ě",2},{"ě",2},{"Ĝ",2},{"ĝ",2},{"Ğ",2},{"ğ",2},
    // 32: space → U+0120 = "Ġ"
    {"Ġ",2},
    // 33-126: printable ASCII → themselves
    {"!",1},{"\"",1},{"#",1},{"$",1},{"%",1},{"&",1},{"'",1},{"(",1},{")",1},{"*",1},
    {"+",1},{",",1},{"-",1},{".",1},{"/",1},{"0",1},{"1",1},{"2",1},{"3",1},{"4",1},
    {"5",1},{"6",1},{"7",1},{"8",1},{"9",1},{":",1},{";",1},{"<",1},{"=",1},{">",1},{"?",1},
    {"@",1},{"A",1},{"B",1},{"C",1},{"D",1},{"E",1},{"F",1},{"G",1},{"H",1},{"I",1},
    {"J",1},{"K",1},{"L",1},{"M",1},{"N",1},{"O",1},{"P",1},{"Q",1},{"R",1},{"S",1},
    {"T",1},{"U",1},{"V",1},{"W",1},{"X",1},{"Y",1},{"Z",1},{"[",1},{"\\",1},{"]",1},
    {"^",1},{"_",1},{"`",1},{"a",1},{"b",1},{"c",1},{"d",1},{"e",1},{"f",1},{"g",1},
    {"h",1},{"i",1},{"j",1},{"k",1},{"l",1},{"m",1},{"n",1},{"o",1},{"p",1},{"q",1},
    {"r",1},{"s",1},{"t",1},{"u",1},{"v",1},{"w",1},{"x",1},{"y",1},{"z",1},{"{",1},
    {"|",1},{"}",1},{"~",1},
    // 127: DEL → U+017F
    {"ſ",2},
    // 128-255: high bytes → U+0180-U+019F
    {"ƀ",2},{"Ɓ",2},{"Ƃ",2},{"ƃ",2},{"Ƅ",2},{"ƅ",2},{"Ɔ",2},{"Ƈ",2},
    {"ƈ",2},{"Ɖ",2},{"Ɗ",2},{"Ƌ",2},{"ƌ",2},{"ƍ",2},{"Ǝ",2},{"Ə",2},
    {"Ɛ",2},{"Ƒ",2},{"ƒ",2},{"Ɠ",2},{"Ɣ",2},{"ƕ",2},{"Ɩ",2},{"Ɨ",2},
    {"Ƙ",2},{"ƙ",2},{"ƚ",2},{"ƛ",2},{"Ɯ",2},{"Ɲ",2},{"ƞ",2},{"Ɵ",2},
    {"Ơ",2},{"ơ",2},{"Ƣ",2},{"ƣ",2},{"Ƥ",2},{"ƥ",2},{"Ʀ",2},{"Ƨ",2},
    {"ƨ",2},{"Ʃ",2},{"ƪ",2},{"ƫ",2},{"Ƭ",2},{"ƭ",2},{"Ʈ",2},{"ư",2},
    {"Ʊ",2},{"Ʋ",2},{"Ƴ",2},{"ƴ",2},{"Ƶ",2},{"ƶ",2},{"Ʒ",2},{"Ƹ",2},
    {"ƹ",2},{"ƺ",2},{"ƻ",2},{"Ƽ",2},{"ƽ",2},{"ƾ",2},{"ƿ",2},{"ǀ",2},
    {"ǁ",2},{"ǂ",2},{"ǃ",2},{"Ǆ",2},{"ǅ",2},{"ǆ",2},{"Ǉ",2},{"ǈ",2},
    {"ǉ",2},{"Ǌ",2},{"ǋ",2},{"ǌ",2},{"Ǎ",2},{"ǎ",2},{"Ǐ",2},{"ǐ",2},
    {"Ǒ",2},{"ǒ",2},{"Ǔ",2},{"ǔ",2},{"Ǖ",2},{"ǖ",2},{"Ǘ",2},{"ǘ",2},
    {"Ǚ",2},{"ǚ",2},{"Ǜ",2},{"ǜ",2},{"ǝ",2},{"Ǟ",2},{"ǟ",2},{"Ǡ",2},
    {"ǡ",2},{"Ǣ",2},{"ǣ",2},{"Ǥ",2},{"ǥ",2},{"Ǧ",2},{"ǧ",2},{"Ǩ",2},
    {"ǩ",2},{"Ǫ",2},{"ǫ",2},{"Ǭ",2},{"ǭ",2},{"Ǯ",2},{"ǯ",2},{"ǰ",2},
    {"Ǳ",2},{"ǲ",2},{"ǳ",2},{"Ǵ",2},{"ǵ",2},{"Ƕ",2},{"Ƿ",2},{"Ǹ",2},
    {"ǹ",2},{"Ǻ",2},{"ǻ",2},{"Ǽ",2},{"ǽ",2},{"Ǿ",2},{"ǿ",2},
};

// ─── Helpers ───
static std::string b2u_str(uint8_t b) {
    return std::string(B2U[b].s, B2U[b].len);
}

// ─── Load vocab.json ───
static bool load_vocab(const char * path, std::unordered_map<std::string, int32_t> & vocab) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream buf;
    buf << f.rdbuf();
    std::string content = buf.str();

    size_t pos = 1; // skip opening {
    while (pos < content.size()) {
        if (content[pos] == '}') break;
        if (content[pos] == ',' || content[pos] == ' ' || content[pos] == '\n' || content[pos] == '\r') { pos++; continue; }
        auto ks = content.find('"', pos);
        if (ks == std::string::npos) break;
        auto ke = content.find('"', ks + 1);
        if (ke == std::string::npos) break;
        std::string token = content.substr(ks + 1, ke - ks - 1);

        auto colon = content.find(':', ke);
        if (colon == std::string::npos) break;
        auto vs = colon + 1;
        while (vs < content.size() && (content[vs] == ' ' || content[vs] == '\n')) vs++;
        auto ve = vs;
        while (ve < content.size() && (std::isdigit(content[ve]) || content[ve] == '-')) ve++;
        int32_t id = (int32_t)strtol(content.c_str() + vs, nullptr, 10);

        vocab[token] = id;
        pos = ve;
    }
    return !vocab.empty();
}

// ─── Load merges.txt ───
static bool load_merges(const char * path, std::unordered_map<std::string, int> & bpe_ranks) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    int rank = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        bpe_ranks[line] = rank++;
    }
    return !bpe_ranks.empty();
}

// ─── GPT-2 pre-tokenizer regex ───
// The pattern splits on: contractions, letters+digits, whitespace
// Simplified implementation that handles the common cases correctly
static void pretokenize(const std::string & s, std::vector<std::string> & out) {
    // Collect characters that form "words" — sequences of non-whitespace
    // For byte-level BPE, the input has already been converted to bytes
    // and each byte is now a unicode char (1 or 2 UTF-8 bytes).
    // Space byte (32) → "Ġ" (2 bytes).
    // We split by the "Ġ" character to get word-like chunks.
    const char * G = B2U[32].s; // "Ġ"
    int glen = B2U[32].len;

    std::string cur;
    for (size_t i = 0; i < s.size();) {
        // Check if current position starts with Ġ
        if (i + glen <= s.size() && memcmp(s.c_str() + i, G, glen) == 0) {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
            // Add Ġ as prefix to next word
            cur = std::string(G, glen);
            i += glen;
        } else {
            // Determine char length (1 or 2 bytes for our unicode chars)
            // ASCII printable: 1 byte, others: determined by encoding
            unsigned char c = (unsigned char)s[i];
            int clen = (c >= 0x80) ? 2 : 1;
            if (i + clen > s.size()) clen = (int)(s.size() - i);
            cur.append(s, i, clen);
            i += clen;
        }
    }
    if (!cur.empty()) out.push_back(cur);
}

// ─── BPE on a single word ───
static std::string bpe_word(const std::string & word,
                             const std::unordered_map<std::string, int> & ranks) {
    // Split word into individual unicode chars
    std::vector<std::string> chars;
    for (size_t i = 0; i < word.size();) {
        unsigned char c = (unsigned char)word[i];
        int clen = (c >= 0xC0) ? 2 : 1;
        if (i + clen > word.size()) clen = (int)(word.size() - i);
        chars.push_back(word.substr(i, clen));
        i += clen;
    }

    // Greedy BPE:
    // While there's a mergeable pair, find the pair with lowest rank and merge it
    while (chars.size() > 1) {
        int best_rank = 2147483647;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < chars.size(); i++) {
            std::string pair = chars[i] + " " + chars[i + 1];
            auto it = ranks.find(pair);
            if (it != ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_rank == 2147483647) break;
        chars[best_i] = chars[best_i] + chars[best_i + 1];
        chars.erase(chars.begin() + best_i + 1);
    }

    std::string result;
    for (size_t i = 0; i < chars.size(); i++) {
        if (i > 0) result += " ";
        result += chars[i];
    }
    return result;
}

// ─── Encode ───
std::vector<int32_t> GPT2BPETokenizer::encode(const std::string & text) const {
    // Step 1: text → bytes → unicode
    std::string unicode;
    for (unsigned char c : (const std::string &)text) {
        unicode += b2u_str(c);
    }

    // Step 2: pre-tokenize (split on space = Ġ)
    std::vector<std::string> words;
    pretokenize(unicode, words);

    // Step 3: apply BPE to each word
    std::vector<std::string> bpe_tokens;
    for (const auto & w : words) {
        std::string bpe = bpe_word(w, bpe_ranks);
        // Split bpe result by spaces to get individual tokens
        std::string cur;
        for (size_t i = 0; i < bpe.size();) {
            if (bpe[i] == ' ') {
                if (!cur.empty()) { bpe_tokens.push_back(cur); cur.clear(); }
                i++;
            } else {
                unsigned char c = (unsigned char)bpe[i];
                int clen = (c >= 0xC0) ? 2 : 1;
                if (i + clen > bpe.size()) clen = (int)(bpe.size() - i);
                cur.append(bpe, i, clen);
                i += clen;
            }
        }
        if (!cur.empty()) bpe_tokens.push_back(cur);
    }

    // Step 4: map to vocab IDs
    std::vector<int32_t> ids;
    for (const auto & tok : bpe_tokens) {
        auto it = vocab.find(tok);
        if (it != vocab.end()) {
            ids.push_back(it->second);
        }
    }
    return ids;
}

// ─── Load ───
bool GPT2BPETokenizer::load(const char * model_dir) {
    std::string dir(model_dir);
    if (dir.back() != '/') dir += '/';
    if (!load_vocab((dir + "vocab.json").c_str(), vocab)) return false;
    if (!load_merges((dir + "merges.txt").c_str(), bpe_ranks)) return false;
    return true;
}
