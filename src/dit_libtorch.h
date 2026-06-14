// SPDX-License-Identifier: GPL-3.0-or-later  
// LibTorch DiT — byte-identical with Python MKL
#pragma once
#include <torch/torch.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>

struct DiTWeightsLT {
    std::unordered_map<std::string, torch::Tensor> tensors;
    bool load(const char * dir) {
        std::ifstream sf(std::string(dir) + "/shapes.txt");
        if (!sf) return false;
        std::string line;
        while (std::getline(sf, line)) {
            std::istringstream iss(line);
            std::string name; iss >> name;
            std::vector<int64_t> shape;
            int64_t s; while (iss >> s) shape.push_back(s);
            std::string path = std::string(dir) + "/" + name + ".bin";
            FILE * f = fopen(path.c_str(), "rb");
            if (!f) return false;
            int64_t n = 1; for (auto d : shape) n *= d;
            float * data = new float[n];
            fread(data, sizeof(float), n, f); fclose(f);
            tensors[name] = torch::from_blob(data, shape, torch::kFloat32).clone();
            delete[] data;
        }
        return true;
    }
    torch::Tensor get(const char * n) { auto it = tensors.find(n); return it != tensors.end() ? it->second : torch::Tensor(); }
};

// RoPE (same as Python: apply rotary position embedding)
static std::pair<torch::Tensor, torch::Tensor> rope_lt(torch::Tensor q, torch::Tensor k, torch::Tensor pos_ids, float theta = 10000.0f) {
    int head_dim = q.size(-1), half = head_dim / 2, N = q.size(1);
    auto pos = pos_ids.unsqueeze(-1);  // [1, N, 1]
    auto freqs = torch::arange(0, half, torch::kFloat32).unsqueeze(0);  // [1, half]
    freqs = torch::pow(theta, -2.0f * freqs / head_dim);  // [1, half]
    auto angles = pos * freqs;  // [1, N, half]
    auto cos_a = torch::cos(angles).unsqueeze(0);  // [1, 1, N, half]
    auto sin_a = torch::sin(angles).unsqueeze(0);  // [1, 1, N, half]
    
    // q, k: [n_heads, N, head_dim]
    auto q1 = q.slice(-1, 0, half), q2 = q.slice(-1, half);
    auto k1 = k.slice(-1, 0, half), k2 = k.slice(-1, half);
    
    auto q_rot = torch::cat({q1 * cos_a - q2 * sin_a, q1 * sin_a + q2 * cos_a}, -1);
    auto k_rot = torch::cat({k1 * cos_a - k2 * sin_a, k1 * sin_a + k2 * cos_a}, -1);
    
    return {q_rot, k_rot};
}

// Single DiT block with attention mask and position IDs
static torch::Tensor dit_block_lt(torch::Tensor x, torch::Tensor cond, DiTWeightsLT & w, 
    int block_idx, torch::Tensor attn_mask, torch::Tensor pos_ids) {
    int hidden = 1024;
    char buf[256];
    int N = x.size(0);
    
    // adaLN modulation (cond is [1, hidden], si is [1, hidden])
    snprintf(buf, sizeof(buf), "blocks.%d.adaLN_modulation.1.weight", block_idx);
    auto adaln_w = w.get(buf);
    snprintf(buf, sizeof(buf), "blocks.%d.adaLN_modulation.1.bias", block_idx);
    auto adaln_b = w.get(buf);
    auto s = torch::sigmoid(cond); auto si = cond * s;  // [1, hidden]
    auto adaln = torch::matmul(si, adaln_w.t());  // [1, 6*hidden]
    if (adaln_b.defined()) adaln += adaln_b;
    
    auto sm = adaln.slice(1, 0, hidden);       // [1, hidden]
    auto scm = adaln.slice(1, hidden, 2*hidden);
    auto gm = adaln.slice(1, 2*hidden, 3*hidden);
    auto sml = adaln.slice(1, 3*hidden, 4*hidden);
    auto scl = adaln.slice(1, 4*hidden, 5*hidden);
    auto gml = adaln.slice(1, 5*hidden, 6*hidden);
    
    // LayerNorm + modulation: normed [N, hidden], mod [N, hidden]
    auto normed = torch::layer_norm(x, {hidden}, {}, {}, 1e-5);
    auto mod = normed * (1.0f + scm) + sm;
    
    // Self-attention QKV projections
    snprintf(buf, sizeof(buf), "blocks.%d.attn.q_proj.weight", block_idx);
    auto qw = w.get(buf);
    snprintf(buf, sizeof(buf), "blocks.%d.attn.k_proj.weight", block_idx);
    auto kw = w.get(buf);
    snprintf(buf, sizeof(buf), "blocks.%d.attn.v_proj.weight", block_idx);
    auto vw = w.get(buf);
    snprintf(buf, sizeof(buf), "blocks.%d.attn.o_proj.weight", block_idx);
    auto ow = w.get(buf);
    snprintf(buf, sizeof(buf), "blocks.%d.attn.o_proj.bias", block_idx);
    auto ob = w.get(buf);
    
    int n_heads = 16, head_dim = 64;
    auto q = torch::matmul(mod, qw.t()).view({N, n_heads, head_dim}).transpose(0, 1);  // [n_heads, N, head_dim]
    auto k = torch::matmul(mod, kw.t()).view({N, n_heads, head_dim}).transpose(0, 1);
    auto v = torch::matmul(mod, vw.t()).view({N, n_heads, head_dim}).transpose(0, 1);
    
    // QK RMSNorm
    snprintf(buf, sizeof(buf), "blocks.%d.attn.q_norm.weight", block_idx);
    auto qnw = w.get(buf);
    snprintf(buf, sizeof(buf), "blocks.%d.attn.k_norm.weight", block_idx);
    auto knw = w.get(buf);
    if (qnw.defined()) q = torch::layer_norm(q, {head_dim}, qnw, {}, 1e-6);
    if (knw.defined()) k = torch::layer_norm(k, {head_dim}, knw, {}, 1e-6);
    
    // RoPE
    auto r = rope_lt(q, k, pos_ids);
    q = r.first; k = r.second;
    
    // Attention scores with mask
    float scale = 1.0f / sqrtf((float)head_dim);
    auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale;  // [n_heads, N, N]
    auto mask_f = attn_mask.to(torch::kFloat32);  // [1, N, N]
    mask_f = torch::where(mask_f > 0.5f, 0.0f, -1e9f);  // convert bool mask to additive
    scores = scores + mask_f;
    auto attn_weights = torch::softmax(scores, -1);
    auto attn_out = torch::matmul(attn_weights, v);  // [n_heads, N, head_dim]
    attn_out = attn_out.transpose(0, 1).contiguous().view({N, hidden});  // [N, hidden]
    
    // O projection
    auto ao = torch::matmul(attn_out, ow.t());
    if (ob.defined()) ao += ob;
    
    // Residual gate
    auto h = x + gm * ao;
    
    // FFN
    normed = torch::layer_norm(h, {hidden}, {}, {}, 1e-5);
    mod = normed * (1.0f + scl) + sml;
    
    snprintf(buf, sizeof(buf), "blocks.%d.ffn.fc1.weight", block_idx);
    auto fw1 = w.get(buf);
    snprintf(buf, sizeof(buf), "blocks.%d.ffn.fc1.bias", block_idx);
    auto fb1 = w.get(buf);
    snprintf(buf, sizeof(buf), "blocks.%d.ffn.fc2.weight", block_idx);
    auto fw2 = w.get(buf);
    snprintf(buf, sizeof(buf), "blocks.%d.ffn.fc2.bias", block_idx);
    auto fb2 = w.get(buf);
    
    auto ffn = torch::matmul(mod, fw1.t());
    if (fb1.defined()) ffn += fb1;
    ffn = 0.5f * ffn * (1.0f + torch::tanh(0.7978845608028654f * (ffn + 0.044715f * ffn * ffn * ffn)));
    ffn = torch::matmul(ffn, fw2.t());
    if (fb2.defined()) ffn += fb2;
    
    return h + gml * ffn;
}

// Full DiT forward
static torch::Tensor dit_forward_lt(torch::Tensor x, torch::Tensor cond, DiTWeightsLT & w, 
    int n_layers, int patch_size, torch::Tensor attn_mask, torch::Tensor pos_ids) {
    int hidden = 1024, latent_dim = 128;
    int N = x.size(0);
    
    // Input layer
    auto iw = w.get("input_layer.weight");
    auto ib = w.get("input_layer.bias");
    auto h = torch::matmul(x, iw.t());
    if (ib.defined()) h += ib;
    
    // DiT blocks
    for (int i = 0; i < n_layers; i++) {
        h = dit_block_lt(h, cond, w, i, attn_mask, pos_ids);
    }
    
    // Output adaLN
    auto ow = w.get("output_layer.adaLN_modulation.1.weight");
    auto ob = w.get("output_layer.adaLN_modulation.1.bias");
    auto s = torch::sigmoid(cond); auto si = cond * s;
    auto mod_raw = torch::matmul(si, ow.t());
    if (ob.defined()) mod_raw += ob;
    auto shift = mod_raw.slice(1, 0, hidden);
    auto scale = mod_raw.slice(1, hidden, 2*hidden);
    
    auto normed = torch::layer_norm(h, {hidden}, {}, {}, 1e-5);
    auto modulated = normed * (1.0f + scale) + shift;
    
    // Output projection
    auto out_w = w.get("output_layer.linear.weight");
    auto out_b = w.get("output_layer.linear.bias");
    auto out = torch::matmul(modulated, out_w.t());
    if (out_b.defined()) out += out_b;
    
    int latent_start = N - patch_size;
    return out.slice(0, latent_start, N);
}
