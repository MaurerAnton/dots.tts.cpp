#include <torch/torch.h>
#include "../src/dit_libtorch.h"
#include <cstdio>

int main() {
    DiTWeightsLT w; w.load("models/dit_weights");
    
    auto load_bin = [](const char * p, std::vector<int64_t> s) {
        int64_t n = 1; for (auto d : s) n *= d;
        auto * d = new float[n];
        FILE * f = fopen(p, "rb"); fread(d, sizeof(float), n, f); fclose(f);
        auto t = torch::from_blob(d, s, torch::kFloat32).clone();
        delete[] d; return t;
    };
    
    auto x = load_bin("debug_v4/dit_input_ref.bin", {5, 1024});
    auto cond = load_bin("debug_v4/py_cond_t0.bin", {1, 1024});
    auto py_il = load_bin("debug_v4/py_il_out.bin", {5, 1024});
    auto py_b0 = load_bin("debug_v4/py_b0_out.bin", {5, 1024});
    
    // Input layer
    auto iw = w.get("input_layer.weight");
    auto ib = w.get("input_layer.bias");
    auto il = torch::matmul(x, iw.t());
    if (ib.defined()) il += ib;
    
    float d_il = (il - py_il).abs().max().item<float>();
    printf("Input layer: CppRMS=%.4f PyRMS=%.4f Max|diff|=%.2e %s\n",
        il.std().item<float>(), py_il.std().item<float>(), d_il, d_il < 1e-5 ? "OK" : "FAIL");
    
    // Block 0
    auto mask = torch::ones({1, 5, 5}, torch::kBool);
    auto pos_ids = torch::tensor({{0.f, 1.f, 2.f, 3.f, 4.f}});
    auto b0 = dit_block_lt(il, cond, w, 0, mask, pos_ids);
    float d_b0 = (b0 - py_b0).abs().max().item<float>();
    printf("Block 0:     CppRMS=%.4f PyRMS=%.4f Max|diff|=%.2e %s\n",
        b0.std().item<float>(), py_b0.std().item<float>(), d_b0, d_b0 < 1e-5 ? "OK" : "FAIL");
    
    return 0;
}
