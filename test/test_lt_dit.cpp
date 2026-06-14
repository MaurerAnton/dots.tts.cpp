#include <torch/torch.h>
#include "../src/dit_libtorch.h"
#include <cstdio>

int main() {
    DiTWeightsLT w;
    if (!w.load("models/dit_weights")) return 1;
    
    // Load inputs
    auto load_bin = [](const char * p, std::vector<int64_t> s) {
        int64_t n = 1; for (auto d : s) n *= d;
        auto * d = new float[n];
        FILE * f = fopen(p, "rb"); fread(d, sizeof(float), n, f); fclose(f);
        auto t = torch::from_blob(d, s, torch::kFloat32).clone();
        delete[] d; return t;
    };
    
    auto x = load_bin("debug_v4/dit_input_ref.bin", {5, 1024});
    auto cond = load_bin("debug_v4/py_cond_t0.bin", {1, 1024});
    auto py_vel = load_bin("debug_v4/py_vel_test_v2.bin", {4, 128});
    
    // Mask and pos_ids
    auto mask = torch::ones({1, 5, 5}, torch::kBool);
    auto pos_ids = torch::tensor({{0.f, 1.f, 2.f, 3.f, 4.f}});
    
    auto cpp_vel = dit_forward_lt(x, cond, w, 18, 4, mask, pos_ids);
    
    printf("LibTorch RMS: %.6f  Py RMS: %.6f\n", cpp_vel.std().item<float>(), py_vel.std().item<float>());
    float max_d = (cpp_vel - py_vel).abs().max().item<float>();
    printf("Max|diff|: %.10f  IDENTICAL: %s\n", max_d, max_d < 1e-5 ? "YES" : "NO");
    return 0;
}
