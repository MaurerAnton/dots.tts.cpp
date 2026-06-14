#include <torch/torch.h>
#include <torch/script.h>
#include <cstdio>

int main() {
    auto dit = torch::jit::load("models/dit_traced.pt");
    
    // Load exact same input as Python
    auto load_bin = [](const char * p, std::vector<int64_t> s) {
        int64_t n = 1; for (auto d : s) n *= d;
        auto * d = new float[n];
        FILE * f = fopen(p, "rb"); fread(d, sizeof(float), n, f); fclose(f);
        auto t = torch::from_blob(d, s, torch::kFloat32).clone();
        delete[] d; return t;
    };
    
    auto x = load_bin("debug_v4/dit_input_ref.bin", {1, 5, 1024});
    auto t = torch::tensor({0.0f});
    auto mask = torch::ones({1, 5, 5}, torch::kBool);
    auto pos_ids = torch::zeros({1, 5}, torch::kFloat32);
    
    // Print input details
    printf("x: %s\n", x.toString().c_str());
    printf("t: %s\n", t.toString().c_str());
    printf("mask: %s\n", mask.toString().c_str());
    printf("pos: %s\n", pos_ids.toString().c_str());
    printf("x[0,0,0:3] = %.6f %.6f %.6f\n", x[0][0][0].item<float>(), x[0][0][1].item<float>(), x[0][0][2].item<float>());
    
    // Try forward
    auto out = dit.forward({x, t, mask, pos_ids}).toTensor();
    printf("out shape: [%ld, %ld, %ld]\n", out.size(0), out.size(1), out.size(2));
    printf("out[0,0:3,0] = %.6f %.6f %.6f\n", out[0][0][0].item<float>(), out[0][1][0].item<float>(), out[0][2][0].item<float>());
    
    return 0;
}
