#include <torch/torch.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static torch::Tensor load_bin(const char * path, std::vector<int64_t> shape) {
    int64_t n = 1; for (auto s : shape) n *= s;
    float * data = new float[n];
    FILE * f = fopen(path, "rb"); 
    if (!f) { printf("MISSING: %s\n", path); exit(1); }
    size_t got = fread(data, sizeof(float), n, f); fclose(f);
    printf("  Loaded %s: %zu floats\n", path, got);
    auto t = torch::from_blob(data, shape, torch::kFloat32).clone();
    delete[] data;
    return t;
}

int main() {
    auto x = load_bin("debug_v4/dit_input_ref.bin", {5, 1024});
    auto py_ref = load_bin("debug_v4/py_input_layer_ref.bin", {5, 1024});
    auto w = load_bin("models/input_layer_weight.bin", {1024, 1024});
    auto b = load_bin("models/input_layer_bias.bin", {1024});
    
    printf("w[0,0]=%.6f w[0,1]=%.6f\n", w[0][0].item<float>(), w[0][1].item<float>());
    printf("x[0,0]=%.6f  b[0]=%.6f\n", x[0][0].item<float>(), b[0].item<float>());
    
    // Forward: position 0 only
    auto h0 = torch::matmul(x[0], w.t()) + b;
    printf("LibTorch h0[0:3] = %.6f %.6f %.6f\n", h0[0].item<float>(), h0[1].item<float>(), h0[2].item<float>());
    printf("Python   py[0:3] = %.6f %.6f %.6f\n", py_ref[0][0].item<float>(), py_ref[0][1].item<float>(), py_ref[0][2].item<float>());
    
    // Compare all 5 positions
    auto h = torch::matmul(x, w.t()) + b;
    auto diff = (h - py_ref).abs();
    float max_d = diff.max().item<float>();
    printf("\nFull: LibRMS=%.4f PyRMS=%.4f Max|diff|=%.10f\n", h.std().item<float>(), py_ref.std().item<float>(), max_d);
    printf("IDENTICAL: %s\n", max_d == 0.0f ? "YES" : "NO");
    
    return 0;
}
