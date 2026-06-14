// Quick audio generation using traced LibTorch DiT
#include <torch/torch.h>
#include <torch/script.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// Load raw binary tensor
static torch::Tensor load_bin(const char * path, std::vector<int64_t> shape) {
    int64_t n = 1; for (auto s : shape) n *= s;
    float * data = new float[n];
    FILE * f = fopen(path, "rb"); fread(data, sizeof(float), n, f); fclose(f);
    auto t = torch::from_blob(data, shape, torch::kFloat32).clone();
    delete[] data; return t;
}

int main(int argc, char ** argv) {
    setbuf(stdout, NULL);
    const char * text = argc > 1 ? argv[1] : "Hello world";
    printf("LibTorch TTS: '%s'\n", text);
    
    // Load traced DiT
    auto dit = torch::jit::load("models/dit_traced.pt");
    printf("Loaded traced DiT\n");
    
    // Load weights for other components (hidden_proj, coord_proj, latent_proj)
    // For now, use a simple FM buffer with pre-computed hidden_proj
    // Actually, let me just generate one call of DitT output to verify the pipeline works
    
    // Build a simple dit_input from scratch using manual coord_proj
    // This is a minimal test — full pipeline needs LM + PE + BigVGAN integration
    
    // Load coord_proj weights from models/dit_weights/
    auto coord_w = load_bin("models/dit_weights/coordinate_proj.weight", {1024, 128});
    auto coord_b = load_bin("models/dit_weights/coordinate_proj.bias", {1024});
    
    // Generate random noise
    torch::manual_seed(42);
    auto noise = torch::randn({1, 4, 128});
    auto noise_proj = torch::matmul(noise[0], coord_w.t()) + coord_b;  // [4, 1024]
    
    // Use zeros for hidden_proj (no text conditioning for this quick test)
    auto hidden_proj = torch::zeros({1, 1024});
    
    // Build dit_input [1, 5, 1024]
    auto dit_input = torch::cat({hidden_proj, noise_proj}, 0).unsqueeze(0);  // [1, 5, 1024]
    
    auto t = torch::tensor({0.0f});
    auto mask = torch::ones({1, 5, 5}, torch::kBool);
    auto pos_ids = torch::zeros({1, 5}, torch::kFloat32);
    
    // Euler solver: 10 steps
    auto z = noise.clone();  // [1, 4, 128]
    float dt = 0.1f;
    for (int step = 0; step < 10; step++) {
        // Build DiT input with current noise
        auto noise_p = torch::matmul(z[0], coord_w.t()) + coord_b;
        dit_input[0].slice(0, 1, 5) = noise_p;
        
        // Run DiT
        auto vel = dit.forward({dit_input, t, mask, pos_ids}).toTensor();  // [1, 5, 128]
        auto v = vel.slice(1, 1, 5);  // [1, 4, 128]
        
        // Euler step
        z = z + v * dt;
        t += dt;
    }
    
    printf("Final latent RMS: %.4f\n", z.std().item<float>());
    printf("LibTorch DiT pipeline works!\n");
    
    return 0;
}
