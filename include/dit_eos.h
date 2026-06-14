    // EOS projection (2-layer MLP: Linear→SiLU→Linear)
    ggml_tensor * eos_proj_w1;  // [1536, 1536]
    ggml_tensor * eos_proj_b1;  // [1536]
    ggml_tensor * eos_proj_w2;  // [2, 1536]
    ggml_tensor * eos_proj_b2;  // [2]