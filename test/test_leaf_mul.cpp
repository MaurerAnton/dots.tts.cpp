// Test: ggml graph with leaf tensor as 2nd arg to ggml_mul
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <cstring>

int main() {
    ggml_init_params gp = { .mem_size = 64*1024*1024 };
    ggml_context * ctx = ggml_init(gp);
    
    // Create graph node (rms_norm of random input)
    float inp[16]; for(int i=0;i<16;i++) inp[i]=(float)(i-8)/4.0f;
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    memcpy(x->data, inp, 16*sizeof(float));
    ggml_tensor * normed = ggml_rms_norm(ctx, x, 1e-5f);
    
    // Create leaf tensor (scale from manual computation)
    float scale_vals[8] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    ggml_tensor * scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    for(int i=0;i<8;i++) for(int b=0;b<2;b++) ((float*)scale->data)[b*8+i] = scale_vals[i];
    
    // Test: ggml_mul(graph_node, leaf_tensor)
    ggml_tensor * mod = ggml_mul(ctx, normed, scale);
    
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, mod);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    
    float * md = (float*)mod->data;
    printf("mod first4: %.4f %.4f %.4f %.4f\n", md[0], md[1], md[2], md[3]);
    // Expected: normed[i] * 0.5
    float * nd = (float*)normed->data;
    printf("normed first4: %.4f %.4f %.4f %.4f\n", nd[0], nd[1], nd[2], nd[3]);
    
    ggml_free(ctx);
    return 0;
}
