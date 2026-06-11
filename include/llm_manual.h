// Manual LLM forward pass declarations (implementation in src/llm_manual.cpp)
#pragma once
#include "safetensors.h"

struct LLMWeights {
    float *embed_w;
    struct Layer {
        float *an, *fn;
        float *qw,*qb, *kw,*kb, *vw,*vb, *ow,*ob;
        float *gw, *uw, *dw;
    };
    Layer * layers;
    float * final_norm;
    int n_layers;
};

bool load_llm_weights(SafeTensorsFile & sf, LLMWeights & w, int n_layers);
void free_llm_weights(LLMWeights & w);
void llm_manual_forward(const LLMWeights & w, const int * token_ids, int n_tok, float * hiddens_out);
