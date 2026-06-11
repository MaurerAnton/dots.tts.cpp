// Test: verify template tokenization matches Python
#include "gpt2_bpe_tokenizer.h"
#include <cstdio>
#include <cstdlib>

int main() {
    GPT2BPETokenizer tok;
    const char * model_dir = getenv("DOTS_TTS_DIR");
    if (!model_dir) model_dir = "models";
    if (!tok.load(model_dir)) { fprintf(stderr, "FAILED to load\n"); return 1; }
    
    const char * tmpl = "[文本]hello world[文本对应语音]<|audio_gen_start|>";
    auto ids = tok.encode(tmpl);
    printf("Template: %zu tokens\n", ids.size());
    printf("C++ IDs: [");
    for (size_t i = 0; i < ids.size(); i++) {
        if (i > 0) printf(", ");
        printf("%d", ids[i]);
    }
    printf("]\n");
    
    // Python: [58, 108704, 60, 14990, 1879, 58, 108704, 103124, 105761, 60, 151668]
    printf("Py  IDs: [58, 108704, 60, 14990, 1879, 58, 108704, 103124, 105761, 60, 151668]\n");
    
    bool match = ids.size() == 11;
    int expected[] = {58, 108704, 60, 14990, 1879, 58, 108704, 103124, 105761, 60, 151668};
    for (size_t i = 0; i < 11 && i < ids.size(); i++) {
        if (ids[i] != expected[i]) { match = false; break; }
    }
    printf("Match: %s\n", match ? "YES" : "NO");
    return match ? 0 : 1;
}
