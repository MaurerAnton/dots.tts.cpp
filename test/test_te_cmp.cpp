#include <torch/torch.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <vector>

struct DiTLT {
    std::unordered_map<std::string, torch::Tensor> w;
    bool load(const char * dir) {
        std::ifstream sf(std::string(dir)+"/shapes.txt");
        if(!sf) return false;
        std::string line;
        while(std::getline(sf,line)) {
            std::istringstream is(line); std::string n; is>>n;
            std::vector<int64_t> s; int64_t d; while(is>>d) s.push_back(d);
            std::string p=std::string(dir)+"/"+n+".bin";
            FILE*f=fopen(p.c_str(),"rb"); if(!f) return false;
            int64_t N=1; for(auto x:s) N*=x;
            auto*b=new float[N]; fread(b,sizeof(float),N,f); fclose(f);
            w[n]=torch::from_blob(b,s,torch::kFloat32).clone(); delete[] b;
        }
        return w.size()>100;
    }
    torch::Tensor get(const char*n){auto it=w.find(n);return it!=w.end()?it->second:torch::Tensor();}
};
static torch::Tensor lb(const char*p,std::vector<int64_t> s){
    int64_t n=1;for(auto d:s)n*=d;auto*b=new float[n];
    FILE*f=fopen(p,"rb");fread(b,sizeof(float),n,f);fclose(f);
    return torch::from_blob(b,s,torch::kFloat32).clone();delete[] b;
}

int main() {
    DiTLT w; w.load("models/dit_weights");
    
    // Time embedding
    float se_buf[256]; int half=128;
    for(int i=0;i<half;i++){float f=expf(-logf(10000.0f)*i/half);se_buf[i]=cosf(0*f);se_buf[half+i]=sinf(0*f);}
    auto se=torch::from_blob(se_buf,{1,256},torch::kFloat32).clone();
    
    auto tw1=w.get("time_embedder.mlp.0.weight");auto tb1=w.get("time_embedder.mlp.0.bias");
    auto l1=torch::matmul(se,tw1.t());if(tb1.defined())l1+=tb1;
    
    auto py_l1=lb("debug_v4/py_te_l1.bin",{1,1024});
    auto dl1=(l1-py_l1).abs().max().item<float>();
    printf("Lin1: CppRMS=%.4f PyRMS=%.4f diff=%.2e %s\n",l1.std().item<float>(),py_l1.std().item<float>(),dl1,dl1<1e-5?"OK":"FAIL");
    
    auto si=l1*torch::sigmoid(l1);  // SiLU
    auto py_si=lb("debug_v4/py_te_silu.bin",{1,1024});
    auto dsi=(si-py_si).abs().max().item<float>();
    printf("SiLU: CppRMS=%.4f PyRMS=%.4f diff=%.2e %s\n",si.std().item<float>(),py_si.std().item<float>(),dsi,dsi<1e-5?"OK":"FAIL");
    
    if(dsi<1e-5) printf("\nTIME EMBEDDING MATCHES! Bug is elsewhere.\n");
    else printf("\nTIME EMBEDDING FAILS at SiLU! This is the bug.\n");
    
    return 0;
}
