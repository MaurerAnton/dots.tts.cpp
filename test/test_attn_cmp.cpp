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

static torch::Tensor load_bin(const char*p,std::vector<int64_t> s){
    int64_t n=1;for(auto d:s)n*=d;auto*b=new float[n];
    FILE*f=fopen(p,"rb");fread(b,sizeof(float),n,f);fclose(f);
    return torch::from_blob(b,s,torch::kFloat32).clone();delete[] b;
}

int main() {
    DiTLT w; w.load("models/dit_weights");
    int hidden=1024,n_heads=16,head_dim=64,N=5,half=32;
    
    // Build same inputs as Python
    float se_buf[256];
    for(int i=0;i<128;i++){float f=expf(-logf(10000.0f)*i/128);se_buf[i]=cosf(0*f);se_buf[128+i]=sinf(0*f);}
    auto se=torch::from_blob(se_buf,{1,256},torch::kFloat32).clone();
    auto tw1=w.get("time_embedder.mlp.0.weight");auto tb1=w.get("time_embedder.mlp.0.bias");
    auto te=torch::matmul(se,tw1.t());if(tb1.defined())te+=tb1;te=torch::sigmoid(te)*te;
    auto tw2=w.get("time_embedder.mlp.2.weight");auto tb2=w.get("time_embedder.mlp.2.bias");
    auto cond=torch::matmul(te,tw2.t());if(tb2.defined())cond+=tb2;
    
    auto x=load_bin("debug_v4/dit_input_ref.bin",{5,1024});
    auto iw=w.get("input_layer.weight");auto ib=w.get("input_layer.bias");
    auto il=torch::matmul(x,iw.t());if(ib.defined())il+=ib;
    
    auto aw=w.get("blocks.0.adaLN_modulation.1.weight");auto ab=w.get("blocks.0.adaLN_modulation.1.bias");
    auto si=torch::sigmoid(cond)*cond;
    auto adaln=torch::matmul(si,aw.t());if(ab.defined())adaln+=ab;
    auto sm=adaln.slice(1,0,hidden);auto scm=adaln.slice(1,hidden,2*hidden);
    auto mod=torch::layer_norm(il,{hidden},{},{},1e-5)*(1.0f+scm)+sm;
    
    // QKV
    auto q=torch::matmul(mod,w.get("blocks.0.attn.q_proj.weight").t()).view({N,n_heads,head_dim}).transpose(0,1);
    auto k=torch::matmul(mod,w.get("blocks.0.attn.k_proj.weight").t()).view({N,n_heads,head_dim}).transpose(0,1);
    auto v=torch::matmul(mod,w.get("blocks.0.attn.v_proj.weight").t()).view({N,n_heads,head_dim}).transpose(0,1);
    
    auto qn=w.get("blocks.0.attn.q_norm.weight");auto kn=w.get("blocks.0.attn.k_norm.weight");
    if(qn.defined()){auto r=torch::sqrt(torch::mean(q*q,-1,true)+1e-6f);q=q*qn/r;}
    if(kn.defined()){auto r=torch::sqrt(torch::mean(k*k,-1,true)+1e-6f);k=k*kn/r;}
    
    // RoPE
    auto pos=torch::tensor({{0.f,1.f,2.f,3.f,4.f}});
    auto inv_freq=torch::pow(10000.0f,-2.0f*torch::arange(0,half,torch::kFloat32)/head_dim);
    auto ang=pos.unsqueeze(-1)*inv_freq.unsqueeze(0);
    auto cos_a=torch::cos(ang).unsqueeze(0);auto sin_a=torch::sin(ang).unsqueeze(0);
    auto q1=q.slice(-1,0,half),q2=q.slice(-1,half);
    auto k1=k.slice(-1,0,half),k2=k.slice(-1,half);
    auto q_rot=torch::cat({q1*cos_a-q2*sin_a,q1*sin_a+q2*cos_a},-1);
    auto k_rot=torch::cat({k1*cos_a-k2*sin_a,k1*sin_a+k2*cos_a},-1);
    
    // Compare each step with Python
    auto py_qr=load_bin("debug_v4/py_qrot.bin",{n_heads,N,head_dim});
    auto dq=(q_rot-py_qr).abs().max().item<float>();
    printf("Q_rot: CppRMS=%.4f PyRMS=%.4f diff=%.2e %s\n",q_rot.std().item<float>(),py_qr.std().item<float>(),dq,dq<1e-4?"OK":"FAIL");
    
    if(dq<1e-4) {
        // Scores  
        float sc=1.0f/sqrtf(head_dim);
        auto scores=torch::matmul(q_rot,k_rot.transpose(-2,-1))*sc;
        auto py_sc=load_bin("debug_v4/py_scores.bin",{n_heads,N,N});
        auto ds=(scores-py_sc).abs().max().item<float>();
        printf("Scores: CppRMS=%.4f PyRMS=%.4f diff=%.2e %s\n",scores.std().item<float>(),py_sc.std().item<float>(),ds,ds<1e-4?"OK":"FAIL");
        
        // Attn weights
        auto attn_w=torch::softmax(scores,-1);
        auto py_aw=load_bin("debug_v4/py_attnw.bin",{n_heads,N,N});
        auto da=(attn_w-py_aw).abs().max().item<float>();
        printf("AttnW:  CppRMS=%.4f PyRMS=%.4f diff=%.2e %s\n",attn_w.std().item<float>(),py_aw.std().item<float>(),da,da<1e-4?"OK":"FAIL");
        
        // Attn output
        auto ao_h=torch::matmul(attn_w,v).transpose(0,1).contiguous().view({N,hidden});
        auto ao=torch::matmul(ao_h,w.get("blocks.0.attn.o_proj.weight").t());
        if(w.get("blocks.0.attn.o_proj.bias").defined())ao+=w.get("blocks.0.attn.o_proj.bias");
        auto py_ao=load_bin("debug_v4/py_ao.bin",{N,hidden});
        auto dao=(ao-py_ao).abs().max().item<float>();
        printf("AO:     CppRMS=%.4f PyRMS=%.4f diff=%.2e %s\n",ao.std().item<float>(),py_ao.std().item<float>(),dao,dao<1e-4?"OK":"FAIL");
    }
    
    return 0;
}

    // FFN comparison
    { {
        auto sml=adaln.slice(1,3*hidden,4*hidden);auto scl=adaln.slice(1,4*hidden,5*hidden);auto gml=adaln.slice(1,5*hidden,6*hidden);
        auto gm=adaln.slice(1,2*hidden,3*hidden);
        
        // Residual + gate
        auto h_res=il+gm*ao;
        
        // FFN
        auto norm2=torch::layer_norm(h_res,{hidden},{},{},1e-5);
        auto mod2=norm2*(1.0f+scl)+sml;
        auto fw1=w.get("blocks.0.ffn.fc1.weight");auto fb1=w.get("blocks.0.ffn.fc1.bias");
        auto ff=torch::matmul(mod2,fw1.t());if(fb1.defined())ff+=fb1;
        ff=0.5f*ff*(1.0f+torch::tanh(0.79788456f*(ff+0.044715f*ff*ff*ff)));
        auto fw2=w.get("blocks.0.ffn.fc2.weight");auto fb2=w.get("blocks.0.ffn.fc2.bias");
        ff=torch::matmul(ff,fw2.t());if(fb2.defined())ff+=fb2;
        
        auto py_ffn=load_bin("debug_v4/py_ffn.bin",{N,hidden});
        auto df=(ff-py_ffn).abs().max().item<float>();
        printf("FFN:    CppRMS=%.4f PyRMS=%.4f diff=%.2e %s\n",ff.std().item<float>(),py_ffn.std().item<float>(),df,df<1e-4?"OK":"FAIL");
        
        auto final_blk=h_res+gml*ff;
        auto py_final=load_bin("debug_v4/py_blk0_full.bin",{N,hidden});
        auto dblk=(final_blk-py_final).abs().max().item<float>();
        printf("Block0: CppRMS=%.4f PyRMS=%.4f diff=%.2e %s\n",final_blk.std().item<float>(),py_final.std().item<float>(),dblk,dblk<1e-4?"OK":"FAIL");
    }
