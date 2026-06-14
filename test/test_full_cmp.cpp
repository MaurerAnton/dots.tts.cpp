#include <torch/torch.h>
#include <cstdio>
#include <cmath>
#include "../src/dit_libtorch.h" // for DiTWeightsLT

int main() {
    DiTWeightsLT w; w.load("models/dit_weights");
    int hidden=1024,n_heads=16,head_dim=64,N=5,half=32;
    
    // Time embed
    float se_buf[256];
    for(int i=0;i<128;i++){float f=expf(-logf(10000.0f)*i/128);se_buf[i]=cosf(0*f);se_buf[128+i]=sinf(0*f);}
    auto se=torch::from_blob(se_buf,{1,256},torch::kFloat32).clone();
    auto tw1=w.get("time_embedder.mlp.0.weight");auto tb1=w.get("time_embedder.mlp.0.bias");
    auto te=torch::matmul(se,tw1.t());if(tb1.defined())te+=tb1;te=torch::sigmoid(te)*te;
    auto tw2=w.get("time_embedder.mlp.2.weight");auto tb2=w.get("time_embedder.mlp.2.bias");
    auto cond=torch::matmul(te,tw2.t());if(tb2.defined())cond+=tb2;
    
    // Input
    auto lb=[](const char*p,std::vector<int64_t> s){
        int64_t n=1;for(auto d:s)n*=d;auto*b=new float[n];
        FILE*f=fopen(p,"rb");fread(b,sizeof(float),n,f);fclose(f);
        return torch::from_blob(b,s,torch::kFloat32).clone();delete[] b;
    };
    auto x=lb("debug_v4/dit_input_ref.bin",{5,1024});
    auto iw=w.get("input_layer.weight");auto ib=w.get("input_layer.bias");
    auto h=torch::matmul(x,iw.t());if(ib.defined())h+=ib;
    
    // Run all 18 blocks
    auto mask=torch::ones({1,5,5},torch::kBool);
    auto pos=torch::tensor({{0.f,1.f,2.f,3.f,4.f}});
    auto pos_e=pos.unsqueeze(-1);
    auto inv_freq=torch::pow(10000.0f,-2.0f*torch::arange(0,half,torch::kFloat32)/head_dim);
    auto ang=pos_e*inv_freq.unsqueeze(0);
    auto cos_a=torch::cos(ang).unsqueeze(0);auto sin_a=torch::sin(ang).unsqueeze(0);
    
    for(int blk=0;blk<18;blk++){ if(blk==0){auto in_rms=h.std().item<float>();}{
        char b[256];
        snprintf(b,sizeof(b),"blocks.%d.adaLN_modulation.1.weight",blk);
        auto aw=w.get(b); snprintf(b,sizeof(b),"blocks.%d.adaLN_modulation.1.bias",blk); auto ab=w.get(b);
        auto si=torch::sigmoid(cond)*cond;
        auto adaln=torch::matmul(si,aw.t());if(ab.defined())adaln+=ab;
        auto sm=adaln.slice(1,0,hidden);auto scm=adaln.slice(1,hidden,2*hidden);auto gm=adaln.slice(1,2*hidden,3*hidden);
        auto sml=adaln.slice(1,3*hidden,4*hidden);auto scl=adaln.slice(1,4*hidden,5*hidden);auto gml=adaln.slice(1,5*hidden,6*hidden);
        
        auto mod=torch::layer_norm(h,{hidden},{},{},1e-5)*(1.0f+scm)+sm;
        snprintf(b,sizeof(b),"blocks.%d.attn.q_proj.weight",blk); auto qw=w.get(b);
        snprintf(b,sizeof(b),"blocks.%d.attn.k_proj.weight",blk); auto kw=w.get(b);
        snprintf(b,sizeof(b),"blocks.%d.attn.v_proj.weight",blk); auto vw=w.get(b);
        auto q=torch::matmul(mod,qw.t()).view({N,n_heads,head_dim}).transpose(0,1);
        auto k=torch::matmul(mod,kw.t()).view({N,n_heads,head_dim}).transpose(0,1);
        auto v=torch::matmul(mod,vw.t()).view({N,n_heads,head_dim}).transpose(0,1);
        
        snprintf(b,sizeof(b),"blocks.%d.attn.q_norm.weight",blk); auto qn=w.get(b);
        snprintf(b,sizeof(b),"blocks.%d.attn.k_norm.weight",blk); auto kn=w.get(b);
        if(qn.defined()){auto r=torch::sqrt(torch::mean(q*q,-1,true)+1e-6f);q=q*qn/r;}
        if(kn.defined()){auto r=torch::sqrt(torch::mean(k*k,-1,true)+1e-6f);k=k*kn/r;}
        
        auto q1=q.slice(-1,0,half),q2=q.slice(-1,half),k1=k.slice(-1,0,half),k2=k.slice(-1,half);
        q=torch::cat({q1*cos_a-q2*sin_a,q1*sin_a+q2*cos_a},-1);
        k=torch::cat({k1*cos_a-k2*sin_a,k1*sin_a+k2*cos_a},-1);
        
        float sc=1.0f/sqrtf(head_dim);
        auto scores=torch::matmul(q,k.transpose(-2,-1))*sc;
        auto awts=torch::softmax(scores,-1);
        auto ao=torch::matmul(awts,v).transpose(0,1).contiguous().view({N,hidden});
        snprintf(b,sizeof(b),"blocks.%d.attn.o_proj.weight",blk); auto ow=w.get(b);
        snprintf(b,sizeof(b),"blocks.%d.attn.o_proj.bias",blk); auto ob=w.get(b);
        ao=torch::matmul(ao,ow.t());if(ob.defined())ao+=ob;
        
        h=h+gm*ao;
        
        auto norm2=torch::layer_norm(h,{hidden},{},{},1e-5);
        auto mod2=norm2*(1.0f+scl)+sml;
        snprintf(b,sizeof(b),"blocks.%d.ffn.fc1.weight",blk); auto fw1=w.get(b);
        snprintf(b,sizeof(b),"blocks.%d.ffn.fc1.bias",blk); auto fb1=w.get(b);
        auto ff=torch::matmul(mod2,fw1.t());if(fb1.defined())ff+=fb1;
        ff=0.5f*ff*(1.0f+torch::tanh(0.79788456f*(ff+0.044715f*ff*ff*ff)));
        snprintf(b,sizeof(b),"blocks.%d.ffn.fc2.weight",blk); auto fw2=w.get(b);
        snprintf(b,sizeof(b),"blocks.%d.ffn.fc2.bias",blk); auto fb2=w.get(b);
        ff=torch::matmul(ff,fw2.t());if(fb2.defined())ff+=fb2;
        h=h+gml*ff;
    if(blk==0) printf("Block0: in=%.4f out=%.4f\n",in_rms,h.std().item<float>());
    }
    
    auto py_h18=lb("debug_v4/py_h18.bin",{N,hidden});
    auto d18=(h-py_h18).abs().max().item<float>();
    printf("Block18: CppRMS=%.4f PyRMS=%.4f diff=%.2e %s\n",h.std().item<float>(),py_h18.std().item<float>(),d18,d18<1e-4?"OK":"FAIL");
    
    if(d18<1e-4) printf("\nFULL DiT MATCHES PERFECTLY!\n");
    else printf("\nDiverges at block loop.\n");
    
    return 0;
}
