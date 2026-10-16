// Trace real native-DFlash generation using the unchanged production CUDA object.
#include "../qwen38_mk.h"
#include "../dflash_native.h"
#include "llama.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char ** argv){
 try{
    if(argc!=7)throw std::runtime_error("usage: dflash_features_test model draft prompt.ids output_prefix output_tokens depth");
    std::ifstream input(argv[3]); std::vector<int> prompt; int id;
    while(input>>id)prompt.push_back(id);
    if(!input.eof()||prompt.empty())throw std::runtime_error("invalid prompt IDs");
    const int limit=std::stoi(argv[5]),depth=std::stoi(argv[6]);
    if(limit<2||depth<1||depth>7)throw std::runtime_error("invalid generation parameters");
    setenv("MK_PREFILL_CHUNK","128",1);
    llama_backend_init(); auto mp=llama_model_default_params();mp.vocab_only=true;
    using Model=std::unique_ptr<llama_model,decltype(&llama_model_free)>;
    Model dm(llama_model_load_from_file(argv[2],mp),llama_model_free);
    Model tm(llama_model_load_from_file(argv[1],mp),llama_model_free);
    if(!dm||!tm)throw std::runtime_error("cannot load vocabularies");
    const auto * vocab=llama_model_get_vocab(tm.get());
    MKEngine target(argv[1],4096,0,"",1,size_t(1792)*1024*1024);
    MKDFlashNative draft(target,argv[2],4096,llama_vocab_mask(llama_model_get_vocab(dm.get())));
    std::vector<int> layers(64);std::iota(layers.begin(),layers.end(),0);
    float * selected=nullptr;check(cudaMalloc(&selected,size_t(128)*5*5120*sizeof(float)));
    std::ofstream features(std::string(argv[4])+".f32",std::ios::binary);
    std::ofstream events(std::string(argv[4])+".jsonl");
    if(!features||!events)throw std::runtime_error("cannot write diagnostic outputs");
    target.enable_device_features(layers,[&](int pos,int n,const float * f){
        // The real draft still receives exactly the same five taps and layout.
        for(int k=0;k<5;k++)check(cudaMemcpy2D(selected+size_t(k)*5120,size_t(5)*5120*4,
            f+size_t(draft.layers()[k])*5120,size_t(64)*5120*4,5120*4,n,cudaMemcpyDeviceToDevice));
        draft.inject_device(pos,n,selected);
        std::vector<float> host(size_t(n)*64*5120);check(cudaMemcpy(host.data(),f,host.size()*4,cudaMemcpyDeviceToHost));
        features.seekp(size_t(pos)*64*5120*4);features.write(reinterpret_cast<const char *>(host.data()),host.size()*4);
        if(!features)throw std::runtime_error("feature write failed");
        events<<"{\"position\":"<<pos<<",\"count\":"<<n<<"}\n";
    });
    std::vector<int> ids{target.prefill(prompt)};
    while(int(ids.size())<limit&&!llama_vocab_is_eog(vocab,ids.back())){
        const int needed=std::min(depth,limit-int(ids.size())-1);
        auto p=needed?draft.draft(target.position(),ids.back(),depth):std::vector<int>{};
        p.resize(needed);std::vector<int> out;const int pos=target.position();
        const long accepted=target.stats.accepted;
        target.verify_external(p,out);
        events<<"{\"cycle_position\":"<<pos<<",\"accepted\":"<<target.stats.accepted-accepted<<",\"draft\":[";
        for(size_t i=0;i<p.size();i++)events<<(i?",":"")<<p[i];events<<"]}\n";
        for(int tok:out){if(int(ids.size())>=limit)break;ids.push_back(tok);if(llama_vocab_is_eog(vocab,tok))break;}
    }
    std::ofstream output(std::string(argv[4])+".ids");for(int tok:ids)output<<tok<<' ';output<<'\n';
    check(cudaFree(selected));printf("PASS traced %zu output tokens, %ld cycles, all 64 layer inputs\n",ids.size(),target.stats.cycles);
 }catch(const std::exception & e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}return 0;
}
