#include "../qwen38_mk.h"
#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc,char ** argv) {
 try {
    if(argc!=5)throw std::runtime_error("usage: target_features_test mk|llama model.gguf tokens.ids output.f32");
    std::ifstream input(argv[3]);std::vector<int> tokens;int id;
    while(input>>id)tokens.push_back(id);
    if(!input.eof()||tokens.empty())throw std::runtime_error("invalid IDs");
    const std::vector<int> layers{6,20,34,48,62};
    std::vector<int> positions{0,int(tokens.size()/2),int(tokens.size()-1)};
    std::vector<float> result(positions.size()*layers.size()*5120);
    const std::string mode=argv[1];
    if(mode=="mk") {
        setenv("MK_PREFILL_CHUNK","128",1);
        MKEngine engine(argv[2],4096,0,"",1);
        engine.enable_features(layers,[&](int pos,int n,const std::vector<float> & f) {
            for(size_t i=0;i<positions.size();i++)if(positions[i]>=pos&&positions[i]<pos+n)
                std::copy_n(f.data()+size_t(positions[i]-pos)*5*5120,5*5120,result.data()+i*5*5120);
        });
        engine.prefill(tokens);
    } else if(mode=="llama") {
        llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=99;mp.load_mtp=false;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[2],mp),llama_model_free);
        if(!model)throw std::runtime_error("cannot load llama target");
        auto cp=llama_context_default_params();cp.n_ctx=4096;cp.n_seq_max=1;cp.n_batch=128;cp.n_ubatch=128;
        cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q8_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.n_threads=4;cp.n_threads_batch=4;
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
        if(!ctx)throw std::runtime_error("cannot initialize llama target");
        for(int layer:layers)llama_set_embeddings_layer_inp(ctx.get(),layer,true);
        auto batch=llama_batch_init(128,0,1);
        for(size_t off=0;off<tokens.size();off+=128) {
            int n=std::min<size_t>(128,tokens.size()-off);batch.n_tokens=n;
            for(int t=0;t<n;t++){batch.token[t]=tokens[off+t];batch.pos[t]=off+t;batch.n_seq_id[t]=1;batch.seq_id[t][0]=0;batch.logits[t]=t==n-1;}
            if(llama_decode(ctx.get(),batch))throw std::runtime_error("llama teacher forcing failed");
            for(size_t k=0;k<layers.size();k++) {
                const float * f=llama_get_embeddings_layer_inp(ctx.get(),layers[k]);
                for(size_t i=0;i<positions.size();i++)if(positions[i]>=int(off)&&positions[i]<int(off)+n)
                    std::copy_n(f+size_t(positions[i]-off)*5120,5120,result.data()+(i*5+k)*5120);
            }
        }
        llama_batch_free(batch);
    } else throw std::runtime_error("mode must be mk or llama");
    std::ofstream out(argv[4],std::ios::binary);out.write(reinterpret_cast<const char *>(result.data()),result.size()*4);
    if(!out)throw std::runtime_error("cannot write feature file");
    printf("PASS dumped three positions: 0 %d %d, five layers, F32\n",positions[1],positions[2]);
 } catch(const std::exception & e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}return 0;
}
