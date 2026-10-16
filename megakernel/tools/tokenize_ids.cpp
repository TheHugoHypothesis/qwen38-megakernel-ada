// Vocab-only tokenizer for reproducible local benchmark prompts; no model tensors.
#include "llama.h"
#include <fstream>
#include <iterator>
#include <memory>
#include <cstdio>
#include <stdexcept>
#include <vector>
int main(int argc,char ** argv){try{
    if(argc!=4)throw std::runtime_error("usage: tokenize_ids model.gguf input.txt output.ids");
    std::ifstream f(argv[2],std::ios::binary);if(!f)throw std::runtime_error("cannot read text");
    std::string text((std::istreambuf_iterator<char>(f)),{});
    llama_backend_init();auto mp=llama_model_default_params();mp.vocab_only=true;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
    if(!model)throw std::runtime_error("cannot load vocabulary");
    const auto * v=llama_model_get_vocab(model.get());
    int n=llama_tokenize(v,text.data(),text.size(),nullptr,0,false,true);
    if(n>=0)throw std::runtime_error("unexpected empty tokenization");
    std::vector<int> ids(-n);n=llama_tokenize(v,text.data(),text.size(),ids.data(),ids.size(),false,true);
    if(n<0)throw std::runtime_error("tokenizer size changed");
    std::ofstream out(argv[3]);for(int id:ids)out<<id<<' ';out<<'\n';
    if(!out)throw std::runtime_error("cannot write IDs");printf("%d tokens\n",n);
}catch(const std::exception & e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}return 0;}
