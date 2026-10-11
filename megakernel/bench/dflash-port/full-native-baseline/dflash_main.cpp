#include "dflash_bridge.h"
#include "llama.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc,char ** argv) {
    try {
        std::string model,draft,chunk="512";
        std::vector<std::string> prompt_files;
        int count=128,depth=4,ctx=4096,reps=1;
        float temp=0;
        unsigned long long seed=123;
        for (int i=1;i<argc;i++) {
            const std::string arg=argv[i];
            auto next=[&]() -> std::string { if (++i>=argc) throw std::runtime_error("missing argument value");return argv[i]; };
            if (arg=="-m") model=next();else if (arg=="--draft-model") draft=next();
            else if (arg=="--prompt-ids") prompt_files.push_back(next());else if (arg=="--prefill-chunk") chunk=next();
            else if (arg=="-n") count=std::stoi(next());else if (arg=="-d") depth=std::stoi(next());
            else if (arg=="-c") ctx=std::stoi(next());else if (arg=="--reps") reps=std::stoi(next());
            else if (arg=="--temp") temp=std::stof(next());else if (arg=="--seed") seed=std::stoull(next());
            else throw std::runtime_error("usage: qwen38_dflash -m target --draft-model draft --prompt-ids ids [-n 128] [-d 4] [-c 4096] [--reps 1] [--temp 0] [--seed 123] [--prefill-chunk 512]");
        }
        if (model.empty() || draft.empty() || prompt_files.empty() || count<1 || depth<1 || depth>4 || reps<1)
            throw std::runtime_error("target, draft, prompt IDs and valid generation parameters are required");
        std::vector<std::vector<int>> prompts;
        for (const auto & path:prompt_files) {
            std::ifstream input(path);
            if (!input) throw std::runtime_error("cannot open prompt IDs");
            std::vector<int> prompt;int id;
            while (input>>id) prompt.push_back(id);
            if (!input.eof() || prompt.empty()) throw std::runtime_error("invalid prompt IDs file");
            prompts.push_back(std::move(prompt));
        }
        setenv("MK_PREFILL_CHUNK",chunk.c_str(),1);
        llama_backend_init();
        auto mp=llama_model_default_params();mp.vocab_only=true;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> vocab_model(llama_model_load_from_file(model.c_str(),mp),llama_model_free);
        if (!vocab_model) throw std::runtime_error("cannot load target vocabulary");
        const auto * vocab=llama_model_get_vocab(vocab_model.get());
        MKEngine target(model,ctx,0,"",1);
        MKDFlash bridge(target,draft,ctx,vocab);
        target.enable_features(bridge.layers(),[&](int pos,int n,const std::vector<float> & f){bridge.inject(pos,n,f);});
        target.set_sampling(temp,.95f,20,0,seed);
        using Clock=std::chrono::steady_clock;
        auto elapsed=[](Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();};
        for (size_t case_index=0;case_index<prompts.size();case_index++) {
          for (int rep=-1;rep<reps;rep++) {
            target.reset();bridge.reset();
            const auto pf=Clock::now();
            std::vector<int> ids{target.prefill(prompts[case_index])};
            const double prefill=elapsed(pf),inject_prefill=bridge.injection_seconds;
            const auto start=Clock::now();
            double verify_s=0;
            const int limit=rep<0?16:count;
            while (int(ids.size())<limit && !llama_vocab_is_eog(vocab,ids.back())) {
                auto proposals=bridge.draft(target.position(),ids.back(),depth);
                const auto verify=Clock::now();
                std::vector<int> out;
                target.verify_external(proposals,out);
                verify_s+=elapsed(verify);
                for (int tok:out) {
                    if (int(ids.size())>=limit) break;
                    ids.push_back(tok);
                    if (llama_vocab_is_eog(vocab,tok)) break;
                }
            }
            const double decode=elapsed(start);
            if (rep<0) continue;
            printf("IDS:");for (int tok:ids) printf(" %d",tok);printf("\n");
            printf("TEXT_HEX:");
            for (int tok:ids) {
                if (llama_vocab_is_eog(vocab,tok)) break;
                std::vector<char> piece(256);
                int n=llama_token_to_piece(vocab,tok,piece.data(),int(piece.size()),0,true);
                if (n<0) { piece.resize(-n);n=llama_token_to_piece(vocab,tok,piece.data(),int(piece.size()),0,true); }
                if (n<0) throw std::runtime_error("cannot decode generated token");
                for (int i=0;i<n;i++) printf("%02x",static_cast<unsigned char>(piece[i]));
            }
            printf("\n");
            printf("RESULT {\"case_index\":%zu,\"rep\":%d,\"tokens\":%zu,\"decode_seconds\":%.9f,\"tps\":%.6f,\"prefill_seconds\":%.9f,\"draft_seconds\":%.9f,\"verify_with_injection_seconds\":%.9f,\"injection_decode_seconds\":%.9f,\"injection_prefill_seconds\":%.9f,\"cycles\":%ld,\"drafted\":%ld,\"accepted\":%ld}\n",
                   case_index,rep,ids.size(),decode,decode>0?(ids.size()-1)/decode:0,prefill,bridge.draft_seconds,verify_s,
                   bridge.injection_seconds-inject_prefill,inject_prefill,target.stats.cycles,target.stats.drafted,target.stats.accepted);
            fflush(stdout);
          }
        }
    } catch (const std::exception & e) { fprintf(stderr,"[mk-dflash] %s\n",e.what());return 1; }
    return 0;
}
