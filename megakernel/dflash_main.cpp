#include "dflash_bridge.h"
#include "dflash_native.h"
#include "dflash_verify_policy.h"
#include "gguf.h"
#include "llama.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
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
        std::string model,draft,chunk="auto",backend="native";
        bool oracle=false, warmup=true;
        std::vector<std::string> prompt_files;
        int count=128,depth=7,ctx=4096,reps=1,kv_mode=1;
        float temp=0;
        unsigned long long seed=123;
        for (int i=1;i<argc;i++) {
            const std::string arg=argv[i];
            auto next=[&]() -> std::string { if (++i>=argc) throw std::runtime_error("missing argument value");return argv[i]; };
            if (arg=="-m") model=next();else if (arg=="--draft-model") draft=next();
            else if(arg=="--backend")backend=next();else if(arg=="--oracle")oracle=true;
            else if(arg=="--no-warmup")warmup=false;
            else if (arg=="--prompt-ids") prompt_files.push_back(next());else if (arg=="--prefill-chunk") chunk=next();
            else if (arg=="-n") count=std::stoi(next());else if (arg=="-d") depth=std::stoi(next());
            else if (arg=="-c") ctx=std::stoi(next());else if (arg=="--reps") reps=std::stoi(next());
            else if (arg=="--kv") {
                const auto k=next();
                if(k!="f16" && k!="q8" && k!="q4" && k!="auto") throw std::runtime_error("invalid KV format");
                kv_mode=k=="f16"?0:k=="q8"?1:k=="q4"?2:-1;
            }
            else if (arg=="--temp") temp=std::stof(next());else if (arg=="--seed") seed=std::stoull(next());
            else throw std::runtime_error("usage: qwen38_dflash -m target --draft-model draft --prompt-ids ids [-n 128] [-d 7] [-c 4096] [--reps 1] [--no-warmup] [--backend native|ggml] [--oracle] [--temp 0] [--seed 123] [--prefill-chunk auto] [--kv q8|q4|f16|auto]");
        }
        if (model.empty() || draft.empty() || prompt_files.empty() || count<1 || depth<1 || depth>7 || reps<1)
            throw std::runtime_error("target, draft, prompt IDs and valid generation parameters are required");
        if((backend!="native"&&backend!="ggml")||(oracle&&backend!="native"))throw std::runtime_error("backend must be native or ggml; oracle requires native");
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
        MKEngine target(model,ctx,0,"",kv_mode,size_t(backend=="native"?1792:2048)*1024*1024);
        std::unique_ptr<MKDFlashNative> native;
        std::unique_ptr<MKDFlash> bridge;
        if(backend=="native") {
            gguf::File draft_file(draft);
            auto dmp=llama_model_default_params();dmp.vocab_only=true;
            std::unique_ptr<llama_model,decltype(&llama_model_free)> dv_model(llama_model_load_from_file(draft.c_str(),dmp),llama_model_free);
            if(!dv_model)throw std::runtime_error("cannot load draft vocabulary");
            const auto * dv=llama_model_get_vocab(dv_model.get());
            if(llama_vocab_n_tokens(dv)!=llama_vocab_n_tokens(vocab))throw std::runtime_error("native draft vocabulary size differs");
            for(int t=0;t<llama_vocab_n_tokens(vocab);t++)
                if(std::strcmp(llama_vocab_get_text(vocab,t),llama_vocab_get_text(dv,t))||llama_vocab_get_attr(vocab,t)!=llama_vocab_get_attr(dv,t))
                    throw std::runtime_error("native draft token tables differ");
            native.reset(new MKDFlashNative(target,draft,ctx,llama_vocab_mask(dv),oracle?8:512));
            if(oracle)bridge.reset(new MKDFlash(target,draft,ctx,vocab,true));
            target.enable_device_features(native->layers(),[&](int pos,int n,const float * f){
                native->inject_device(pos,n,f);
                if(oracle){std::vector<float> host(size_t(n)*5*5120);
                    if(cudaMemcpy(host.data(),f,host.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("oracle feature copy failed");
                    bridge->inject(pos,n,host);}
            });
        } else {
            bridge.reset(new MKDFlash(target,draft,ctx,vocab));
            target.enable_features(bridge->layers(),[&](int pos,int n,const std::vector<float> & f){bridge->inject(pos,n,f);});
        }
        auto injection=[&](){return native?native->injection_seconds:bridge->injection_seconds;};
        auto drafting=[&](){return native?native->draft_seconds:bridge->draft_seconds;};
        target.set_sampling(temp,.95f,20,0,seed);
        if(oracle&&DFlashVerifyPolicy::environment()&&std::string(DFlashVerifyPolicy::environment())!="full")
            throw std::runtime_error("draft oracle requires full verification policy");
        using Clock=std::chrono::steady_clock;
        auto elapsed=[](Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();};
        for (size_t case_index=0;case_index<prompts.size();case_index++) {
          for (int rep=warmup?-1:0;rep<reps;rep++) {
            target.reset();if(native)native->reset();if(bridge)bridge->reset();
            DFlashVerifyPolicy policy(depth,DFlashVerifyPolicy::environment());
            const auto pf=Clock::now();
            std::vector<int> ids{target.prefill(prompts[case_index])};
            const double prefill=elapsed(pf),inject_prefill=injection();
            const double injection_gpu_prefill=native?native->injection_gpu_seconds:0;
            const auto start=Clock::now();
            double verify_s=0;
            const int limit=rep<0?16:count;
            while (int(ids.size())<limit && !llama_vocab_is_eog(vocab,ids.back())) {
                const bool budget=!std::getenv("MK_DFLASH_BUDGET")||std::strcmp(std::getenv("MK_DFLASH_BUDGET"),"0");
                const int width=policy.width();
                const int needed=budget?std::min(width,limit-int(ids.size())-1):width;
                std::vector<int> proposals;
                if(needed || !native) proposals=native?native->draft(target.position(),ids.back(),depth):bridge->draft(target.position(),ids.back(),depth);
                if(oracle && needed) {
                    auto ref=bridge->draft(target.position(),ids.back(),depth);
                    const auto a=native->lattice(),b=bridge->lattice();
                    int matches=0,total=0;double sse=0,scale=0,max_error=0;
                    for(int t=1;t<=depth;t++)for(int j=0;j<16;j++) {
                        int bj=-1;for(int k=0;k<16;k++)if(a[size_t(t)*5120+j]==b[size_t(t)*5120+k])bj=k;
                        matches++;if(bj<0){matches--;continue;}
                        for(int pred=0;pred<16;pred++) {
                            int bp=0;
                            if(t>1){bp=-1;for(int k=0;k<16;k++)if(a[size_t(t-1)*5120+pred]==b[size_t(t-1)*5120+k])bp=k;}
                            if(bp<0)continue;
                            const double av=a[size_t(t)*5120+16+pred*16+j],bv=b[size_t(t)*5120+16+bp*16+bj];
                            if(!std::isfinite(av)||!std::isfinite(bv))throw std::runtime_error("non-finite oracle selector score");
                            sse+=(av-bv)*(av-bv);scale+=bv*bv;max_error=std::max(max_error,std::abs(av-bv));total++;
                        }
                    }
                    const double nrmse=std::sqrt(sse/std::max(scale,1e-20));
                    printf("ORACLE {\"position\":%d,\"candidate_matches\":%d,\"candidate_total\":%d,\"score_pairs\":%d,\"nrmse\":%.9g,\"max_error\":%.9g,\"proposals_equal\":%s}\n",
                        target.position(),matches,depth*16,total,nrmse,max_error,proposals==ref?"true":"false");fflush(stdout);
                    if(matches<depth*16*.75 || nrmse>.05)throw std::runtime_error("native draft exceeds oracle candidate/score tolerance");
                }
                // Keep the trained full draft block; cap only verification.
                // A one-token tail needs no speculative draft at all.
                proposals.resize(needed);
                const auto verify=Clock::now();
                std::vector<int> out;
                const long before=target.stats.accepted;
                target.verify_external(proposals,out);
                if(needed==width)policy.observe(width,int(target.stats.accepted-before));
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
            printf("RESULT {\"case_index\":%zu,\"rep\":%d,\"tokens\":%zu,\"decode_seconds\":%.9f,\"tps\":%.6f,\"prefill_seconds\":%.9f,\"draft_seconds\":%.9f,\"verify_with_injection_seconds\":%.9f,\"injection_decode_seconds\":%.9f,\"injection_prefill_seconds\":%.9f,\"native_injection_prefill_gpu_seconds\":%.9f,\"native_injection_decode_gpu_seconds\":%.9f,\"cycles\":%ld,\"drafted\":%ld,\"accepted\":%ld}\n",
                   case_index,rep,ids.size(),decode,decode>0?(ids.size()-1)/decode:0,prefill,drafting(),verify_s,
                   injection()-inject_prefill,inject_prefill,injection_gpu_prefill,native?native->injection_gpu_seconds-injection_gpu_prefill:0,target.stats.cycles,target.stats.drafted,target.stats.accepted);
            printf("VERIFY_POLICY {\"three_cycles\":%ld,\"full_cycles\":%ld}\n",policy.three_cycles,policy.full_cycles);
            fflush(stdout);
          }
        }
    } catch (const std::exception & e) { fprintf(stderr,"[mk-dflash] %s\n",e.what());return 1; }
    return 0;
}
