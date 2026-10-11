#include "dflash_bridge.h"
#include "gguf.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-ext.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
}

struct MKDFlash::Impl {
    llama_model * model=nullptr;
    llama_context * ctx=nullptr;
    llama_batch_ext * batch=nullptr;
    std::vector<int> taps;
    int mask=-1,top_k=0,block_size=0;
    ~Impl() {
        if (batch) llama_batch_ext_free(batch);
        if (ctx) llama_free(ctx);
        if (model) { model->tok_embd=nullptr;model->output=nullptr;llama_model_free(model); }
    }
};

MKDFlash::MKDFlash(MKEngine & target,const std::string & path,int context,const llama_vocab * vocab)
    : p_(new Impl) {
    gguf::File file(path);
    if (file.strs.at("general.architecture")!="dflash") throw std::runtime_error("expected DFlash draft");
    for (auto pair:{std::pair<const char *,uint64_t>{"dflash.embedding_length",5120},
                   {"dflash.block_count",5},{"dflash.block_size",8},
                   {"dflash.attention.head_count",32},{"dflash.attention.head_count_kv",8},
                   {"dflash.conv_kernel_size",2},{"dflash.conv_group_size",16},
                   {"dflash.selector_rank",256},{"dflash.selector_top_k",16}})
        if (file.ints.at(pair.first)!=pair.second) throw std::runtime_error(std::string("unsupported draft metadata: ")+pair.first);
    const auto & taps=file.arrays.at("dflash.target_layers");
    if (taps!=std::vector<uint64_t>{6,20,34,48,62}) throw std::runtime_error("unsupported target feature taps");
    if (file.tensors.count("token_embd.weight") || file.tensors.count("output.weight"))
        throw std::runtime_error("this bridge expects borrowed target embedding/head");
    auto weights=target.share_weights();
    auto mp=llama_model_default_params();
    mp.n_gpu_layers=99;mp.load_mtp=false;
    p_->model=llama_model_load_from_file(path.c_str(),mp);
    if (!p_->model) throw std::runtime_error("cannot load DFlash2 model");
    auto * dv=llama_model_get_vocab(p_->model);
    if (llama_vocab_n_tokens(dv)!=llama_vocab_n_tokens(vocab) ||
        llama_vocab_type(dv)!=llama_vocab_type(vocab) ||
        llama_vocab_bos(dv)!=llama_vocab_bos(vocab) || llama_vocab_eos(dv)!=llama_vocab_eos(vocab))
        throw std::runtime_error("draft/target tokenizer metadata differ");
    for (int i=0;i<llama_vocab_n_tokens(vocab);i++)
        if (std::strcmp(llama_vocab_get_text(vocab,i),llama_vocab_get_text(dv,i)) ||
            llama_vocab_get_attr(vocab,i)!=llama_vocab_get_attr(dv,i))
            throw std::runtime_error("draft/target token tables differ");
    p_->model->tok_embd=weights.embedding;
    p_->model->output=weights.output;
    p_->taps.assign(taps.begin(),taps.end());
    p_->top_k=int(file.ints.at("dflash.selector_top_k"));
    p_->block_size=int(file.ints.at("dflash.block_size"));
    p_->mask=llama_vocab_mask(dv);
    if (p_->mask<0) throw std::runtime_error("draft mask token is absent");
    auto cp=llama_context_default_params();
    cp.n_ctx=context;cp.n_seq_max=1;cp.n_batch=512;cp.n_ubatch=512;
    cp.n_outputs_max=5;cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q8_0;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.attention_type=LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    cp.n_threads=4;cp.n_threads_batch=4;
    p_->ctx=llama_init_from_model(p_->model,cp);
    if (!p_->ctx) throw std::runtime_error("cannot initialize DFlash2 context with shared weights");
    llama_set_embeddings_nextn(p_->ctx,true,false);
    p_->batch=llama_batch_ext_init(p_->ctx);
    if (!p_->batch) throw std::runtime_error("cannot allocate draft batch");
}

MKDFlash::~MKDFlash()=default;
const std::vector<int> & MKDFlash::layers() const { return p_->taps; }

void MKDFlash::reset() {
    llama_memory_clear(llama_get_memory(p_->ctx),false);
    injection_seconds=0;draft_seconds=0;
}

void MKDFlash::inject(int position,int count,const std::vector<float> & features) {
    const int width=5120*int(p_->taps.size());
    if (position<0 || count<=0 || features.size()!=size_t(count)*width)
        throw std::runtime_error("invalid target feature batch");
    for (float v:features) if (!std::isfinite(v)) throw std::runtime_error("non-finite target feature");
    const auto start=Clock::now();
    for (int off=0;off<count;off+=512) {
        const int n=std::min(512,count-off);
        llama_batch_ext_clear(p_->batch);
        for (int i=0;i<n;i++) {
            const int idx=llama_batch_ext_add_embd(p_->batch,0,{features.data()+size_t(off+i)*width,1,size_t(width)});
            const llama_pos pos[4]={position+off+i,position+off+i,position+off+i,0};
            if (idx<0 || !llama_batch_ext_set_pos(p_->batch,idx,pos)) throw std::runtime_error("cannot add target features");
        }
        if (llama_process(p_->ctx,LLAMA_PROCESS_TYPE_DECODE,p_->batch)!=0)
            throw std::runtime_error("draft feature injection failed");
    }
    llama_synchronize(p_->ctx);
    injection_seconds+=seconds(start);
}

std::vector<int> MKDFlash::draft(int position,int anchor,int depth) {
    if (depth<1 || depth>4 || depth>=p_->block_size || position<0)
        throw std::runtime_error("hybrid draft depth must be in [1,4]");
    const auto start=Clock::now();
    llama_batch_ext_clear(p_->batch);
    for (int i=0;i<=depth;i++) {
        const int idx=llama_batch_ext_add_token(p_->batch,0,i==0?anchor:p_->mask);
        const llama_pos pos[4]={position+i,position+i,position+i,0};
        if (idx<0 || !llama_batch_ext_set_pos(p_->batch,idx,pos)) throw std::runtime_error("cannot add draft noise block");
    }
    if (llama_process(p_->ctx,LLAMA_PROCESS_TYPE_DECODE,p_->batch)!=0)
        throw std::runtime_error("DFlash2 block decode failed");
    const float * lattice=llama_get_embeddings_nextn(p_->ctx);
    if (!lattice) throw std::runtime_error("DFlash2 selector did not return a lattice");
    std::vector<int> result;
    int pred=0;
    for (int i=1;i<=depth;i++) {
        const float * row=lattice+size_t(i)*5120;
        const float * scores=row+p_->top_k+size_t(pred)*p_->top_k;
        for (int j=0;j<p_->top_k;j++) if (!std::isfinite(scores[j])) throw std::runtime_error("non-finite selector score");
        pred=int(std::max_element(scores,scores+p_->top_k)-scores);
        const int id=int(row[pred]);
        if (id<0 || id>=248320 || row[pred]!=float(id)) throw std::runtime_error("invalid selector token");
        result.push_back(id);
    }
    // Only target features of accepted positions may remain in the draft KV.
    if (!llama_memory_seq_rm(llama_get_memory(p_->ctx),0,position,-1))
        throw std::runtime_error("cannot remove speculative draft noise");
    draft_seconds+=seconds(start);
    return result;
}
