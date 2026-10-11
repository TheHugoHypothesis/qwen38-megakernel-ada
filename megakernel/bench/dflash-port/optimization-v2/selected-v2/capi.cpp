// C API over MKEngine plus libllama's tokenizer (vocab only), for the HTTP server (server/mk_server.py).
#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"
#include "qwen38_mk.h"
#include "dflash_native.h"

namespace {
struct Handle {
    MKEngine * eng = nullptr;
    llama_model * lm = nullptr;
    const llama_vocab * vocab = nullptr;
    MKDFlashNative * dflash=nullptr;
    int draft_depth=0;
    std::string err;
    ~Handle() { delete dflash;delete eng; if (lm) llama_model_free(lm); }
};
thread_local std::string g_err;
}

extern "C" {

const char * mk_last_error() { return g_err.c_str(); }

void * mk_new(const char * model, int ctx, int n_draft, const char * draft_vocab, int kv_mode) {
    try {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        auto h = std::make_unique<Handle>();
        h->lm = llama_model_load_from_file(model, mp);
        if (!h->lm) throw std::runtime_error("failed to load the vocabulary");
        h->vocab = llama_model_get_vocab(h->lm);
        h->eng = new MKEngine(model, ctx, n_draft, draft_vocab ? draft_vocab : "", kv_mode);
        return h.release();
    } catch (const std::exception & e) {
        g_err = e.what();
        return nullptr;
    }
}

void mk_free(void * hp) {
    auto * h = static_cast<Handle *>(hp);
    delete h;
}

void * mk_new_dflash(const char * model,const char * draft,int ctx,int depth,int kv_mode) {
    try {
        if(!model||!draft||depth<1||depth>7)throw std::runtime_error("DFlash2 needs target, draft and depth in [1,7]");
        llama_backend_init();auto mp=llama_model_default_params();mp.vocab_only=true;
        auto h=std::make_unique<Handle>();
        h->lm=llama_model_load_from_file(model,mp);
        if(!h->lm)throw std::runtime_error("cannot load target vocabulary");
        h->vocab=llama_model_get_vocab(h->lm);
        std::unique_ptr<llama_model,decltype(&llama_model_free)> dm(llama_model_load_from_file(draft,mp),llama_model_free);
        if(!dm)throw std::runtime_error("cannot load draft vocabulary");
        const auto * dv=llama_model_get_vocab(dm.get());
        if(llama_vocab_n_tokens(dv)!=llama_vocab_n_tokens(h->vocab)||llama_vocab_bos(dv)!=llama_vocab_bos(h->vocab)||
           llama_vocab_eos(dv)!=llama_vocab_eos(h->vocab)||llama_vocab_type(dv)!=llama_vocab_type(h->vocab))
            throw std::runtime_error("draft/target vocabulary metadata differ");
        for(int id=0;id<llama_vocab_n_tokens(dv);id++)
            if(std::strcmp(llama_vocab_get_text(dv,id),llama_vocab_get_text(h->vocab,id))||
               llama_vocab_get_attr(dv,id)!=llama_vocab_get_attr(h->vocab,id))
                throw std::runtime_error("draft/target token tables differ");
        // Draft weights, sliding KV/checkpoint, encoder workspace, feature
        // capture and CUDA graph/library margin must precede auto chunk sizing.
        h->eng=new MKEngine(model,ctx,0,"",kv_mode,size_t(1792)*1024*1024);
        h->dflash=new MKDFlashNative(*h->eng,draft,ctx,llama_vocab_mask(dv));h->draft_depth=depth;
        auto * owner=h.get();
        h->eng->enable_device_features(h->dflash->layers(),[owner](int pos,int n,const float * f){owner->dflash->inject_device(pos,n,f);});
        return h.release();
    } catch(const std::exception & e){g_err=e.what();return nullptr;}
}

// chat template stored in the GGUF (tokenizer.chat_template); returns its length, copies up to cap
int mk_chat_template(void * hp, char * buf, int cap) {
    auto * h = static_cast<Handle *>(hp);
    const char * t = llama_model_chat_template(h->lm, nullptr);
    if (!t) return -1;
    const int n = (int) strlen(t);
    if (buf && cap > 0) memcpy(buf, t, std::min(n, cap));
    return n;
}

// text -> token ids (special tokens such as <|im_start|> parsed); returns the count (negative: -needed)
int mk_tokenize(void * hp, const char * text, int len, int * out, int cap) {
    auto * h = static_cast<Handle *>(hp);
    return llama_tokenize(h->vocab, text, len, out, cap, false, true);
}

// token -> its bytes (special tokens rendered as text); returns the byte count
int mk_piece(void * hp, int tok, char * buf, int cap) {
    auto * h = static_cast<Handle *>(hp);
    return llama_token_to_piece(h->vocab, tok, buf, cap, 0, true);
}

int mk_is_eog(void * hp, int tok) { return llama_vocab_is_eog(static_cast<Handle *>(hp)->vocab, tok) ? 1 : 0; }

int mk_set_sampling(void * hp, float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    try {
        static_cast<Handle *>(hp)->eng->set_sampling(temp, top_p, top_k, min_p, seed);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// appends tokens to the sequence; returns the first generated token (negative on error)
int mk_prefill(void * hp, const int * toks, int n) {
    try {
        auto * h=static_cast<Handle *>(hp);
        if(!toks||n<1)throw std::runtime_error("prefill requires token IDs");
        int first=h->eng->prefill(std::vector<int>(toks,toks+n));
        if(h->dflash)h->dflash->checkpoint();
        return first;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// One decode cycle. DFlash bounds verification/output by cap, preserving its
// full trained draft block; cap=1 commits the anchor without a speculative draft.
// MTP retains its minimum buffer capacity. Returns the count (negative on error).
int mk_step(void * hp, int * out, int cap) {
    try {
        std::vector<int> v;
        auto * h=static_cast<Handle *>(hp);
        if(!out||cap<1||(!h->dflash&&cap<h->eng->n_draft()+1))throw std::runtime_error("step output capacity is too small");
        if(h->dflash) {
            std::vector<int> proposals;
            if(cap>1) {
                proposals=h->dflash->draft(h->eng->position(),h->eng->sequence().back(),h->draft_depth);
                proposals.resize(std::min(int(proposals.size()),cap-1));
            }
            h->eng->verify_external(proposals,v);
        } else h->eng->step(v);
        const int n = std::min((int) v.size(), cap);
        for (int i = 0; i < n; i++) out[i] = v[i];
        return n;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

void mk_reset(void * hp) { auto * h=static_cast<Handle *>(hp);h->eng->reset();if(h->dflash)h->dflash->reset(); }

int mk_truncate(void * hp, int len) {
    try {
        auto * h=static_cast<Handle *>(hp);h->eng->truncate(len);if(h->dflash)h->dflash->truncate(len);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// copies the sequence so far (up to cap tokens); returns its length
int mk_sequence(void * hp, int * out, int cap) {
    const auto & s = static_cast<Handle *>(hp)->eng->sequence();
    const int n = (int) s.size();
    if (out)
        for (int i = 0; i < std::min(n, cap); i++) out[i] = s[i];
    return n;
}

int mk_committed(void * hp) { return static_cast<Handle *>(hp)->eng->committed(); }
int mk_checkpoint_pos(void * hp) { return static_cast<Handle *>(hp)->eng->checkpoint_pos(); }
int mk_rewind_to_checkpoint(void * hp) {
    try {auto * h=static_cast<Handle *>(hp);
        if(!h->eng->rewind_to_checkpoint())return -1;
        if(h->dflash&&!h->dflash->restore_checkpoint())throw std::runtime_error("draft checkpoint missing");
        return 0;
    } catch(const std::exception & e){g_err=e.what();return -1;}
}

void mk_stats(void * hp, long * cycles, long * drafted, long * accepted) {
    const auto & s = static_cast<Handle *>(hp)->eng->stats;
    *cycles = s.cycles;
    *drafted = s.drafted;
    *accepted = s.accepted;
}

} // extern "C"
