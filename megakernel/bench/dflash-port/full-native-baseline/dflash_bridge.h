#pragma once
#include "qwen38_mk.h"
#include <memory>

struct llama_vocab;

// Experimental bridge against the local llama.cpp ABI. Target owns shared weights.
class MKDFlash {
public:
    MKDFlash(MKEngine & target, const std::string & draft_path, int context,
             const llama_vocab * target_vocab);
    ~MKDFlash();
    MKDFlash(const MKDFlash &) = delete;
    MKDFlash & operator=(const MKDFlash &) = delete;
    void reset();
    void inject(int position, int count, const std::vector<float> & features);
    std::vector<int> draft(int position, int anchor, int depth);
    const std::vector<int> & layers() const;
    double injection_seconds = 0;
    double draft_seconds = 0;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
