#pragma once
#include "qwen38_mk.h"
#include <memory>

// Complete DFlash2 compute path in native CUDA. No llama context/graph execution.
class MKDFlashNative {
public:
    MKDFlashNative(MKEngine & target,const std::string & model,int context,int mask,int chunk=512);
    ~MKDFlashNative();
    MKDFlashNative(const MKDFlashNative &)=delete;
    MKDFlashNative & operator=(const MKDFlashNative &)=delete;
    void reset();
    void inject(int position,int count,const std::vector<float> & features);
    void inject_device(int position,int count,const float * features);
    std::vector<int> draft(int position,int anchor,int depth);
    const std::vector<int> & layers() const;
    int cache_window() const;
    void checkpoint();
    bool restore_checkpoint();
    void truncate(int position);
    std::vector<float> lattice() const;
    double injection_seconds=0,draft_seconds=0;
    double injection_gpu_seconds=0; // excludes waiting for target features
    size_t allocated_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
