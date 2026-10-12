// Validate the entire specialized tensor schema before allocating GPU memory.
#pragma once
#include "gguf.h"
#include <cmath>
#include <set>
namespace gguf {
inline size_t validate_qwen38(const File & f, bool include_mtp = true) {
    auto integer = [&](const char * key, uint64_t value) {
        const auto it = f.ints.find(key);
        if (it == f.ints.end() || it->second != value) throw std::runtime_error(std::string("unsupported metadata: ")+key);
    };
    const auto arch = f.strs.find("general.architecture");
    if (arch == f.strs.end() || arch->second != "qwen35") throw std::runtime_error("expected qwen35 architecture");
    const auto nextn = f.ints.find("qwen35.nextn_predict_layers");
    const uint64_t mtp = nextn == f.ints.end() ? 0 : nextn->second;
    if (mtp > 1) throw std::runtime_error("expected zero or one MTP block");
    integer("qwen35.block_count",64 + mtp); integer("qwen35.embedding_length",5120);
    integer("qwen35.feed_forward_length",17408); integer("qwen35.attention.head_count",24);
    integer("qwen35.attention.head_count_kv",4); integer("qwen35.attention.key_length",256);
    integer("qwen35.attention.value_length",256); integer("qwen35.ssm.conv_kernel",4);
    integer("qwen35.ssm.state_size",128); integer("qwen35.ssm.group_count",16);
    integer("qwen35.ssm.time_step_rank",48); integer("qwen35.ssm.inner_size",6144);
    integer("qwen35.full_attention_interval",4); integer("qwen35.rope.dimension_count",64);
    const auto & emb = f.get("token_embd.weight");
    const auto token_count = f.array_sizes.find("tokenizer.ggml.tokens");
    if (emb.ne.size() != 2 || emb.ne[1] < 32 || emb.ne[1] > 248320 || (emb.ne[1] & 1) ||
        token_count == f.array_sizes.end() || token_count->second != emb.ne[1])
        throw std::runtime_error("unsupported vocabulary: expected matching tokenizer and even embedding rows in [32,248320]");
    const uint64_t vocab = emb.ne[1];
    for (auto pair : {std::pair<const char *,double>{"qwen35.rope.freq_base",1e7}, {"qwen35.attention.layer_norm_rms_epsilon",1e-6}}) {
        const auto it=f.reals.find(pair.first);
        if(it==f.reals.end() || !std::isfinite(it->second) || std::abs(it->second-pair.second)>std::abs(pair.second)*1e-6)
            throw std::runtime_error(std::string("unsupported metadata: ")+pair.first);
    }
    const auto sections=f.arrays.find("qwen35.rope.dimension_sections");
    if(sections==f.arrays.end() || sections->second!=std::vector<uint64_t>{11,11,10,0})
        throw std::runtime_error("unsupported mRoPE sections");
    size_t bytes=0;
    std::set<std::string> used;
    auto tensor = [&](const std::string & name, std::vector<uint64_t> shape, bool matrix, bool alpha=false) {
        const Tensor & t=f.get(name);
        if(t.ne!=shape) throw std::runtime_error(name+": unexpected tensor shape");
        if(matrix) {
            if(t.type!=Q4_K && t.type!=Q5_K && t.type!=Q6_K && t.type!=Q8_0 && !is_native_quant(t.type))
                throw std::runtime_error(name+": unsupported matrix quantization");
        } else if (alpha) {
            if(t.type!=F32 && t.type!=F16 && t.type!=BF16) throw std::runtime_error(name+": expected float alpha/beta");
        } else if(t.type!=F32) throw std::runtime_error(name+": expected F32 scalar weights");
        if (include_mtp || name.rfind("blk.64.", 0) != 0)
            bytes += alpha ? size_t(5120)*48*4 : t.type==Q6_K ? t.nbytes/210*224 : t.nbytes;
        used.insert(name);
    };
    for(int i=0;i<int(64+mtp);i++) {
        const std::string p="blk."+std::to_string(i)+".";
        tensor(p+"attn_norm.weight",{5120},false); tensor(p+"post_attention_norm.weight",{5120},false);
        if(i==64 || (i+1)%4==0) {
            tensor(p+"attn_q.weight",{5120,12288},true);
            for(const char * k:{"attn_k.weight","attn_v.weight"}) tensor(p+k,{5120,1024},true);
            tensor(p+"attn_output.weight",{6144,5120},true);
            for(const char * k:{"attn_q_norm.weight","attn_k_norm.weight"}) tensor(p+k,{256},false);
        } else {
            tensor(p+"attn_qkv.weight",{5120,10240},true); tensor(p+"attn_gate.weight",{5120,6144},true);
            tensor(p+"ssm_out.weight",{6144,5120},true); tensor(p+"ssm_conv1d.weight",{4,10240},false);
            for(const char * k:{"ssm_a","ssm_dt.bias"}) tensor(p+k,{48},false);
            tensor(p+"ssm_norm.weight",{128},false);
            for(const char * k:{"ssm_alpha.weight","ssm_beta.weight"}) tensor(p+k,{5120,48},false,true);
        }
        for(const char * k:{"ffn_gate.weight","ffn_up.weight"}) tensor(p+k,{5120,17408},true);
        tensor(p+"ffn_down.weight",{17408,5120},true);
    }
    if (mtp) {
    tensor("blk.64.nextn.eh_proj.weight",{10240,5120},true);
    for(const char * k:{"nextn.enorm.weight","nextn.hnorm.weight","nextn.shared_head_norm.weight"}) tensor(std::string("blk.64.")+k,{5120},false);
    }
    tensor("output_norm.weight",{5120},false);
    tensor("token_embd.weight",{5120,vocab},true); tensor("output.weight",{5120,vocab},true);
    const auto head=f.get("output.weight").type;
    if(head!=Q4_K && head!=Q5_K && head!=Q6_K && head!=Q8_0)
        throw std::runtime_error("output.weight requires an existing prefill GEMM format");
    if(vocab%128 && head!=Q4_K)
        throw std::runtime_error("condensed unaligned output.weight requires Q4_K");
    if(used.size()!=f.tensors.size()) throw std::runtime_error("unexpected extra tensors in specialized model");
    return bytes;
}
} // namespace gguf
