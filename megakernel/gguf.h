// Minimal read-only GGUF parser (mmap based). Only what the megakernel loader needs.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gguf {

enum GType : uint32_t { F32 = 0, F16 = 1, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q2_K = 10, IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, IQ1_M = 29, BF16 = 30 };

struct Tensor {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> ne;
    uint64_t offset = 0;
    const uint8_t * data = nullptr;
    uint64_t nbytes = 0;
};

inline bool is_native_quant(uint32_t t) {
    return t == Q2_K || t == IQ2_XXS || t == IQ2_XS || t == IQ2_S || t == IQ3_XXS || t == IQ3_S || t == IQ4_XS || t == IQ1_M;
}

inline uint64_t type_row_bytes(uint32_t type, uint64_t n) {
    if (type != F32 && type != F16 && type != BF16 && n % (type == Q8_0 ? 32 : 256)) throw std::runtime_error("unaligned quantized row");
    switch (type) {
        case F32: return n * 4;
        case F16: case BF16: return n * 2;
        case Q2_K: return n / 256 * 84;
        case IQ2_XXS: return n / 256 * 66;
        case IQ2_XS: return n / 256 * 74;
        case IQ2_S: return n / 256 * 82;
        case IQ3_XXS: return n / 256 * 98;
        case IQ3_S: return n / 256 * 110;
        case IQ4_XS: return n / 256 * 136;
        case IQ1_M: return n / 256 * 56;
        case Q8_0: return n / 32 * 34;
        case Q4_K: return n / 256 * 144;
        case Q5_K: return n / 256 * 176;
        case Q6_K: return n / 256 * 210;
        default: throw std::runtime_error("unsupported ggml type " + std::to_string(type));
    }
}

inline std::vector<float> float_values(const Tensor & t) {
    if (t.type != F32 && t.type != BF16 && t.type != F16) throw std::runtime_error(t.name + ": expected floating point weights");
    std::vector<float> out(t.nbytes / (t.type == F32 ? 4 : 2));
    if (t.type == F32) { memcpy(out.data(), t.data, t.nbytes); return out; }
    for (size_t i = 0; i < out.size(); i++) {
        uint16_t h; memcpy(&h, t.data + i * 2, 2);
        uint32_t bits = uint32_t(h) << 16;
        if (t.type == F16) {
            const uint32_t sign = uint32_t(h & 0x8000) << 16;
            int exp = (h >> 10) & 31;
            uint32_t frac = h & 1023;
            if (!exp && frac) { exp = 1; while (!(frac & 1024)) { frac <<= 1; exp--; } frac &= 1023; }
            bits = sign | (exp == 31 ? 0x7f800000u : (exp || frac ? uint32_t(exp + 112) << 23 : 0)) | (frac << 13);
        }
        memcpy(&out[i], &bits, 4);
    }
    return out;
}

struct File {
    const uint8_t * base = nullptr;
    size_t size = 0;
    std::map<std::string, Tensor> tensors;
    std::map<std::string, uint64_t> ints;
    std::map<std::string, std::string> strs;
    std::map<std::string, double> reals;
    std::map<std::string, std::vector<uint64_t>> arrays;
    std::map<std::string, uint64_t> array_sizes;

    explicit File(const std::string & path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 24) { close(fd); throw std::runtime_error("invalid GGUF size"); }
        size = (size_t) st.st_size;
        void * p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
        base = (const uint8_t *) p;
        try { parse(); } catch (...) { munmap((void *) base, size); base = nullptr; throw; }
    }
    ~File() { if (base) munmap((void *) base, size); }

    const Tensor & get(const std::string & name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("missing tensor " + name);
        return it->second;
    }

  private:
    size_t pos = 0;
    template <class T> T rd() { if (pos > size || sizeof(T) > size - pos) throw std::runtime_error("truncated GGUF"); T v; memcpy(&v, base + pos, sizeof(T)); pos += sizeof(T); return v; }
    std::string rds() { uint64_t n = rd<uint64_t>(); if (pos > size || n > size - pos) throw std::runtime_error("truncated GGUF string"); std::string s((const char *) base + pos, n); pos += n; return s; }

    void skip_val(uint32_t t, const std::string & key, bool top) {
        switch (t) {
            case 0: case 1: case 7: { uint8_t v = rd<uint8_t>(); if (top) ints[key] = v; break; }
            case 2: { const auto v = rd<uint16_t>(); if (top) ints[key] = v; break; }
            case 3: { const auto v = rd<int16_t>(); if (top) ints[key] = v; break; }
            case 4: { uint32_t v = rd<uint32_t>(); if (top) ints[key] = v; break; }
            case 5: { int32_t v = rd<int32_t>(); if (top) ints[key] = (uint64_t) v; break; }
            case 6: { const float v = rd<float>(); if (top) reals[key] = v; break; }
            case 8: { std::string s = rds(); if (top) strs[key] = s; break; }
            case 9: {
                uint32_t et = rd<uint32_t>();
                uint64_t n = rd<uint64_t>();
                if (top) array_sizes[key] = n;
                if (n > size || et == 9) throw std::runtime_error("invalid GGUF array");
                for (uint64_t i = 0; i < n; i++) {
                    if (top && et == 4) arrays[key].push_back(rd<uint32_t>());
                    else if (top && et == 5) arrays[key].push_back(rd<int32_t>());
                    else if (top && et == 7) arrays[key].push_back(rd<uint8_t>());
                    else skip_val(et, key, false);
                }
                break;
            }
            case 10: { uint64_t v = rd<uint64_t>(); if (top) ints[key] = v; break; }
            case 11: { int64_t v = rd<int64_t>(); if (top) ints[key] = (uint64_t) v; break; }
            case 12: { const double v = rd<double>(); if (top) reals[key] = v; break; }
            default: throw std::runtime_error("bad gguf kv type");
        }
    }

    void parse() {
        if (memcmp(base, "GGUF", 4) != 0) throw std::runtime_error("not a gguf file");
        pos = 4;
        const uint32_t version = rd<uint32_t>();
        if (version != 2 && version != 3) throw std::runtime_error("unsupported GGUF version");
        uint64_t n_tensors = rd<uint64_t>();
        uint64_t n_kv = rd<uint64_t>();
        if (n_tensors > 1000000 || n_kv > 1000000) throw std::runtime_error("invalid GGUF counts");
        for (uint64_t i = 0; i < n_kv; i++) {
            std::string key = rds();
            uint32_t t = rd<uint32_t>();
            skip_val(t, key, true);
        }
        std::vector<Tensor> infos;
        for (uint64_t i = 0; i < n_tensors; i++) {
            Tensor t;
            t.name = rds();
            uint32_t nd = rd<uint32_t>();
            if (!nd || nd > 4) throw std::runtime_error("invalid tensor rank");
            for (uint32_t d = 0; d < nd; d++) t.ne.push_back(rd<uint64_t>());
            t.type = rd<uint32_t>();
            t.offset = rd<uint64_t>();
            infos.push_back(t);
        }
        uint64_t align = ints.count("general.alignment") ? ints["general.alignment"] : 32;
        if (!align || (align & (align - 1)) || align > size) throw std::runtime_error("invalid GGUF alignment");
        size_t data_start = (pos + align - 1) / align * align;
        for (auto & t : infos) {
            if (data_start > size || t.offset > size - data_start) throw std::runtime_error("invalid tensor offset");
            t.data = base + data_start + t.offset;
            uint64_t rows = 1;
            for (size_t d = 1; d < t.ne.size(); d++) {
                if (!t.ne[d] || rows > std::numeric_limits<uint64_t>::max() / t.ne[d]) throw std::runtime_error("invalid tensor dimensions");
                rows *= t.ne[d];
            }
            const uint64_t rb = type_row_bytes(t.type, t.ne[0]);
            if (!rb || rows > (size - data_start - t.offset) / rb) throw std::runtime_error("truncated tensor data");
            t.nbytes = rb * rows;
            if (tensors.count(t.name)) throw std::runtime_error("duplicate tensor name");
            tensors[t.name] = t;
        }
    }
};

} // namespace gguf
