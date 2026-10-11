// GGUF IQ decoding follows llama.cpp/ggml/src/ggml-cuda/dequantize.cuh (MIT).
// Copyright notice and MIT permission: LICENSE.llama-cpp.
// Each fragment keeps integer grid values and its scale/offset for dp4a reuse.
#pragma once
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

struct Quant8 {
    int2 q;
    float d, offset;
};

__device__ __forceinline__ int2 signed_grid(unsigned a, unsigned b, unsigned signs) {
    const unsigned spread = signs * 0x01010101u;
    const unsigned lo = __vcmpne4(spread & 0x08040201u, 0);
    const unsigned hi = __vcmpne4(spread & 0x80402010u, 0);
    return make_int2(__vsub4(a ^ lo, lo), __vsub4(b ^ hi, hi));
}

// IQ4_NL values in registers: no eight scattered byte loads per fragment.
__device__ __forceinline__ unsigned iq4_lookup(unsigned indices) {
    const unsigned select = (indices & 15) | ((indices >> 4) & 240)
                          | ((indices >> 8) & 3840) | ((indices >> 12) & 61440);
    const unsigned a = __byte_perm(0xbfad9881u, 0xf6eaddcfu, select & 0x7777);
    const unsigned b = __byte_perm(0x26190d01u, 0x71594535u, select & 0x7777);
    const unsigned mask = __vcmpne4(indices & 0x08080808, 0);
    return (a & ~mask) | (b & mask);
}

__device__ __forceinline__ Quant8 quant8(int type, const uint8_t * row, int e, const __half * ds = nullptr) {
    const int j32 = (e & 255) / 32, il = (e & 31) / 8;
    int v[8] = {};
    float d = 0.f, offset = 0.f;
    if (type == 16 || type == 17 || type == 22) {
        const uint8_t * grid;
        unsigned signs;
        if (type == 16) {
            const auto & x = reinterpret_cast<const block_iq2_xxs *>(row)[e / 256];
            const uint16_t * q = x.qs + 4 * j32;
            const uint32_t aux = unsigned(__ldg(q+2)) | (unsigned(__ldg(q+3)) << 16);
            grid = reinterpret_cast<const uint8_t *>(iq2xxs_grid + __ldg(reinterpret_cast<const uint8_t *>(q)+il));
            signs = __ldg(ksigns_iq2xs + ((aux >> (7 * il)) & 127));
            d = __half2float(__ldg(&x.d)) * (0.5f + (aux >> 28)) * 0.25f;
        } else if (type == 17) {
            const auto & x = reinterpret_cast<const block_iq2_xs *>(row)[e / 256];
            const unsigned q = __ldg(x.qs + (4 * j32 + il));
            grid = reinterpret_cast<const uint8_t *>(iq2xs_grid + (q & 511));
            signs = __ldg(ksigns_iq2xs + (q >> 9));
            d = __half2float(__ldg(&x.d)) * (0.5f + ((__ldg(x.scales + (j32)) >> (4 * (il / 2))) & 15)) * 0.25f;
        } else {
            const auto & x = reinterpret_cast<const block_iq2_s *>(row)[e / 256];
            grid = reinterpret_cast<const uint8_t *>(iq2s_grid + (__ldg(x.qs + (4 * j32 + il)) | ((__ldg(x.qh + (j32)) << (8 - 2 * il)) & 768)));
            signs = __ldg(x.qs + (32 + 4 * j32 + il));
            d = __half2float(__ldg(&x.d)) * (0.5f + ((__ldg(x.scales + (j32)) >> (4 * (il / 2))) & 15)) * 0.25f;
        }
        const auto * packed = reinterpret_cast<const unsigned *>(grid);
        return {signed_grid(__ldg(packed), __ldg(packed+1), signs), d, 0.f};
    } else if (type == 18 || type == 21) {
        const uint8_t * g0, * g1;
        unsigned signs;
        if (type == 18) {
            const auto & x = reinterpret_cast<const block_iq3_xxs *>(row)[e / 256];
            const uint16_t * gas = reinterpret_cast<const uint16_t *>(x.qs + 64) + 2 * j32;
            const unsigned aux = unsigned(__ldg(gas)) | (unsigned(__ldg(gas+1)) << 16);
            g0 = reinterpret_cast<const uint8_t *>(iq3xxs_grid + __ldg(x.qs + (8 * j32 + 2 * il)));
            g1 = reinterpret_cast<const uint8_t *>(iq3xxs_grid + __ldg(x.qs + (8 * j32 + 2 * il + 1)));
            signs = __ldg(ksigns_iq2xs + ((aux >> (7 * il)) & 127));
            d = __half2float(__ldg(&x.d)) * (0.5f + (aux >> 28)) * 0.5f;
        } else {
            const auto & x = reinterpret_cast<const block_iq3_s *>(row)[e / 256];
            g0 = reinterpret_cast<const uint8_t *>(iq3s_grid + (__ldg(x.qs + (8 * j32 + 2 * il)) | ((__ldg(x.qh + (j32)) << (8 - 2 * il)) & 256)));
            g1 = reinterpret_cast<const uint8_t *>(iq3s_grid + (__ldg(x.qs + (8 * j32 + 2 * il + 1)) | ((__ldg(x.qh + (j32)) << (7 - 2 * il)) & 256)));
            signs = __ldg(x.signs + (4 * j32 + il));
            d = __half2float(__ldg(&x.d)) * (1 + 2 * ((__ldg(x.scales + (j32 / 2)) >> (4 * (j32 % 2))) & 15));
        }
        return {signed_grid(__ldg(reinterpret_cast<const unsigned *>(g0)),
                            __ldg(reinterpret_cast<const unsigned *>(g1)), signs), d, 0.f};
    } else if (type == 23) {
        const auto & x = reinterpret_cast<const block_iq4_xs *>(row)[e / 256];
        d = __half2float(__ldg(&x.d)) * ((((__ldg(x.scales_l + (j32 / 2)) >> (4 * (j32 % 2))) & 15) | (((__ldg(&x.scales_h) >> (2 * j32)) & 3) << 4)) - 32);
        const unsigned * q = reinterpret_cast<const unsigned *>(x.qs + 16*j32 + (e & 15));
        const int shift = 4*((e & 31)/16);
        return {make_int2(iq4_lookup((__ldg(q) >> shift)&0x0f0f0f0f), iq4_lookup((__ldg(q+1) >> shift)&0x0f0f0f0f)), d, 0.f};
    } else if (type == 29) {
        const auto & x = reinterpret_cast<const block_iq1_m *>(row)[e / 256];
        const uint16_t * sc = reinterpret_cast<const uint16_t *>(x.scales);
        const uint16_t bits = (__ldg(sc+0) >> 12) | ((__ldg(sc+1) >> 8) & 240) | ((__ldg(sc+2) >> 4) & 3840) | (__ldg(sc+3) & 61440);
        const int ib16 = 2 * j32 + il / 2;
        d = __half2float(__ushort_as_half(bits)) * (2 * ((__ldg(sc+ib16/4) >> (3 * (ib16 % 4))) & 7) + 1);
        const unsigned h = __ldg(x.qh + (2 * j32 + il / 2)) >> (4 * (il % 2));
        const unsigned grid = __ldg(iq1s_grid_gpu + (__ldg(x.qs + (4 * j32 + il)) | ((h & 7) << 8)));
        offset = d * ((h & 8) ? -1.f - IQ1M_DELTA : -1.f + IQ1M_DELTA);
        #pragma unroll
        for (int j = 0; j < 8; j++) v[j] = (grid >> (8 * (j & 3) + 4 * (j / 4))) & 15;
    } else if (type == 10) {
        const auto & x = reinterpret_cast<const block_q2_K *>(row)[e / 256];
        const unsigned sc = __ldg(x.scales + ((e & 255) / 16));
        d = __half2float(__low2half(__ldg(&x.dm))) * (sc & 15);
        offset = -__half2float(__high2half(__ldg(&x.dm))) * (sc >> 4);
        #pragma unroll
        for (int j = 0; j < 8; j++) v[j] = (__ldg(x.qs + (32 * ((e & 255) / 128) + (e & 31) + j)) >> (2 * ((e & 127) / 32))) & 3;
    } else if (type == 12 || type == 13) {
        const uint8_t * x = row + (e / 256) * (type == 12 ? 144 : 176);
        const uint8_t * sc = x + 4;
        int s, m;
        if (j32 < 4) { s = sc[j32] & 63; m = sc[j32 + 4] & 63; }
        else { s = (sc[j32 + 4] & 15) | ((sc[j32 - 4] >> 6) << 4); m = (sc[j32 + 4] >> 4) | ((sc[j32] >> 6) << 4); }
        d = __half2float(*reinterpret_cast<const __half *>(x)) * s;
        offset = -__half2float(*reinterpret_cast<const __half *>(x + 2)) * m;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            v[j] = (x[(type == 12 ? 16 : 48) + 32 * (j32 / 2) + (e & 31) + j] >> (4 * (j32 & 1))) & 15;
            if (type == 13) v[j] |= ((x[16 + (e & 31) + j] >> j32) & 1) << 4;
        }
    } else if (type == 14) {
        const uint8_t * x = row + (e / 256) * 224;
        const int p = (e & 255) / 64, h = (e & 31) / 16, part = (e & 63) / 32;
        d = __half2float(*reinterpret_cast<const __half *>(x + 192 + 8 * p + 4)) * int8_t(x[192 + 8 * p + h + 2 * part]);
        const unsigned hi = *reinterpret_cast<const unsigned *>(x + 128 + 8 * (2 * p + h) + 4 * part);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int k = (e & 15) + j;
            v[j] = (((x[32 * p + 16 * h + k] >> (4 * part)) & 15) | (((hi >> (8 * (k % 4) + 2 * (k / 4))) & 3) << 4)) - 32;
        }
    } else if (type == 8) {
        const int u = 8 * (e / 256) + 2 * ((e & 255) / 64) + (e & 31) / 16;
        d = __half2float(ds[e / 32]);
        #pragma unroll
        for (int j = 0; j < 8; j++) v[j] = int8_t(row[32 * u + 16 * ((e & 63) / 32) + (e & 15) + j]);
    }
    unsigned a = 0, b = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) { a |= unsigned(uint8_t(v[j])) << (8 * j); b |= unsigned(uint8_t(v[j + 4])) << (8 * j); }
    return {make_int2(int(a), int(b)), d, offset};
}

__device__ __forceinline__ float native_value(int type, const uint8_t * row, int e, const __half * ds = nullptr) {
    const Quant8 f = quant8(type, row, e & ~7, ds);
    const unsigned word = unsigned((e & 7) < 4 ? f.q.x : f.q.y);
    return f.d * int8_t(word >> (8 * (e & 3))) + f.offset;
}
