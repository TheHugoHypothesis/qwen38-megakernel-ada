// GGUF IQ decoding follows llama.cpp/ggml/src/ggml-cuda/dequantize.cuh (MIT).
// Copyright notice and MIT permission: LICENSE.llama-cpp.
// Each fragment keeps integer grid values and its scale/offset for dp4a reuse.
#pragma once
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include <type_traits>

// Optional Ada experiment: leave L1 available for codebooks while streaming weights.
// The default preserves the established load policy until integrated A/B checks.
template<class T> __device__ __forceinline__ T native_weight_load(const T * p) {
#if defined(MK_IQ_WEIGHT_NOL1) && MK_IQ_WEIGHT_NOL1
    unsigned v;
    if constexpr(sizeof(T)==1)asm("ld.global.nc.L1::no_allocate.u8 %0, [%1];" : "=r"(v) : "l"(p));
    else if constexpr(sizeof(T)==2)asm("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=r"(v) : "l"(p));
    else {static_assert(sizeof(T)==4,"weight load width");asm("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));}
    if constexpr(std::is_same<T,__half>::value)return __ushort_as_half(static_cast<unsigned short>(v));
    else if constexpr(std::is_same<T,__half2>::value)return __halves2half2(__ushort_as_half(v&65535),__ushort_as_half(v>>16));
    else if constexpr(std::is_same<T,float>::value)return __uint_as_float(v);
    else return static_cast<T>(v);
#else
    return __ldg(p);
#endif
}

struct Quant8 {
    int2 q;
    float d, offset;
};

struct Quant16 {
    int4 q;
    float d;
};

struct Quant4 {
    int q;
    float d;
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

// IQ block headers are only two-byte aligned in IQ3_S and IQ3_XXS.
template <int TYPE>
__device__ __forceinline__ Quant16 quant16(const uint8_t * row, int e) {
    static_assert(TYPE == 18 || TYPE == 21 || TYPE == 23, "IQ16 format");
    const int j = (e & 255) / 32, half = (e & 31) / 16;
    if constexpr (TYPE == 23) {
        const auto & x = reinterpret_cast<const block_iq4_xs *>(row)[e / 256];
        const int scale = ((native_weight_load(x.scales_l + j/2) >> (4*(j%2))) & 15)
                        | (((native_weight_load(&x.scales_h) >> (2*j)) & 3) << 4);
        const float d = __half2float(native_weight_load(&x.d)) * (scale - 32);
        const auto * q = reinterpret_cast<const unsigned *>(x.qs + 16*j);
        const int shift = 4*half;
        return {make_int4(iq4_lookup((native_weight_load(q+0)>>shift)&0x0f0f0f0f),
                          iq4_lookup((native_weight_load(q+1)>>shift)&0x0f0f0f0f),
                          iq4_lookup((native_weight_load(q+2)>>shift)&0x0f0f0f0f),
                          iq4_lookup((native_weight_load(q+3)>>shift)&0x0f0f0f0f)), d};
    } else {
        const uint16_t * qs;
        unsigned signs0, signs1, high = 0;
        float d;
        if constexpr (TYPE == 18) {
            const auto & x = reinterpret_cast<const block_iq3_xxs *>(row)[e/256];
            qs = reinterpret_cast<const uint16_t *>(x.qs + 8*j + 4*half);
            const auto * gas = reinterpret_cast<const uint16_t *>(x.qs + 64) + 2*j;
            const unsigned aux = unsigned(native_weight_load(gas)) | (unsigned(native_weight_load(gas+1)) << 16);
            signs0 = __ldg(ksigns_iq2xs + ((aux >> (14*half)) & 127));
            signs1 = __ldg(ksigns_iq2xs + ((aux >> (14*half+7)) & 127));
            d = __half2float(native_weight_load(&x.d)) * (0.5f + (aux >> 28)) * 0.5f;
        } else {
            const auto & x = reinterpret_cast<const block_iq3_s *>(row)[e/256];
            qs = reinterpret_cast<const uint16_t *>(x.qs + 8*j + 4*half);
            const unsigned signs = native_weight_load(reinterpret_cast<const uint16_t *>(x.signs + 4*j + 2*half));
            signs0 = signs & 255; signs1 = signs >> 8;
            high = native_weight_load(x.qh + j) >> (4*half);
            d = __half2float(native_weight_load(&x.d)) * (1 + 2*((native_weight_load(x.scales + j/2) >> (4*(j%2))) & 15));
        }
        const unsigned a = native_weight_load(qs), b = native_weight_load(qs+1);
        unsigned g0, g1, g2, g3;
        if constexpr (TYPE == 18) {
            g0 = __ldg(iq3xxs_grid + (a & 255)); g1 = __ldg(iq3xxs_grid + (a >> 8));
            g2 = __ldg(iq3xxs_grid + (b & 255)); g3 = __ldg(iq3xxs_grid + (b >> 8));
        } else {
            g0 = __ldg(iq3s_grid + ((a & 255) | ((high << 8) & 256)));
            g1 = __ldg(iq3s_grid + ((a >> 8) | ((high << 7) & 256)));
            g2 = __ldg(iq3s_grid + ((b & 255) | ((high << 6) & 256)));
            g3 = __ldg(iq3s_grid + ((b >> 8) | ((high << 5) & 256)));
        }
        const int2 lo = signed_grid(g0, g1, signs0), hi = signed_grid(g2, g3, signs1);
        return {make_int4(lo.x, lo.y, hi.x, hi.y), d};
    }
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
            const uint32_t aux = unsigned(native_weight_load(q+2)) | (unsigned(native_weight_load(q+3)) << 16);
            grid = reinterpret_cast<const uint8_t *>(iq2xxs_grid + native_weight_load(reinterpret_cast<const uint8_t *>(q)+il));
            signs = __ldg(ksigns_iq2xs + ((aux >> (7 * il)) & 127));
            d = __half2float(native_weight_load(&x.d)) * (0.5f + (aux >> 28)) * 0.25f;
        } else if (type == 17) {
            const auto & x = reinterpret_cast<const block_iq2_xs *>(row)[e / 256];
            const unsigned q = native_weight_load(x.qs + (4 * j32 + il));
            grid = reinterpret_cast<const uint8_t *>(iq2xs_grid + (q & 511));
            signs = __ldg(ksigns_iq2xs + (q >> 9));
            d = __half2float(native_weight_load(&x.d)) * (0.5f + ((native_weight_load(x.scales + (j32)) >> (4 * (il / 2))) & 15)) * 0.25f;
        } else {
            const auto & x = reinterpret_cast<const block_iq2_s *>(row)[e / 256];
            grid = reinterpret_cast<const uint8_t *>(iq2s_grid + (native_weight_load(x.qs + (4 * j32 + il)) | ((native_weight_load(x.qh + (j32)) << (8 - 2 * il)) & 768)));
            signs = native_weight_load(x.qs + (32 + 4 * j32 + il));
            d = __half2float(native_weight_load(&x.d)) * (0.5f + ((native_weight_load(x.scales + (j32)) >> (4 * (il / 2))) & 15)) * 0.25f;
        }
        const auto * packed = reinterpret_cast<const unsigned *>(grid);
        return {signed_grid(__ldg(packed), __ldg(packed+1), signs), d, 0.f};
    } else if (type == 18 || type == 21) {
        const uint8_t * g0, * g1;
        unsigned signs;
        if (type == 18) {
            const auto & x = reinterpret_cast<const block_iq3_xxs *>(row)[e / 256];
            const uint16_t * gas = reinterpret_cast<const uint16_t *>(x.qs + 64) + 2 * j32;
            const unsigned aux = unsigned(native_weight_load(gas)) | (unsigned(native_weight_load(gas+1)) << 16);
            g0 = reinterpret_cast<const uint8_t *>(iq3xxs_grid + native_weight_load(x.qs + (8 * j32 + 2 * il)));
            g1 = reinterpret_cast<const uint8_t *>(iq3xxs_grid + native_weight_load(x.qs + (8 * j32 + 2 * il + 1)));
            signs = __ldg(ksigns_iq2xs + ((aux >> (7 * il)) & 127));
            d = __half2float(native_weight_load(&x.d)) * (0.5f + (aux >> 28)) * 0.5f;
        } else {
            const auto & x = reinterpret_cast<const block_iq3_s *>(row)[e / 256];
            g0 = reinterpret_cast<const uint8_t *>(iq3s_grid + (native_weight_load(x.qs + (8 * j32 + 2 * il)) | ((native_weight_load(x.qh + (j32)) << (8 - 2 * il)) & 256)));
            g1 = reinterpret_cast<const uint8_t *>(iq3s_grid + (native_weight_load(x.qs + (8 * j32 + 2 * il + 1)) | ((native_weight_load(x.qh + (j32)) << (7 - 2 * il)) & 256)));
            signs = native_weight_load(x.signs + (4 * j32 + il));
            d = __half2float(native_weight_load(&x.d)) * (1 + 2 * ((native_weight_load(x.scales + (j32 / 2)) >> (4 * (j32 % 2))) & 15));
        }
        return {signed_grid(__ldg(reinterpret_cast<const unsigned *>(g0)),
                            __ldg(reinterpret_cast<const unsigned *>(g1)), signs), d, 0.f};
    } else if (type == 23) {
        const auto & x = reinterpret_cast<const block_iq4_xs *>(row)[e / 256];
        d = __half2float(native_weight_load(&x.d)) * ((((native_weight_load(x.scales_l + (j32 / 2)) >> (4 * (j32 % 2))) & 15) | (((native_weight_load(&x.scales_h) >> (2 * j32)) & 3) << 4)) - 32);
        const unsigned * q = reinterpret_cast<const unsigned *>(x.qs + 16*j32 + (e & 15));
        const int shift = 4*((e & 31)/16);
        return {make_int2(iq4_lookup((native_weight_load(q) >> shift)&0x0f0f0f0f), iq4_lookup((native_weight_load(q+1) >> shift)&0x0f0f0f0f)), d, 0.f};
    } else if (type == 29) {
        const auto & x = reinterpret_cast<const block_iq1_m *>(row)[e / 256];
        const uint16_t * sc = reinterpret_cast<const uint16_t *>(x.scales);
        const uint16_t bits = (__ldg(sc+0) >> 12) | ((__ldg(sc+1) >> 8) & 240) | ((__ldg(sc+2) >> 4) & 3840) | (__ldg(sc+3) & 61440);
        const int ib16 = 2 * j32 + il / 2;
        d = __half2float(__ushort_as_half(bits)) * (2 * ((__ldg(sc+ib16/4) >> (3 * (ib16 % 4))) & 7) + 1);
        const unsigned h = native_weight_load(x.qh + (2 * j32 + il / 2)) >> (4 * (il % 2));
        const unsigned grid = __ldg(iq1s_grid_gpu + (native_weight_load(x.qs + (4 * j32 + il)) | ((h & 7) << 8)));
        offset = d * ((h & 8) ? -1.f - IQ1M_DELTA : -1.f + IQ1M_DELTA);
        #pragma unroll
        for (int j = 0; j < 8; j++) v[j] = (grid >> (8 * (j & 3) + 4 * (j / 4))) & 15;
    } else if (type == 10) {
        const auto & x = reinterpret_cast<const block_q2_K *>(row)[e / 256];
        const unsigned sc = native_weight_load(x.scales + ((e & 255) / 16));
        d = __half2float(__low2half(native_weight_load(&x.dm))) * (sc & 15);
        offset = -__half2float(__high2half(native_weight_load(&x.dm))) * (sc >> 4);
        const auto * q = reinterpret_cast<const unsigned *>(x.qs + 32 * ((e & 255) / 128) + (e & 31));
        const int shift = 2 * ((e & 127) / 32);
        return {make_int2((native_weight_load(q) >> shift) & 0x03030303, (native_weight_load(q + 1) >> shift) & 0x03030303), d, offset};
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

// Decode the common IQ2 header/scale once for two adjacent eight-value grids.
template<int TYPE>
__device__ __forceinline__ Quant16 iq2_direct16(const uint8_t * row,int e) {
    static_assert(TYPE==16||TYPE==17||TYPE==22,"IQ2 format");
    const int j=(e&255)/32,il=(e&31)/8;
    const uint8_t * grids[2];unsigned signs[2];float d;
    if constexpr(TYPE==16){
        const auto & x=reinterpret_cast<const block_iq2_xxs *>(row)[e/256];const uint16_t * q=x.qs+4*j;
        const unsigned aux=unsigned(native_weight_load(q+2))|(unsigned(native_weight_load(q+3))<<16);
        d=__half2float(native_weight_load(&x.d))*(.5f+(aux>>28))*.25f;
        #pragma unroll
        for(int h=0;h<2;h++){grids[h]=reinterpret_cast<const uint8_t *>(iq2xxs_grid+native_weight_load(reinterpret_cast<const uint8_t *>(q)+il+h));signs[h]=__ldg(ksigns_iq2xs+((aux>>(7*(il+h)))&127));}
    }else if constexpr(TYPE==17){
        const auto & x=reinterpret_cast<const block_iq2_xs *>(row)[e/256];
        d=__half2float(native_weight_load(&x.d))*(.5f+((native_weight_load(x.scales+j)>>(4*(il/2)))&15))*.25f;
        #pragma unroll
        for(int h=0;h<2;h++){const unsigned q=native_weight_load(x.qs+4*j+il+h);grids[h]=reinterpret_cast<const uint8_t *>(iq2xs_grid+(q&511));signs[h]=__ldg(ksigns_iq2xs+(q>>9));}
    }else{
        const auto & x=reinterpret_cast<const block_iq2_s *>(row)[e/256];const unsigned high=native_weight_load(x.qh+j);
        d=__half2float(native_weight_load(&x.d))*(.5f+((native_weight_load(x.scales+j)>>(4*(il/2)))&15))*.25f;
        #pragma unroll
        for(int h=0;h<2;h++){grids[h]=reinterpret_cast<const uint8_t *>(iq2s_grid+(native_weight_load(x.qs+4*j+il+h)|((high<<(8-2*(il+h)))&768)));signs[h]=native_weight_load(x.qs+32+4*j+il+h);}
    }
    int2 q[2];
    #pragma unroll
    for(int h=0;h<2;h++){auto p=reinterpret_cast<const unsigned *>(grids[h]);q[h]=signed_grid(__ldg(p),__ldg(p+1),signs[h]);}
    return {make_int4(q[0].x,q[0].y,q[1].x,q[1].y),d};
}

// Six offset-free IQ formats have one scale for each 16-value fragment.
// IQ2 joins two original Quant8 payloads; its scale is shared by both halves.
template<int TYPE>
__device__ __forceinline__ Quant16 native_iq16(const uint8_t * row, int e) {
    if constexpr (TYPE == 18 || TYPE == 21 || TYPE == 23) return quant16<TYPE>(row, e);
    else {
        static_assert(TYPE == 16 || TYPE == 17 || TYPE == 22, "offset-free IQ16 format");
#if defined(MK_IQ2_DIRECT) && MK_IQ2_DIRECT
        return iq2_direct16<TYPE>(row,e);
#else
        const Quant8 a = quant8(TYPE, row, e), b = quant8(TYPE, row, e + 8);
        return {make_int4(a.q.x, a.q.y, b.q.x, b.q.y), a.d};
#endif
    }
}

// A Tensor Core lane consumes four signed bytes, not the full Quant8 payload.
// Keep the same codebook/sign/scale decoding and avoid fetching the discarded half.
template<int TYPE>
__device__ __forceinline__ Quant4 quant4(const uint8_t * row, int e) {
    static_assert(TYPE == 16 || TYPE == 17 || TYPE == 18 || TYPE == 21 || TYPE == 22 || TYPE == 23, "IQ4 fragment format");
    const int j = (e & 255) / 32, il = (e & 31) / 8, half = (e & 7) / 4;
    unsigned grid = 0, signs = 0; float d = 0.f;
    if constexpr (TYPE == 16 || TYPE == 17 || TYPE == 22) {
        const uint8_t * table;
        if constexpr (TYPE == 16) {
            const auto & x = reinterpret_cast<const block_iq2_xxs *>(row)[e / 256];
            const uint16_t * q = x.qs + 4 * j;
            const unsigned aux = unsigned(native_weight_load(q + 2)) | (unsigned(native_weight_load(q + 3)) << 16);
            table = reinterpret_cast<const uint8_t *>(iq2xxs_grid + native_weight_load(reinterpret_cast<const uint8_t *>(q) + il));
            signs = __ldg(ksigns_iq2xs + ((aux >> (7 * il)) & 127));
            d = __half2float(native_weight_load(&x.d)) * (0.5f + (aux >> 28)) * 0.25f;
        } else if constexpr (TYPE == 17) {
            const auto & x = reinterpret_cast<const block_iq2_xs *>(row)[e / 256];
            const unsigned q = native_weight_load(x.qs + 4 * j + il);
            table = reinterpret_cast<const uint8_t *>(iq2xs_grid + (q & 511));
            signs = __ldg(ksigns_iq2xs + (q >> 9));
            d = __half2float(native_weight_load(&x.d)) * (0.5f + ((native_weight_load(x.scales + j) >> (4 * (il / 2))) & 15)) * 0.25f;
        } else {
            const auto & x = reinterpret_cast<const block_iq2_s *>(row)[e / 256];
            table = reinterpret_cast<const uint8_t *>(iq2s_grid + (native_weight_load(x.qs + 4 * j + il) | ((native_weight_load(x.qh + j) << (8 - 2 * il)) & 768)));
            signs = native_weight_load(x.qs + 32 + 4 * j + il);
            d = __half2float(native_weight_load(&x.d)) * (0.5f + ((native_weight_load(x.scales + j) >> (4 * (il / 2))) & 15)) * 0.25f;
        }
        grid = __ldg(reinterpret_cast<const unsigned *>(table) + half);
    } else if constexpr (TYPE == 18) {
        const auto & x = reinterpret_cast<const block_iq3_xxs *>(row)[e / 256];
        const uint16_t * gas = reinterpret_cast<const uint16_t *>(x.qs + 64) + 2 * j;
        const unsigned aux = unsigned(native_weight_load(gas)) | (unsigned(native_weight_load(gas + 1)) << 16);
        grid = __ldg(iq3xxs_grid + native_weight_load(x.qs + 8 * j + (e & 31) / 4));
        signs = __ldg(ksigns_iq2xs + ((aux >> (7 * il)) & 127));
        d = __half2float(native_weight_load(&x.d)) * (0.5f + (aux >> 28)) * 0.5f;
    } else if constexpr (TYPE == 21) {
        const auto & x = reinterpret_cast<const block_iq3_s *>(row)[e / 256];
        const int code = (e & 31) / 4;
        grid = __ldg(iq3s_grid + (native_weight_load(x.qs + 8 * j + code) | ((native_weight_load(x.qh + j) << (8 - code)) & 256)));
        signs = native_weight_load(x.signs + 4 * j + il);
        d = __half2float(native_weight_load(&x.d)) * (1 + 2 * ((native_weight_load(x.scales + j / 2) >> (4 * (j % 2))) & 15));
    } else {
        const auto & x = reinterpret_cast<const block_iq4_xs *>(row)[e / 256];
        const int scale = ((native_weight_load(x.scales_l + j / 2) >> (4 * (j % 2))) & 15) | (((native_weight_load(&x.scales_h) >> (2 * j)) & 3) << 4);
        d = __half2float(native_weight_load(&x.d)) * (scale - 32);
        const auto * q = reinterpret_cast<const unsigned *>(x.qs + 16 * j + (e & 15));
        return {int(iq4_lookup((native_weight_load(q) >> (4 * ((e & 31) / 16))) & 0x0f0f0f0f)), d};
    }
    const unsigned spread = (signs >> (4 * half)) * 0x01010101u;
    const unsigned mask = __vcmpne4(spread & 0x08040201u, 0);
    return {int(__vsub4(grid ^ mask, mask)), d};
}

__device__ __forceinline__ float native_value(int type, const uint8_t * row, int e, const __half * ds = nullptr) {
    const Quant8 f = quant8(type, row, e & ~7, ds);
    const unsigned word = unsigned((e & 7) < 4 ? f.q.x : f.q.y);
    return f.d * int8_t(word >> (8 * (e & 3))) + f.offset;
}
