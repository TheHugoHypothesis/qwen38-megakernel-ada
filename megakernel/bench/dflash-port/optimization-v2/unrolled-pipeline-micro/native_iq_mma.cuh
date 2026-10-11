#pragma once
#include "native_quant.cuh"

// These IQ encodings have no additive offset and a scale uniform over 16 values.
__host__ __device__ __forceinline__ bool iq_mma_supported(int type) {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23;
}

__device__ __forceinline__ int2 iq_mma16(int a, int b) {
    int2 d;
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
                 : "=r"(d.x), "=r"(d.y) : "r"(a), "r"(b), "r"(0), "r"(0));
    return d;
}

// A warp owns eight output rows and eight tokens. A: row=lane/4, K=4*(lane%4).
// B: token=lane/4, K=4*(lane%4). C: row=lane/4, tokens=2*(lane%4)+{0,1}.
// Integer MMA covers exactly one Q16 activation scale, without requantization.
__device__ __forceinline__ void iq_mma_dual(
        const uint8_t * gate, int gate_stride, int gate_type,
        const uint8_t * up, int up_stride, int up_type,
        const int4 * xa, const int4 * xb, const uint2 * meta, int units,
        float (&g)[2], float (&u)[2]) {
    const int lane = threadIdx.x & 31, row = lane >> 2, word = lane & 3;
    gate += size_t(row) * gate_stride;
    up += size_t(row) * up_stride;
    g[0] = g[1] = u[0] = u[1] = 0.f;
#pragma unroll 1
    for (int e = 0; e < units * 32; e += 16) {
        const int unit = 8 * (e / 256) + 2 * ((e & 255) / 64) + ((e & 31) / 16);
        const int part = (e & 63) / 32;
        const Quant8 wg = quant8(gate_type, gate, e + 8 * (word / 2));
        const Quant8 wu = quant8(up_type, up, e + 8 * (word / 2));
        const int a = (word & 1) ? wg.q.y : wg.q.x;
        const int b = (word & 1) ? wu.q.y : wu.q.x;
        const int ix = row * units + unit;
        const int4 x = part ? xb[ix] : xa[ix];
        const int packed_x = word == 0 ? x.x : word == 1 ? x.y : word == 2 ? x.z : x.w;
        const int2 dg = iq_mma16(a, packed_x);
        const int2 du = iq_mma16(b, packed_x);
#pragma unroll
        for (int t = 0; t < 2; t++) {
            const int token = 2 * word + t;
            const float d = __half2float(reinterpret_cast<const __half *>(meta + token * units + unit)[part]);
            g[t] += d * wg.d * (t ? dg.y : dg.x);
            u[t] += d * wu.d * (t ? du.y : du.x);
        }
    }
}

// Full 16-row tile, one Q16 scale at a time. Compile-time IQ formats remove
// the generic decoder branches from the K loop; no padding/requantization.
__device__ __forceinline__ int4 iq_mma16x8(int a0, int a1, int b) {
    int4 d;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                 : "r"(a0), "r"(a1), "r"(b), "r"(0), "r"(0), "r"(0), "r"(0));
    return d;
}

template<int TYPE, bool NARROW>
__device__ __forceinline__ Quant4 iq_mma_fragment(const uint8_t * row, int e, int word) {
    if constexpr (NARROW) return quant4<TYPE>(row, e + 4 * word);
    else {
        const Quant8 q = quant8(TYPE, row, e + 8 * (word / 2));
        return {(word & 1) ? q.q.y : q.q.x, q.d};
    }
}

template<int GT, int UT, bool NARROW = false>
__device__ __forceinline__ void iq_mma16_dual(
        const uint8_t * gate, int gate_stride, const uint8_t * up, int up_stride,
        const int4 * xa, const int4 * xb, const uint2 * meta, int units,
        float (&g)[4], float (&u)[4]) {
    static_assert(GT == 16 || GT == 17 || GT == 18 || GT == 21 || GT == 22 || GT == 23, "gate IQ format");
    static_assert(UT == 16 || UT == 17 || UT == 18 || UT == 21 || UT == 22 || UT == 23, "up IQ format");
    const int lane = threadIdx.x & 31, row = lane >> 2, word = lane & 3;
    gate += size_t(row) * gate_stride;
    up += size_t(row) * up_stride;
#pragma unroll
    for (int i = 0; i < 4; i++) g[i] = u[i] = 0.f;
#pragma unroll 1
    for (int e = 0; e < units * 32; e += 16) {
        const int unit = 8 * (e / 256) + 2 * ((e & 255) / 64) + ((e & 31) / 16);
        const int part = (e & 63) / 32;
        const Quant4 wg0 = iq_mma_fragment<GT,NARROW>(gate,e,word), wg1 = iq_mma_fragment<GT,NARROW>(gate+size_t(8)*gate_stride,e,word);
        const Quant4 wu0 = iq_mma_fragment<UT,NARROW>(up,e,word), wu1 = iq_mma_fragment<UT,NARROW>(up+size_t(8)*up_stride,e,word);
        const int4 x = part ? xb[row * units + unit] : xa[row * units + unit];
        const int packed_x = word == 0 ? x.x : word == 1 ? x.y : word == 2 ? x.z : x.w;
        const int4 dg = iq_mma16x8(wg0.q, wg1.q, packed_x);
        const int4 du = iq_mma16x8(wu0.q, wu1.q, packed_x);
        const float d0 = __half2float(reinterpret_cast<const __half *>(meta + 2 * word * units + unit)[part]);
        const float d1 = __half2float(reinterpret_cast<const __half *>(meta + (2 * word + 1) * units + unit)[part]);
        g[0] += d0 * wg0.d * dg.x; g[1] += d1 * wg0.d * dg.y;
        g[2] += d0 * wg1.d * dg.z; g[3] += d1 * wg1.d * dg.w;
        u[0] += d0 * wu0.d * du.x; u[1] += d1 * wu0.d * du.y;
        u[2] += d0 * wu1.d * du.z; u[3] += d1 * wu1.d * du.w;
    }
}

// DP4A companion: the original IQ16 order for each dot, with separate gate/up
// encodings and one load of the activation per pair of weight fragments.
template<int T, int UNITS, int GT, int UT>
__device__ __forceinline__ void iq_dp4a_dual(
        const uint8_t * gate, const uint8_t * up,
        const int4 * xa, const int4 * xb, const uint2 * meta,
        float (&g)[T], float (&u)[T]) {
    static_assert(UNITS % 32 == 0, "full warp K tiles");
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < T; t++) g[t] = u[t] = 0.f;
#pragma unroll 1
    for (int i = 0; i < UNITS / 32; i++) {
        const int unit = lane + 32 * i;
        const int e0 = 256 * (unit / 8) + 64 * ((unit / 2) & 3) + 16 * (unit & 1);
#pragma unroll
        for (int part = 0; part < 2; part++) {
            const Quant16 wg = quant16<GT>(gate, e0 + 32 * part), wu = quant16<UT>(up, e0 + 32 * part);
#pragma unroll
            for (int t = 0; t < T; t++) {
                const int ix = t * UNITS + unit;
                const int4 x = part ? xb[ix] : xa[ix];
                const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(&meta[ix].x));
                const float d = part ? dx.y : dx.x;
                const int dg = __dp4a(wg.q.w, x.w, __dp4a(wg.q.z, x.z, __dp4a(wg.q.y, x.y, __dp4a(wg.q.x, x.x, 0))));
                const int du = __dp4a(wu.q.w, x.w, __dp4a(wu.q.z, x.z, __dp4a(wu.q.y, x.y, __dp4a(wu.q.x, x.x, 0))));
                g[t] += d * wg.d * dg; u[t] += d * wu.d * du;
            }
        }
    }
}
