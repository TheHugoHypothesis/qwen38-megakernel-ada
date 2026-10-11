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
