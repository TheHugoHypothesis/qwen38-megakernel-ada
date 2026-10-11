#pragma once
#include "native_iq_mma.cuh"

// Eight producer warps decode coalesced K128 slices for eight 16-row tiles.
// Eight consumer warps execute IMMA16x8 on the previous slice. Both stages fit
// below SM_MISC=87040 in the existing target shared-memory allocation.
constexpr int IQ_PIPE_TILES = 8, IQ_PIPE_TILE_BYTES = 5120;
constexpr int IQ_PIPE_WEIGHTS = 2 * IQ_PIPE_TILES * IQ_PIPE_TILE_BYTES;
constexpr int IQ_PIPE_INPUT_BYTES = 1440; // 8 tokens x 5 units x (int4 + int4 + half2)
constexpr int IQ_PIPE_BYTES = IQ_PIPE_WEIGHTS + 2 * IQ_PIPE_INPUT_BYTES;

template<int TYPE>
__device__ __forceinline__ Quant16 iq_pipeline_fragment(const uint8_t * row, int e) {
    if constexpr (TYPE == 18 || TYPE == 21 || TYPE == 23) return quant16<TYPE>(row, e);
    else {
        static_assert(TYPE == 16 || TYPE == 17 || TYPE == 22, "offset-free IQ fragment");
        const Quant8 a = quant8(TYPE, row, e), b = quant8(TYPE, row, e + 8);
        return {make_int4(a.q.x, a.q.y, b.q.x, b.q.y), a.d};
    }
}

template<int GT, int UT>
__device__ __forceinline__ void iq_pipeline_produce(
        const uint8_t * gate, int gs, const uint8_t * up, int us,
        int first_row, int rows, int units, int slice, int stage,
        const int4 * xa, const int4 * xb, const uint2 * meta, uint8_t * sm) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp < 8) return;
    if (warp == 8) {
        uint8_t * input = sm + IQ_PIPE_WEIGHTS + stage * IQ_PIPE_INPUT_BYTES;
        auto * a = reinterpret_cast<int4 *>(input); auto * b = a + 40;
        auto * m = reinterpret_cast<unsigned *>(b + 40);
        const int token = lane / 4, unit = lane & 3;
        const int src = token * units + 4 * slice + unit, dst = token * 5 + unit;
        a[dst] = xa[src]; b[dst] = xb[src]; m[dst] = meta[src].x;
    }
    const int tile = warp - 8;
    if (first_row + 16 * tile >= rows) return;
    uint8_t * weights = sm + stage * IQ_PIPE_TILES * IQ_PIPE_TILE_BYTES + tile * IQ_PIPE_TILE_BYTES;
    auto * g = reinterpret_cast<int4 *>(weights); auto * u = g + 128;
    auto * gd = reinterpret_cast<float *>(u + 128); auto * ud = gd + 128;
    const int chunk = lane & 7, e = 128 * slice + 16 * chunk;
#pragma unroll 1
    for (int i = 0; i < 4; i++) {
        const int r = 4 * i + (lane >> 3), row = first_row + 16 * tile + r, dst = 16 * chunk + r;
        const Quant16 qg = iq_pipeline_fragment<GT>(gate + size_t(row) * gs, e);
        g[dst] = qg.q; gd[dst] = qg.d;
        const Quant16 qu = iq_pipeline_fragment<UT>(up + size_t(row) * us, e);
        u[dst] = qu.q; ud[dst] = qu.d;
    }
}

__device__ __forceinline__ void iq_pipeline_consume(int first_row, int rows, int stage, uint8_t * sm,
                                                  float (&gate)[4], float (&up)[4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= 8 || first_row + 16 * warp >= rows) return;
    const int row = lane >> 2, word = lane & 3;
    const uint8_t * weights = sm + stage * IQ_PIPE_TILES * IQ_PIPE_TILE_BYTES + warp * IQ_PIPE_TILE_BYTES;
    const auto * g = reinterpret_cast<const int *>(weights);
    const auto * u = g + 512;
    const auto * gd = reinterpret_cast<const float *>(u + 512); const auto * ud = gd + 128;
    const uint8_t * input = sm + IQ_PIPE_WEIGHTS + stage * IQ_PIPE_INPUT_BYTES;
    const auto * a = reinterpret_cast<const int *>(input);
    const auto * b = a + 160;
    const auto * m = reinterpret_cast<const unsigned *>(b + 160);
#pragma unroll 1
    for (int chunk = 0; chunk < 8; chunk++) {
        const int unit = 2 * (chunk / 4) + (chunk & 1), part = (chunk & 3) / 2;
        const int packed_x = (part ? b : a)[4 * (row * 5 + unit) + word];
        const int ix0 = 4 * (16 * chunk + row) + word, ix1 = ix0 + 32;
        const int4 dg = iq_mma16x8(g[ix0], g[ix1], packed_x);
        const int4 du = iq_mma16x8(u[ix0], u[ix1], packed_x);
        const float d0 = __half2float(reinterpret_cast<const __half *>(m + 2 * word * 5 + unit)[part]);
        const float d1 = __half2float(reinterpret_cast<const __half *>(m + (2 * word + 1) * 5 + unit)[part]);
        const float g0 = gd[16 * chunk + row], g1 = gd[16 * chunk + row + 8];
        const float u0 = ud[16 * chunk + row], u1 = ud[16 * chunk + row + 8];
        gate[0] += d0 * g0 * dg.x; gate[1] += d1 * g0 * dg.y;
        gate[2] += d0 * g1 * dg.z; gate[3] += d1 * g1 * dg.w;
        up[0] += d0 * u0 * du.x; up[1] += d1 * u0 * du.y;
        up[2] += d0 * u1 * du.z; up[3] += d1 * u1 * du.w;
    }
}

template<int GT, int UT>
__device__ __forceinline__ void iq_pipeline_dual(
        const uint8_t * gate, int gs, const uint8_t * up, int us, int first_row, int rows, int units,
        const int4 * xa, const int4 * xb, const uint2 * meta, uint8_t * sm,
        float (&g)[4], float (&u)[4]) {
#pragma unroll
    for (int i = 0; i < 4; i++) g[i] = u[i] = 0.f;
    const int slices = units / 4;
    iq_pipeline_produce<GT, UT>(gate, gs, up, us, first_row, rows, units, 0, 0, xa, xb, meta, sm);
    __syncthreads();
#pragma unroll 1
    for (int k = 0; k < slices; k++) {
        if ((threadIdx.x >> 5) >= 8) {
            if (k + 1 < slices) iq_pipeline_produce<GT, UT>(gate, gs, up, us, first_row, rows, units, k + 1, (k + 1) & 1, xa, xb, meta, sm);
        } else {
            iq_pipeline_consume(first_row, rows, k & 1, sm, g, u);
        }
        __syncthreads();
    }
}
