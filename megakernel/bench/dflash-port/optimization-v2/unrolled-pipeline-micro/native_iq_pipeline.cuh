#pragma once
#include "native_iq_mma.cuh"

// Eight producer warps decode coalesced K128 slices for eight 16-row tiles.
// Eight consumer warps execute IMMA16x8 on the previous slice. Both stages fit
// within the existing 92544-byte target shared-memory allocation. The padded
// K-major tile avoids bank conflicts when producers store coalesced K fragments.
// An integrated caller must preserve RMS scales beyond IQ_PIPE_BYTES, since the
// old SM_MISC region is reused by the staged input.
constexpr int IQ_PIPE_TILES = 8, IQ_PIPE_ROW_STRIDE = 17, IQ_PIPE_TILE_BYTES = 5440;
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

__device__ __forceinline__ void iq_pipeline_input(int units, int slice, const int4 * xa, const int4 * xb,
                                                 const uint2 * meta, uint8_t * input) {
    const int lane=threadIdx.x&31;
    auto * a=reinterpret_cast<int4 *>(input);auto * b=a+40;
    auto * m=reinterpret_cast<unsigned *>(b+40);
    const int token=lane/4,unit=lane&3,src=token*units+4*slice+unit,dst=token*5+unit;
    a[dst]=xa[src];b[dst]=xb[src];m[dst]=meta[src].x;
}

template<int GT,int UT>
__device__ __forceinline__ void iq_pipeline_weights(const uint8_t * gate,int gs,const uint8_t * up,int us,
                                                   int first_row,int slice,uint8_t * weights) {
    const int lane=threadIdx.x&31;
    auto * g = reinterpret_cast<int4 *>(weights); auto * u = g + 8*IQ_PIPE_ROW_STRIDE;
    auto * gd = reinterpret_cast<float *>(u + 8*IQ_PIPE_ROW_STRIDE); auto * ud = gd + 8*IQ_PIPE_ROW_STRIDE;
    const int chunk = lane & 7, e = 128 * slice + 16 * chunk;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = 4 * i + (lane >> 3), row = first_row + r, dst = IQ_PIPE_ROW_STRIDE * chunk + r;
        const Quant16 qg = iq_pipeline_fragment<GT>(gate + size_t(row) * gs, e);
        const Quant16 qu = iq_pipeline_fragment<UT>(up + size_t(row) * us, e);
        g[dst] = qg.q; gd[dst] = qg.d;
        u[dst] = qu.q; ud[dst] = qu.d;
    }
}

template<int GT, int UT>
__device__ __forceinline__ void iq_pipeline_produce(
        const uint8_t * gate, int gs, const uint8_t * up, int us,
        int first_row, int rows, int units, int slice, int stage,
        const int4 * xa, const int4 * xb, const uint2 * meta, uint8_t * sm) {
    const int warp=threadIdx.x>>5;
    if (warp<8) return;
    if (warp==8) iq_pipeline_input(units,slice,xa,xb,meta,sm+IQ_PIPE_WEIGHTS+stage*IQ_PIPE_INPUT_BYTES);
    const int tile=warp-8;
    if (first_row+16*tile>=rows) return;
    iq_pipeline_weights<GT,UT>(gate,gs,up,us,first_row+16*tile,slice,
        sm+stage*IQ_PIPE_TILES*IQ_PIPE_TILE_BYTES+tile*IQ_PIPE_TILE_BYTES);
}

__device__ __forceinline__ int4 iq_mma32x8(int a0,int a1,int a2,int a3,int b0,int b1) {
    int4 d;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d.x),"=r"(d.y),"=r"(d.z),"=r"(d.w)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0),"r"(0),"r"(0),"r"(0));
    return d;
}

template<bool K32 = false>
__device__ __forceinline__ void iq_pipeline_consume_view(const uint8_t * weights,const uint8_t * input,
                                                       float (&gate)[4],float (&up)[4]) {
    const int lane=threadIdx.x&31;
    const int row = lane >> 2, word = lane & 3;
    const auto * g = reinterpret_cast<const int *>(weights);
    const auto * u = g + 32*IQ_PIPE_ROW_STRIDE;
    const auto * gd = reinterpret_cast<const float *>(u + 32*IQ_PIPE_ROW_STRIDE); const auto * ud = gd + 8*IQ_PIPE_ROW_STRIDE;
    const auto * a = reinterpret_cast<const int *>(input);
    const auto * b = a + 160;
    const auto * m = reinterpret_cast<const unsigned *>(b + 160);
#pragma unroll
    for (int chunk = 0; chunk < 8; chunk += K32 ? 2 : 1) {
        const int unit = 2 * (chunk / 4) + (chunk & 1), part = (chunk & 3) / 2;
        const int packed_x = (part ? b : a)[4 * (row * 5 + unit) + word];
        const int ix0 = 4 * (IQ_PIPE_ROW_STRIDE * chunk + row) + word, ix1 = ix0 + 32;
        int4 dg,du;
        if constexpr (K32) {
            const int x1=(part?b:a)[4*(row*5+unit+1)+word];
            dg=iq_mma32x8(g[ix0],g[ix1],g[ix0+4*IQ_PIPE_ROW_STRIDE],g[ix1+4*IQ_PIPE_ROW_STRIDE],packed_x,x1);
            du=iq_mma32x8(u[ix0],u[ix1],u[ix0+4*IQ_PIPE_ROW_STRIDE],u[ix1+4*IQ_PIPE_ROW_STRIDE],packed_x,x1);
        } else {
            dg=iq_mma16x8(g[ix0],g[ix1],packed_x);
            du=iq_mma16x8(u[ix0],u[ix1],packed_x);
        }
        const float d0 = __half2float(reinterpret_cast<const __half *>(m + 2 * word * 5 + unit)[part]);
        const float d1 = __half2float(reinterpret_cast<const __half *>(m + (2 * word + 1) * 5 + unit)[part]);
        const float g0 = gd[IQ_PIPE_ROW_STRIDE * chunk + row], g1 = gd[IQ_PIPE_ROW_STRIDE * chunk + row + 8];
        const float u0 = ud[IQ_PIPE_ROW_STRIDE * chunk + row], u1 = ud[IQ_PIPE_ROW_STRIDE * chunk + row + 8];
        gate[0] += d0 * g0 * dg.x; gate[1] += d1 * g0 * dg.y;
        gate[2] += d0 * g1 * dg.z; gate[3] += d1 * g1 * dg.w;
        up[0] += d0 * u0 * du.x; up[1] += d1 * u0 * du.y;
        up[2] += d0 * u1 * du.z; up[3] += d1 * u1 * du.w;
    }
}

__device__ __forceinline__ void iq_pipeline_consume(int first_row,int rows,int stage,uint8_t * sm,
                                                   float (&g)[4],float (&u)[4]) {
    const int warp=threadIdx.x>>5;
    if (warp>=8 || first_row+16*warp>=rows) return;
    iq_pipeline_consume_view(sm+stage*IQ_PIPE_TILES*IQ_PIPE_TILE_BYTES+warp*IQ_PIPE_TILE_BYTES,
                            sm+IQ_PIPE_WEIGHTS+stage*IQ_PIPE_INPUT_BYTES,g,u);
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

// Every warp decodes and consumes its own tile. This doubles the number of
// simultaneous row tiles and halves CTA barriers versus split producer warps.
// Weight staging uses warp barriers; the CTA barrier advances double-buffered X.
template<int GT,int UT,bool K32 = false>
__device__ __forceinline__ void iq_warp_pipeline_dual(
        const uint8_t * gate,int gs,const uint8_t * up,int us,int first_row,int rows,int units,
        const int4 * xa,const int4 * xb,const uint2 * meta,uint8_t * sm,
        float (&g)[4],float (&u)[4]) {
    static_assert(!K32 || ((GT==16||GT==18||GT==21||GT==23) && (UT==16||UT==18||UT==21||UT==23)),"K32 requires weight scales uniform over 32");
    const int warp=threadIdx.x>>5;
#pragma unroll
    for (int i=0;i<4;i++) g[i]=u[i]=0.f;
    if (warp==0) iq_pipeline_input(units,0,xa,xb,meta,sm+IQ_PIPE_WEIGHTS);
    __syncthreads();
    for (int k=0;k<units/4;k++) {
        if (warp==0 && k+1<units/4)
            iq_pipeline_input(units,k+1,xa,xb,meta,sm+IQ_PIPE_WEIGHTS+((k+1)&1)*IQ_PIPE_INPUT_BYTES);
        if (first_row+16*warp<rows) {
            uint8_t * weights=sm+warp*IQ_PIPE_TILE_BYTES;
            iq_pipeline_weights<GT,UT>(gate,gs,up,us,first_row+16*warp,k,weights);
            __syncwarp();
            iq_pipeline_consume_view<K32>(weights,sm+IQ_PIPE_WEIGHTS+(k&1)*IQ_PIPE_INPUT_BYTES,g,u);
        }
        __syncthreads();
    }
}
