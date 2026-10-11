#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Target KV formats: 0 FP16, 1 signed Q8, 2 packed symmetric signed Q4.
// Quantized scales live in separate arrays: one FP16 scale per 32 values.
// Q4 uses [-7,7], low nibble first, and is NOT the GGML Q4_0 block layout.
__host__ __device__ constexpr size_t kv_payload_bytes(int mode, size_t dims = 256) {
    return mode == 2 ? dims / 2 : mode == 1 ? dims : dims * 2;
}
__host__ __device__ constexpr size_t kv_cache_bytes(int mode, size_t rows, size_t dims = 256) {
    return 2 * rows * (kv_payload_bytes(mode, dims) + (mode ? dims / 32 * 2 : 0));
}

// Each warp stores one row; four adjacent lanes share a 32-value scale.
__device__ __forceinline__ void kv_store_q8(int8_t * q, __half * sc, const float (&v)[8]) {
    const int lane = threadIdx.x & 31;
    float am = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) am = fmaxf(am, fabsf(v[k]));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    const float id = am > 0.f ? 127.0f / am : 0.f;
    unsigned w[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        unsigned x = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) x |= ((unsigned) __float2int_rn(v[4 * h + k] * id) & 0xffu) << (8 * k);
        w[h] = x;
    }
    *reinterpret_cast<uint2 *>(q + lane * 8) = make_uint2(w[0], w[1]);
    if ((lane & 3) == 0) sc[lane >> 2] = __float2half_rn(am / 127.0f);
}

__device__ __forceinline__ void kv_store_q4(uint8_t * q, __half * sc, const float (&v)[8]) {
    const int lane = threadIdx.x & 31;
    float am = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) am = fmaxf(am, fabsf(v[k]));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    const float id = am > 0.f ? 7.0f / am : 0.f;
    unsigned packed = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const int vq = max(-7, min(7, __float2int_rn(v[k] * id)));
        packed |= (unsigned(vq) & 0xfu) << (4 * k);
    }
    *reinterpret_cast<unsigned *>(q + lane * 4) = packed;
    if ((lane & 3) == 0) sc[lane >> 2] = __float2half_rn(am / 7.0f);
}

__device__ __forceinline__ void kv_store_f16(__half * dst, const float (&v)[8]) {
    __half2 h[4];
#pragma unroll
    for (int k = 0; k < 4; k++) h[k] = __floats2half2_rn(v[2 * k], v[2 * k + 1]);
    *reinterpret_cast<int4 *>(dst + (threadIdx.x & 31) * 8) = *reinterpret_cast<int4 *>(h);
}

__device__ __forceinline__ void q8_to_f16(uint2 q, float d, __half2 (&h)[4]) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const unsigned x = k < 2 ? q.x : q.y;
        const int sh = 16 * (k & 1);
        h[k] = __floats2half2_rn(d * (float) (int8_t) (x >> sh), d * (float) (int8_t) (x >> (sh + 8)));
    }
}

__device__ __forceinline__ void q4_to_f16(unsigned q, float d, __half2 (&h)[4]) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int a = int(((q >> (8 * k)) & 15u) ^ 8u) - 8;
        const int b = int(((q >> (8 * k + 4)) & 15u) ^ 8u) - 8;
        h[k] = __floats2half2_rn(d * a, d * b);
    }
}

template <int MODE>
__device__ __forceinline__ void kv_store(void * cache, __half * scales, size_t row, const float (&v)[8]) {
    if constexpr (MODE == 2) kv_store_q4(static_cast<uint8_t *>(cache) + row * 128, scales + row * 8, v);
    else if constexpr (MODE == 1) kv_store_q8(static_cast<int8_t *>(cache) + row * 256, scales + row * 8, v);
    else kv_store_f16(static_cast<__half *>(cache) + row * 256, v);
}
