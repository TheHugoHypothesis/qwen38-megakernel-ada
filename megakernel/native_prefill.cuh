#pragma once

namespace pf {

// Direct register-dequantized m16n8k16 tensor-core GEMM for IQ weights (types 16,17,18,21,22,23).
// Each warp owns a distinct 16-row weight tile and reuses each register-decoded weight fragment
// across 8 token tiles (64 tokens), eliminating redundant cross-warp weight decoding and the
// 178 MiB global-memory pf_weights round-trip.
template <int TYPE, int BN_T = 64>
__global__ void __launch_bounds__(256, BN_T <= 64 ? 2 : 1) native_mmh(WMat W, const __half * X, int ntok, Out O) {
    constexpr int WM = 8;
    constexpr int WN = 1;
    constexpr int BM_M = WM * 16;
    constexpr int BK_K = 32;
    constexpr int NT_TILES = BN_T / (WN * 8);
    __shared__ __align__(16) __half sx[2][BN_T * BK_K];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int warp_m = warp % WM;
    const int warp_n = warp / WM;

    const int row0 = blockIdx.x * BM_M;
    const int tok0 = blockIdx.y * BN_T;

    float acc[NT_TILES][4];
    #pragma unroll
    for (int n = 0; n < NT_TILES; n++)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;

    auto stage_x = [&](int buf, int k0) {
        #pragma unroll
        for (int idx = tid; idx < (BN_T * BK_K) / 8; idx += 256) {
            const int tok = idx >> 2;
            const int col8 = (idx & 3) << 3;
            __half * dst = &sx[buf][tok * BK_K + col8];
            if (tok0 + tok < ntok) {
                cpa16((unsigned) __cvta_generic_to_shared(dst), X + size_t(tok0 + tok) * W.K + k0 + col8);
            } else {
                *reinterpret_cast<uint4 *>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
        cpa_commit();
    };

    stage_x(0, 0);

    const int r0 = row0 + warp_m * 16 + g;
    const int r1 = r0 + 8;
    const uint8_t * wrow0 = r0 < W.N ? W.w + size_t(r0) * W.rb : nullptr;
    const uint8_t * wrow1 = r1 < W.N ? W.w + size_t(r1) * W.rb : nullptr;

    for (int k0 = 0, step = 0; k0 < W.K; k0 += BK_K, step++) {
        const int cur = step & 1;
        if (k0 + BK_K < W.K) {
            stage_x(cur ^ 1, k0 + BK_K);
            cpa_wait<1>();
        } else {
            cpa_wait<0>();
        }
        __syncthreads();

        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            const int k_col = k0 + kh * 16 + 4 * t;
            unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            if (wrow0) {
                const Quant4 q0 = quant4<TYPE>(wrow0, k_col);
                const __half2 h01 = __floats2half2_rn(q0.d * int8_t(q0.q), q0.d * int8_t(q0.q >> 8));
                const __half2 h23 = __floats2half2_rn(q0.d * int8_t(q0.q >> 16), q0.d * int8_t(q0.q >> 24));
                a0 = *reinterpret_cast<const unsigned *>(&h01);
                a2 = *reinterpret_cast<const unsigned *>(&h23);
            }
            if (wrow1) {
                const Quant4 q1 = quant4<TYPE>(wrow1, k_col);
                const __half2 h01 = __floats2half2_rn(q1.d * int8_t(q1.q), q1.d * int8_t(q1.q >> 8));
                const __half2 h23 = __floats2half2_rn(q1.d * int8_t(q1.q >> 16), q1.d * int8_t(q1.q >> 24));
                a1 = *reinterpret_cast<const unsigned *>(&h01);
                a3 = *reinterpret_cast<const unsigned *>(&h23);
            }

            #pragma unroll
            for (int nt = 0; nt < NT_TILES; nt++) {
                const int tok = tok0 + (warp_n * NT_TILES + nt) * 8 + g;
                unsigned b0 = 0, b1 = 0;
                if (tok < ntok) {
                    const __half * xp = &sx[cur][(tok - tok0) * BK_K + kh * 16 + 2 * t];
                    b0 = *reinterpret_cast<const unsigned *>(xp);
                    b1 = *reinterpret_cast<const unsigned *>(xp + 8);
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < NT_TILES; nt++) {
        const int c0 = tok0 + (warp_n * NT_TILES + nt) * 8 + 2 * t;
        const int c1 = c0 + 1;
        if (r0 < W.N && c0 < ntok) {
            float * dst = O.y + size_t(c0) * O.ldy + r0;
            if (O.mode == OUT_ADD) *dst += acc[nt][0]; else *dst = acc[nt][0];
        }
        if (r0 < W.N && c1 < ntok) {
            float * dst = O.y + size_t(c1) * O.ldy + r0;
            if (O.mode == OUT_ADD) *dst += acc[nt][1]; else *dst = acc[nt][1];
        }
        if (r1 < W.N && c0 < ntok) {
            float * dst = O.y + size_t(c0) * O.ldy + r1;
            if (O.mode == OUT_ADD) *dst += acc[nt][2]; else *dst = acc[nt][2];
        }
        if (r1 < W.N && c1 < ntok) {
            float * dst = O.y + size_t(c1) * O.ldy + r1;
            if (O.mode == OUT_ADD) *dst += acc[nt][3]; else *dst = acc[nt][3];
        }
    }
}

template <int TYPE>
inline void launch_native_mmh(const WMat & W, const __half * X, int ntok, const Out & O) {
    if (ntok <= 32) {
        const dim3 grid((W.N + 127) / 128, (ntok + 31) / 32);
        native_mmh<TYPE, 32><<<grid, 256>>>(W, X, ntok, O);
    } else {
        const dim3 grid((W.N + 127) / 128, (ntok + 63) / 64);
        native_mmh<TYPE, 64><<<grid, 256>>>(W, X, ntok, O);
    }
}

} // namespace pf
