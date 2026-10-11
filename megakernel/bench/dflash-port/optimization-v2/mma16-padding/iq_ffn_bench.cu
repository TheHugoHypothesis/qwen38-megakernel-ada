// IQ gate/up microbenchmark with real GGUF tensors and production dot helpers.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../native_iq_mma.cuh"
#include "../gguf.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <vector>

#define CK(call) do { const auto e = (call); if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while (0)
constexpr int K = 5120, FF = 17408, T = 8, U = K / 32, NT = 512, SHARED = 92544;

// Prototype with padded token strides. Only this benchmark uses it until the
// CPU oracle and integrated target validate the new shared layout.
template<int GT, int UT, int STRIDE>
__device__ __forceinline__ void padded_dual(const uint8_t * gate, int gs, const uint8_t * up, int us,
        const int4 * a, const int4 * b, const uint2 * m, float (&g)[4], float (&u)[4]) {
    const int lane = threadIdx.x & 31, row = lane >> 2, word = lane & 3;
    gate += size_t(row)*gs; up += size_t(row)*us;
#pragma unroll
    for (int i=0;i<4;i++) g[i]=u[i]=0.f;
#pragma unroll 1
    for (int e=0;e<K;e+=16) {
        const int unit=8*(e/256)+2*((e&255)/64)+((e&31)/16), part=(e&63)/32, element=e+8*(word/2);
        const Quant8 wg0=quant8(GT,gate,element),wg1=quant8(GT,gate+size_t(8)*gs,element);
        const Quant8 wu0=quant8(UT,up,element),wu1=quant8(UT,up+size_t(8)*us,element);
        const int4 x=part?b[row*STRIDE+unit]:a[row*STRIDE+unit];
        const int packed=word==0?x.x:word==1?x.y:word==2?x.z:x.w;
        const int4 dg=iq_mma16x8((word&1)?wg0.q.y:wg0.q.x,(word&1)?wg1.q.y:wg1.q.x,packed);
        const int4 du=iq_mma16x8((word&1)?wu0.q.y:wu0.q.x,(word&1)?wu1.q.y:wu1.q.x,packed);
        const float d0=__half2float(reinterpret_cast<const __half *>(m+2*word*STRIDE+unit)[part]);
        const float d1=__half2float(reinterpret_cast<const __half *>(m+(2*word+1)*STRIDE+unit)[part]);
        g[0]+=d0*wg0.d*dg.x;g[1]+=d1*wg0.d*dg.y;g[2]+=d0*wg1.d*dg.z;g[3]+=d1*wg1.d*dg.w;
        u[0]+=d0*wu0.d*du.x;u[1]+=d1*wu0.d*du.y;u[2]+=d0*wu1.d*du.z;u[3]+=d1*wu1.d*du.w;
    }
}

template<int GT, int UT, bool DP4A, int PAD = 0>
__global__ void __launch_bounds__(NT, 1) phase(const uint8_t * gate, int gs, const uint8_t * up, int us, float * out) {
    extern __shared__ __align__(16) uint8_t sm[];
    constexpr int STRIDE=U+PAD;
    auto * a = reinterpret_cast<int4 *>(sm); auto * b = a + T * STRIDE;
    auto * m = reinterpret_cast<uint2 *>(b + T * STRIDE);
    for (int i = threadIdx.x; i < T * U; i += NT) {
        const int dst=(i/U)*STRIDE+i%U;
        a[dst] = make_int4(0x0e010a07, 0x03050204, 0x7f10132a, 0xe006ff02); b[dst] = a[dst];
        *reinterpret_cast<__half2 *>(&m[dst].x) = __floats2half2_rn(.002f*(1+i%7), .003f*(1+i%11)); m[dst].y = 0;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int first = warp * gridDim.x + blockIdx.x, stride = (NT / 32) * gridDim.x;
    for (int j = first; j < FF / 16; j += stride) {
        if constexpr (DP4A) {
#pragma unroll 1
            for (int row = 0; row < 16; row++) {
                float g[T], u[T];
                iq_dp4a_dual<T, U, GT, UT>(gate + size_t(16*j + row)*gs, up + size_t(16*j + row)*us, a, b, m, g, u);
#pragma unroll
                for (int t = 0; t < T; t++) {
#pragma unroll
                    for (int shift = 16; shift; shift /= 2) {
                        g[t] += __shfl_down_sync(0xffffffffu, g[t], shift);
                        u[t] += __shfl_down_sync(0xffffffffu, u[t], shift);
                    }
                    if (lane == 0) { out[t*FF+16*j+row] = g[t]; out[(T+t)*FF+16*j+row] = u[t]; }
                }
            }
        } else {
            float g[4], u[4];
            if constexpr (PAD) padded_dual<GT,UT,STRIDE>(gate+size_t(16*j)*gs,gs,up+size_t(16*j)*us,us,a,b,m,g,u);
            else iq_mma16_dual<GT, UT>(gate + size_t(16*j)*gs, gs, up + size_t(16*j)*us, us, a, b, m, U, g, u);
#pragma unroll
            for (int half = 0; half < 2; half++) {
#pragma unroll
                for (int t = 0; t < 2; t++) {
                    const int row = 16*j + (lane >> 2) + 8*half, token = 2*(lane & 3)+t;
                    out[token*FF+row] = g[2*half+t]; out[(T+token)*FF+row] = u[2*half+t];
                }
            }
        }
    }
}

template<int GT, int UT, bool DP4A, int PAD = 0> static float measure(const uint8_t * g, int gs, const uint8_t * u, int us, float * out, int sms) {
    CK(cudaFuncSetAttribute(phase<GT, UT, DP4A, PAD>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED));
    for (int i = 0; i < 5; i++) phase<GT, UT, DP4A, PAD><<<sms, NT, SHARED>>>(g, gs, u, us, out);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEvent_t start, end; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&end));
    CK(cudaEventRecord(start));
    for (int i = 0; i < 20; i++) phase<GT, UT, DP4A, PAD><<<sms, NT, SHARED>>>(g, gs, u, us, out);
    CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end)); float ms; CK(cudaEventElapsedTime(&ms, start, end));
    CK(cudaEventDestroy(start)); CK(cudaEventDestroy(end)); return ms / 20;
}

template<int GT, int UT> static void pair(const uint8_t * g, int gs, const uint8_t * u, int us, float * out, int sms) {
    const float mma = measure<GT, UT, false>(g, gs, u, us, out, sms);
    std::vector<float> reference(2*T*FF), actual(2*T*FF);
    CK(cudaMemcpy(reference.data(),out,reference.size()*sizeof(float),cudaMemcpyDeviceToHost));
    const float padded = measure<GT, UT, false, 1>(g, gs, u, us, out, sms);
    CK(cudaMemcpy(actual.data(),out,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    double error2=0,ref2=0,maximum=0;
    for (size_t i=0;i<actual.size();i++) {
        if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) throw std::runtime_error("nonfinite padded MMA result");
        const double delta=double(actual[i])-reference[i];error2+=delta*delta;ref2+=double(reference[i])*reference[i];
        maximum=std::max(maximum,std::abs(delta));
    }
    const double nrmse=ref2?std::sqrt(error2/ref2):std::sqrt(error2);
    if (nrmse>1e-5 || maximum>1e-4*(1+std::sqrt(ref2/actual.size()))) throw std::runtime_error("padding changed IQ MMA values");
    float dp4a = -1;
    if constexpr ((GT == 18 || GT == 21 || GT == 23) && (UT == 18 || UT == 21 || UT == 23))
        dp4a = measure<GT, UT, true>(g, gs, u, us, out, sms);
    printf("%d,%d,%.6f,%.6f,%.6f,%.4f,%.9g\n", GT, UT, dp4a, mma, padded, dp4a > 0 ? dp4a/padded : -1,nrmse); fflush(stdout);
}

template<int GT> static void up_type(int ut, const uint8_t * g, int gs, const uint8_t * u, int us, float * out, int sms) {
#define PAIR(IQ) case IQ: pair<GT, IQ>(g, gs, u, us, out, sms); break;
    switch (ut) { PAIR(16) PAIR(17) PAIR(18) PAIR(21) PAIR(22) PAIR(23) }
#undef PAIR
}

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: iq_ffn_bench model.gguf");
        gguf::File file(argv[1]); std::set<std::pair<int,int>> seen;
        cudaDeviceProp props; CK(cudaGetDeviceProperties(&props, 0));
        float * out; CK(cudaMalloc(&out, 2*T*FF*sizeof(float)));
        printf("gate_type,up_type,dp4a_ms,mma16_ms,padded_ms,dp4a_over_padded,padded_nrmse\n");
        for (int layer = 0; layer < 64; layer++) {
            const auto prefix = std::string("blk.")+std::to_string(layer)+".ffn_";
            const auto & gt = file.get(prefix+"gate.weight"); const auto & ut = file.get(prefix+"up.weight");
            if (!iq_mma_supported(gt.type) || !iq_mma_supported(ut.type) || !seen.insert({gt.type,ut.type}).second) continue;
            if (gt.ne != std::vector<uint64_t>{K,FF} || ut.ne != gt.ne) throw std::runtime_error("unexpected FFN shape");
            uint8_t * g, * u; CK(cudaMalloc(&g, gt.nbytes)); CK(cudaMalloc(&u, ut.nbytes));
            CK(cudaMemcpy(g, gt.data, gt.nbytes, cudaMemcpyHostToDevice)); CK(cudaMemcpy(u, ut.data, ut.nbytes, cudaMemcpyHostToDevice));
            const int gs = gguf::type_row_bytes(gt.type, K), us = gguf::type_row_bytes(ut.type, K);
#define GATE(IQ) case IQ: up_type<IQ>(ut.type, g, gs, u, us, out, props.multiProcessorCount); break;
            switch (gt.type) { GATE(16) GATE(17) GATE(18) GATE(21) GATE(22) GATE(23) }
#undef GATE
            CK(cudaFree(g)); CK(cudaFree(u));
        }
        CK(cudaFree(out));
    } catch (const std::exception & error) { fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
}
