#pragma once
#include <mma.h>

namespace pf {
// IQ weights and permuted FP16 activations are unpacked into a bounded tile.
template <int TYPE>
__global__ void __launch_bounds__(256) native_mmh(WMat W, const __half * X, int ntok, Out O) {
    constexpr int M=64,N=64,K=64;
    __shared__ __align__(32) __half stage[(M+N)*K];
    __half * a=stage, * b=stage+M*K;
    const int tid=threadIdx.x, warp=tid/32;
    const int row0=blockIdx.x*M, tok0=blockIdx.y*N;
    const int wm=warp%4, wn=warp/4;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> acc[2];
    nvcuda::wmma::fill_fragment(acc[0],0.f);
    nvcuda::wmma::fill_fragment(acc[1],0.f);
    for(int k0=0;k0<W.K;k0+=K) {
        const int r=tid/4, e=16*(tid%4);
        Quant16 q;
        if(row0+r<W.N) q=quant16<TYPE>(W.w+size_t(row0+r)*W.rb,k0+e);
        else q={make_int4(0,0,0,0),0.f};
        const int * words=reinterpret_cast<const int *>(&q.q);
        #pragma unroll
        for(int j=0;j<8;j++) {
            const unsigned word=unsigned(words[j/2])>>(16*(j%2));
            reinterpret_cast<__half2 *>(a+r*K+e)[j]=__floats2half2_rn(q.d*int8_t(word),q.d*int8_t(word>>8));
        }
        for(int i=tid;i<N*K;i+=256) {
            const int t=i/K,k=i%K;
            b[i]=tok0+t<ntok ? X[size_t(tok0+t)*W.K+k0+hperm(k)] : __float2half(0.f);
        }
        __syncthreads();
        #pragma unroll
        for(int j=0;j<K;j+=16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__half,nvcuda::wmma::row_major> af;
            nvcuda::wmma::load_matrix_sync(af,a+16*wm*K+j,K);
            #pragma unroll
            for(int n=0;n<2;n++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__half,nvcuda::wmma::col_major> bf;
                nvcuda::wmma::load_matrix_sync(bf,b+(32*wn+16*n)*K+j,K);
                nvcuda::wmma::mma_sync(acc[n],af,bf,acc[n]);
            }
        }
        __syncthreads();
    }
    float * output=reinterpret_cast<float *>(stage);
    #pragma unroll
    for(int n=0;n<2;n++)
        nvcuda::wmma::store_matrix_sync(output+16*wm*N+32*wn+16*n,acc[n],N,nvcuda::wmma::mem_row_major);
    __syncthreads();
    for(int i=tid;i<M*N;i+=256) {
        const int r=row0+i/N,t=tok0+i%N;
        if(r<W.N && t<ntok) {
            float * dst=O.y+size_t(t)*O.ldy+r;
            if(O.mode==OUT_ADD) *dst+=output[i]; else *dst=output[i];
        }
    }
}
} // namespace pf
