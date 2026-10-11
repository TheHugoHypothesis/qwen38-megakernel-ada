#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "ggml.h"
#include "gguf.h"
#include "native_quant.cuh"
#include "prefill_gemm.cuh"
#include "native_prefill.cuh"

#define CU(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
#define BL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure"); } while(0)

template<class F> float timing(F f) {
    cudaEvent_t a,b; CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b));
    f(); CU(cudaDeviceSynchronize());
    CU(cudaEventRecord(a));
    for(int i=0;i<10;i++) f();
    CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
    float ms; CU(cudaEventElapsedTime(&ms,a,b));
    cudaEventDestroy(a); cudaEventDestroy(b); return ms/10;
}

int main(int argc,char ** argv) {
    try {
        if(argc!=2) return 2;
        gguf::File file(argv[1]);
        cublasHandle_t blas; BL(cublasCreate(&blas));
        for(int type : {18,21,23}) {
            const gguf::Tensor * tensor=nullptr;
            for(const auto & entry:file.tensors) {
                const auto & t=entry.second;
                if(t.type==unsigned(type) && t.name.find("ffn_")!=std::string::npos) { tensor=&t; break; }
            }
            if(!tensor) throw std::runtime_error("missing test tensor");
            const auto & t=*tensor;
            const int K=t.ne[0],N=t.ne[1],maxT=512;
            const size_t rb=gguf::type_row_bytes(type,K);
            std::vector<__half> weights(size_t(N)*K),x(size_t(maxT)*K),xp(x.size());
            std::vector<float> row(K);
            for(int r=0;r<N;r++) {
                ggml_get_type_traits(ggml_type(type))->to_float(t.data+size_t(r)*rb,row.data(),K);
                for(int k=0;k<K;k++) weights[size_t(r)*K+k]=__float2half(row[k]);
            }
            for(int i=0;i<maxT;i++) for(int k=0;k<K;k++) {
                const __half value=__float2half(std::sin(float((i*31+k)%1009))*.2f);
                x[size_t(i)*K+k]=value;
                xp[size_t(i)*K+32*(k/32)+pf::hperm(k%32)]=value;
            }
            uint8_t * w; __half * wf,* xd,* xpd; float * y,* ref;
            CU(cudaMalloc(&w,rb*N)); CU(cudaMalloc(&wf,weights.size()*2));
            CU(cudaMalloc(&xd,x.size()*2)); CU(cudaMalloc(&xpd,xp.size()*2));
            CU(cudaMalloc(&y,size_t(maxT)*N*4)); CU(cudaMalloc(&ref,size_t(maxT)*N*4));
            CU(cudaMemcpy(w,t.data,rb*N,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(wf,weights.data(),weights.size()*2,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(xd,x.data(),x.size()*2,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(xpd,xp.data(),xp.size()*2,cudaMemcpyHostToDevice));
            for(int T : {1,127,128,512}) {
                const pf::WMat wm{w,nullptr,type,K,N,int(rb),0};
                const pf::Out out{y,N,pf::OUT_STORE,nullptr};
                auto native=[&] {
                    const dim3 grid((N+63)/64,(T+63)/64);
                    if(type==18) pf::native_mmh<18><<<grid,256>>>(wm,xpd,T,out);
                    if(type==21) pf::native_mmh<21><<<grid,256>>>(wm,xpd,T,out);
                    if(type==23) pf::native_mmh<23><<<grid,256>>>(wm,xpd,T,out);
                    CU(cudaGetLastError());
                };
                auto reference=[&] {
                    const float a=1,b=0;
                    BL(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,N,T,K,&a,wf,CUDA_R_16F,K,xd,CUDA_R_16F,K,
                        &b,ref,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                };
                const float nt=timing(native),rt=timing(reference);
                std::vector<float> got(size_t(T)*N),expected(got.size());
                CU(cudaMemcpy(got.data(),y,got.size()*4,cudaMemcpyDeviceToHost));
                CU(cudaMemcpy(expected.data(),ref,expected.size()*4,cudaMemcpyDeviceToHost));
                double maxerr=0,maxval=0,sse=0,ssr=0;
                for(size_t i=0;i<got.size();i++) {
                    if(!std::isfinite(got[i])) throw std::runtime_error("nonfinite result");
                    const double e=double(got[i])-expected[i];
                    maxerr=std::max(maxerr,std::abs(e)); maxval=std::max(maxval,double(std::abs(expected[i])));
                    sse+=e*e; ssr+=double(expected[i])*expected[i];
                }
                const double relative=std::sqrt(sse/std::max(1e-30,ssr));
                printf("GEMM type %d tensor %s K %d N %d T %d native_ms %.6f cublas_ms %.6f max_error %.9g relative_rms %.9g\n",
                    type,t.name.c_str(),K,N,T,nt,rt,maxerr,relative); fflush(stdout);
                if(maxerr>1e-3*std::max(1.,maxval) || relative>1e-4) throw std::runtime_error("numerical GEMM mismatch");
            }
            cudaFree(w);cudaFree(wf);cudaFree(xd);cudaFree(xpd);cudaFree(y);cudaFree(ref);
        }
        cublasDestroy(blas);
        puts("PASS: native IQ tile GEMMs match CPU-dequantized FP16/cuBLAS reference");
    } catch(const std::exception & e) { fprintf(stderr,"FAIL: %s\n",e.what());return 1; }
}
