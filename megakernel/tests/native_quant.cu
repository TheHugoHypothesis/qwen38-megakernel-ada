// Compare actual GGUF blocks against llama.cpp's independent CPU dequantizers.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "ggml.h"
#include "../gguf.h"
#include "native_quant.cuh"
#define CUDA_CHECK(x) do { auto e = (x); if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while (0)
__global__ void decode(int type, const uint8_t * row, float * y, int n, const __half * ds) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = native_value(type, row, i, ds);
    if (i < n && (type == 18 || type == 21 || type == 23)) {
        Quant16 w;
        if (type == 18) w = quant16<18>(row, i & ~15);
        else if (type == 21) w = quant16<21>(row, i & ~15);
        else w = quant16<23>(row, i & ~15);
        const int * words = reinterpret_cast<const int *>(&w.q);
        y[n+i] = w.d * int8_t(unsigned(words[(i&15)/4]) >> (8*(i&3)));
    }
}
__global__ void product16(int type, const uint8_t * row, float * y, int n) {
    const int i = 16*(blockIdx.x*blockDim.x+threadIdx.x);
    if (i >= n) return;
    Quant16 w;
    if (type == 18) w=quant16<18>(row,i);
    else if (type == 21) w=quant16<21>(row,i);
    else w=quant16<23>(row,i);
    int dot=0;
    const int * words=reinterpret_cast<const int *>(&w.q);
    for(int j=0;j<4;j++) {
        unsigned x=0;
        for(int k=0;k<4;k++) x |= unsigned(uint8_t(int8_t(((i+4*j+k)*37)%255-127))) << (8*k);
        dot=__dp4a(words[j],int(x),dot);
    }
    y[i/16]=w.d*dot;
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    gguf::File f(argv[1]);
    size_t tested = 0;
    for (const auto & entry : f.tensors) {
        const auto & t = entry.second;
        if (!gguf::is_native_quant(t.type) && t.type != gguf::Q4_K && t.type != gguf::Q5_K && t.type != gguf::Q6_K && t.type != gguf::Q8_0) continue;
        const int n = std::min(uint64_t(4096), t.ne[0]);
        const size_t rb = gguf::type_row_bytes(t.type, t.ne[0]);
        const size_t bytes = gguf::type_row_bytes(t.type, n);
        const size_t gpu_bytes=t.type==gguf::Q6_K ? size_t(n/256)*224 : t.type==gguf::Q8_0 ? size_t(n) : bytes;
        uint8_t * w; float * y; __half * ds;
        const bool wide=t.type==18 || t.type==21 || t.type==23;
        CUDA_CHECK(cudaMalloc(&w, gpu_bytes)); CUDA_CHECK(cudaMalloc(&y, 2*n*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ds, size_t(n/32)*2));
        for (size_t r : {size_t(0), size_t(t.ne.size() > 1 ? t.ne[1]/2 : 0), size_t(t.ne.size() > 1 ? t.ne[1]-1 : 0)}) {
            std::vector<float> got(wide?2*n:n), expected(n);
            ggml_get_type_traits(ggml_type(t.type))->to_float(t.data + r*rb, expected.data(), n);
            const uint8_t * src=t.data+r*rb;
            std::vector<uint8_t> packed(gpu_bytes,0);
            if(t.type==gguf::Q6_K) {
                for(int b=0;b<n/256;b++) {
                    const auto * x=src+b*210;
                    auto * out=packed.data()+b*224;
                    for(int e=0;e<256;e++) {
                        const int m=e/128,l=e%128,p=e/64,h=(e%32)/16,part=(e%64)/32,k=e%16;
                        const int lo=(x[64*m+(l%64)]>>(4*(l/64)))&15;
                        const unsigned hi=(x[128+32*m+(l%32)]>>(2*(l/32)))&3;
                        out[32*p+16*h+k] |= lo<<(4*part);
                        unsigned word=0;auto * dst=out+128+8*(2*p+h)+4*part;
                        memcpy(&word,dst,4);word |= hi<<(8*(k%4)+2*(k/4));memcpy(dst,&word,4);
                    }
                    for(int p=0;p<4;p++) {memcpy(out+192+8*p,x+192+4*p,4);memcpy(out+196+8*p,x+208,2);}
                }
            } else if(t.type==gguf::Q8_0) {
                std::vector<uint16_t> scales(n/32);
                for(int e=0;e<n;e++) {
                    const int u=8*(e/256)+2*((e%256)/64)+(e%32)/16;
                    packed[32*u+16*((e%64)/32)+(e%16)]=src[34*(e/32)+2+(e%32)];
                }
                for(int b=0;b<n/32;b++) memcpy(&scales[b],src+34*b,2);
                CUDA_CHECK(cudaMemcpy(ds,scales.data(),scales.size()*2,cudaMemcpyHostToDevice));
            } else memcpy(packed.data(),src,bytes);
            CUDA_CHECK(cudaMemcpy(w,packed.data(),gpu_bytes,cudaMemcpyHostToDevice));
            decode<<<(n+255)/256,256>>>(t.type,w,y,n,ds);
            CUDA_CHECK(cudaMemcpy(got.data(),y,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
            for (size_t i=0;i<got.size();i++) {
                const float reference=expected[i%n];
                const float error=std::abs(got[i]-reference);
                if (!std::isfinite(got[i]) || error > 2e-6f*std::max(1.f,std::abs(reference))) {
                    fprintf(stderr,"FAIL %s type=%u row=%zu e=%zu GPU=%.9g CPU=%.9g\n",t.name.c_str(),t.type,r,i,got[i],reference);
                    return 1;
                }
            }
            tested += got.size();
            if(wide) {
                product16<<<(n/16+255)/256,256>>>(t.type,w,y,n);
                CUDA_CHECK(cudaMemcpy(got.data(),y,n/16*sizeof(float),cudaMemcpyDeviceToHost));
                for(int i=0;i<n/16;i++) {
                    double sum=0, bound=0;
                    for(int j=0;j<16;j++) {
                        const double v=expected[16*i+j]*int8_t(((16*i+j)*37)%255-127);
                        sum+=v; bound+=std::abs(v);
                    }
                    if(!std::isfinite(got[i]) || std::abs(got[i]-sum)>2e-6*std::max(1.,bound)) {
                        fprintf(stderr,"FAIL dot16 %s row=%zu group=%d GPU=%.9g CPU=%.9g\n",t.name.c_str(),r,i,got[i],sum);
                        return 1;
                    }
                }
            }
        }
        cudaFree(w); cudaFree(y); cudaFree(ds);
    }
    fprintf(stderr,"PASS: %zu real GGUF values match llama.cpp CPU dequantization\n",tested);
}
