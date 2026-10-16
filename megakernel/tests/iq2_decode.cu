#include "../native_quant.cuh"
#include "../gguf.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
template<int TYPE>__global__ void compare(const uint8_t * w,int k,int stride,size_t fragments,unsigned long long * errors){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<fragments;i+=size_t(gridDim.x)*blockDim.x){
        const size_t row=i/(k/16);const int e=16*(i%(k/16));const uint8_t * p=w+row*stride;
        const auto a=quant8(TYPE,p,e),b=quant8(TYPE,p,e+8);const auto v=iq2_direct16<TYPE>(p,e);
        if(a.q.x!=v.q.x||a.q.y!=v.q.y||b.q.x!=v.q.z||b.q.y!=v.q.w||__float_as_uint(a.d)!=__float_as_uint(v.d)||__float_as_uint(b.d)!=__float_as_uint(v.d))atomicAdd(errors,1ull);
    }
}
int main(int argc,char ** argv){try{
    if(argc!=2)throw std::runtime_error("usage: iq2_decode_test model.gguf");
    gguf::File f(argv[1]);unsigned long long * errors;CK(cudaMalloc(&errors,8));CK(cudaMemset(errors,0,8));
    size_t count=0;int tensors=0;
    for(const auto & item:f.tensors){const auto & t=item.second;if(t.type!=16&&t.type!=17&&t.type!=22)continue;
        const int k=t.ne[0],stride=gguf::type_row_bytes(t.type,k);if(k%256)throw std::runtime_error("unexpected IQ2 row width");
        const size_t fragments=(t.nbytes/stride)*(k/16);uint8_t * w;CK(cudaMalloc(&w,t.nbytes));CK(cudaMemcpy(w,t.data,t.nbytes,cudaMemcpyHostToDevice));
        const int blocks=std::min<size_t>(65535,(fragments+127)/128);
        if(t.type==16)compare<16><<<blocks,128>>>(w,k,stride,fragments,errors);
        if(t.type==17)compare<17><<<blocks,128>>>(w,k,stride,fragments,errors);
        if(t.type==22)compare<22><<<blocks,128>>>(w,k,stride,fragments,errors);
        CK(cudaGetLastError());CK(cudaFree(w));count+=fragments;tensors++;
    }
    unsigned long long bad;CK(cudaMemcpy(&bad,errors,8,cudaMemcpyDeviceToHost));CK(cudaFree(errors));
    if(bad)throw std::runtime_error("direct IQ2 decoder differs from legacy fragments");
    printf("PASS %zu exact IQ2 16-value fragments across all %d real IQ2 tensors\n",count,tensors);
}catch(const std::exception & e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}return 0;}
