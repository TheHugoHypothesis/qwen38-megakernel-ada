// Native DFlash2 math follows llama.cpp/src/models/dflash.cpp (MIT).
// Quantized readers reuse native_quant.cuh; license: LICENSE.llama-cpp.
#include "dflash_native.h"
#include "gguf.h"
#include "ggml.h"
#include "native_quant.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#define DC(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e)); } while(0)
namespace dn {
constexpr int D=5120,FF=17408,VOCAB=248320,NL=5,HD=128,QD=4096,KD=1024;
constexpr int WINDOW=2048,TOP=16,RANK=256,MAXT=8;
constexpr int CACHE=WINDOW+MAXT; // preserves history across a <=7-token truncate
struct Matrix { const uint8_t * w; int K,N,type,rb; };
struct Layer { Matrix q,k,v,o,gate,up,down,ac,fc; const float * an,*fn,*qn,*kn,*ab,*fb; };
struct Control { int position,anchor; };

__device__ float sumwarp(float x) { for(int s=16;s;s/=2)x+=__shfl_xor_sync(0xffffffff,x,s);return x; }
__device__ uint2 load_even8(const uint8_t * p) {
    // Q6_K blocks are 210 bytes: odd blocks are aligned to two bytes.
    const auto * h=reinterpret_cast<const uint16_t *>(p);
    return make_uint2(unsigned(h[0])|(unsigned(h[1])<<16),unsigned(h[2])|(unsigned(h[3])<<16));
}
template<int TYPE=0> __device__ __forceinline__ Quant8 fragment_typed(Matrix w,int row,int e) {
    const int type=TYPE ? TYPE : w.type;
    const uint8_t * p=w.w+size_t(row)*w.rb;
    if(type==12) {
        const uint8_t * b=p+size_t(e/256)*144,*sc=b+4;
        const int g=(e%256)/32;int s,m;
        if(g<4){s=sc[g]&63;m=sc[g+4]&63;}
        else{s=(sc[g+4]&15)|((sc[g-4]>>6)<<4);m=(sc[g+4]>>4)|((sc[g]>>6)<<4);}
        const auto v=*reinterpret_cast<const uint2 *>(b+16+32*(g/2)+e%32);
        const unsigned shift=4*(g&1),mask=0x0f0f0f0f;
        return {make_int2(int((v.x>>shift)&mask),int((v.y>>shift)&mask)),
            __half2float(*reinterpret_cast<const __half *>(b))*s,
            -__half2float(*reinterpret_cast<const __half *>(b+2))*m};
    }
    if(type!=14) return quant8(type,p,e);
    const uint8_t * b=p+size_t(e/256)*210;
    const int off=e%256,sub=off/128,g=(off%128)/32,l=off%32;
    const float d=__half2float(*reinterpret_cast<const __half *>(b+208))*int8_t(b[192+off/16]);
    const auto lo=load_even8(b+64*sub+(g%2)*32+l);
    const auto hi=load_even8(b+128+32*sub+l);
    const unsigned a=((lo.x>>(4*(g/2)))&0x0f0f0f0f)|(((hi.x>>(2*g))&0x03030303)<<4);
    const unsigned c=((lo.y>>(4*(g/2)))&0x0f0f0f0f)|(((hi.y>>(2*g))&0x03030303)<<4);
    return {make_int2(int(__vsub4(a,0x20202020)),int(__vsub4(c,0x20202020))),d,0};
}
__device__ Quant8 fragment(Matrix w,int row,int e) { return fragment_typed<0>(w,row,e); }
__device__ float weight(Matrix w,int row,int e) {
    const auto q=fragment(w,row,e&~7);
    const unsigned v=unsigned(e%8<4?q.q.x:q.q.y);
    return q.d*int8_t(v>>(8*(e%4)))+q.offset;
}

// A warp streams two quantized weight rows once for all tokens. F32 inputs,
// F32 accumulation; no activation requantization or vocabulary restriction.
template<int T>
__global__ void mm(Matrix w,const float * x,float * y) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int r=(blockIdx.x*4+warp)*2;
    if(r>=w.N)return;
    float acc[2][T]={};
    for(int e=8*lane;e<w.K;e+=256) {
        const auto a=fragment(w,r,e),b=fragment(w,r+1,e);
        #pragma unroll
        for(int j=0;j<8;j++) {
            const float wa=a.d*int8_t(unsigned(j<4?a.q.x:a.q.y)>>(8*(j%4)))+a.offset;
            const float wb=b.d*int8_t(unsigned(j<4?b.q.x:b.q.y)>>(8*(j%4)))+b.offset;
            #pragma unroll
            for(int t=0;t<T;t++) {
                const float v=__ldg(x+size_t(t)*w.K+e+j);
                acc[0][t]=fmaf(wa,v,acc[0][t]);acc[1][t]=fmaf(wb,v,acc[1][t]);
            }
        }
    }
    #pragma unroll
    for(int t=0;t<T;t++) {
        float a=sumwarp(acc[0][t]),b=sumwarp(acc[1][t]);
        if(lane==0){y[size_t(t)*w.N+r]=a;y[size_t(t)*w.N+r+1]=b;}
    }
}

__global__ void dequant_half(Matrix w,__half * y) {
    const size_t i=(size_t(blockIdx.x)*256+threadIdx.x)*8;
    if(i>=size_t(w.K)*w.N)return;
    const auto q=fragment(w,int(i/w.K),int(i%w.K));
    #pragma unroll
    for(int j=0;j<8;j++) {
        const unsigned word=unsigned(j<4?q.q.x:q.q.y);
        y[i+j]=__float2half_rn(q.d*int8_t(word>>(8*(j%4)))+q.offset);
    }
}

__global__ void quant_input(const float * x,int8_t * q,float * scale,int K,int n) {
    const int block=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(block>=n*K/32)return;
    float v=x[block*32+lane],m=fabsf(v);
    for(int s=16;s;s/=2)m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,s));
    const float d=m/127.f;
    q[block*32+lane]=int8_t(d?__float2int_rn(v/d):0);
    if(lane==0)scale[block]=__half2float(__float2half_rn(d));
}
template<int T>__global__ void mmq(Matrix w,const int8_t * x,const float * scale,float * y) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,r=(blockIdx.x*4+warp)*2;
    if(r>=w.N)return;float acc[2][T]={};
    for(int e=8*lane;e<w.K;e+=256) {
        const auto a=fragment(w,r,e),b=fragment(w,r+1,e);
        #pragma unroll
        for(int t=0;t<T;t++) {
            const int2 v=*reinterpret_cast<const int2 *>(x+size_t(t)*w.K+e);
            const float d=scale[size_t(t)*(w.K/32)+e/32];
            const float sum=float(__dp4a(v.x,0x01010101,__dp4a(v.y,0x01010101,0)))*d;
            acc[0][t]=fmaf(float(__dp4a(a.q.x,v.x,__dp4a(a.q.y,v.y,0))),a.d*d,acc[0][t])+a.offset*sum;
            acc[1][t]=fmaf(float(__dp4a(b.q.x,v.x,__dp4a(b.q.y,v.y,0))),b.d*d,acc[1][t])+b.offset*sum;
        }
    }
    #pragma unroll
    for(int t=0;t<T;t++){const float a=sumwarp(acc[0][t]),b=sumwarp(acc[1][t]);if(lane==0){y[size_t(t)*w.N+r]=a;y[size_t(t)*w.N+r+1]=b;}}
}
// Same dot8/FMA/reduction order as mmq. Reuse the activation/scale for more
// rows and select the GGUF format before the K loop.
template<int T,int TYPE,int ROWS> __global__ void mmq_typed(Matrix w,const int8_t * x,const float * scale,float * y) {
    static_assert((TYPE==12||TYPE==14)&&(ROWS==2||ROWS==4),"native MMQ tile");
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,r=(blockIdx.x*4+warp)*ROWS;
    if(r>=w.N)return;
    float acc[ROWS][T]={};
    for(int e=8*lane;e<w.K;e+=256) {
        Quant8 q[ROWS];
        #pragma unroll
        for(int row=0;row<ROWS;row++) q[row]=fragment_typed<TYPE>(w,r+row,e);
        #pragma unroll
        for(int t=0;t<T;t++) {
            const int2 v=*reinterpret_cast<const int2 *>(x+size_t(t)*w.K+e);
            const float d=scale[size_t(t)*(w.K/32)+e/32];
            const float sum=float(__dp4a(v.x,0x01010101,__dp4a(v.y,0x01010101,0)))*d;
            #pragma unroll
            for(int row=0;row<ROWS;row++)
                acc[row][t]=fmaf(float(__dp4a(q[row].q.x,v.x,__dp4a(q[row].q.y,v.y,0))),q[row].d*d,acc[row][t])+q[row].offset*sum;
        }
    }
    #pragma unroll
    for(int row=0;row<ROWS;row++) {
        #pragma unroll
        for(int t=0;t<T;t++) {
            const float value=sumwarp(acc[row][t]);
            if(lane==0)y[size_t(t)*w.N+r+row]=value;
        }
    }
}
__global__ void to_half(const float * x,__half * y,size_t n) {
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i<n)y[i]=__float2half_rn(x[i]);
}
__global__ void embed(Matrix w,float * x,int n,int mask,const Control * c) {
    const int i=blockIdx.x*256+threadIdx.x;
    if(i<n*D)x[i]=weight(w,i/D==0?c->anchor:mask,i%D);
}
__global__ void norm(const float * x,const float * w,float * y,int width,int n) {
    const int r=blockIdx.x;if(r>=n)return;
    __shared__ float sums[8];
    float ss=0;for(int e=threadIdx.x;e<width;e+=256){float v=x[size_t(r)*width+e];ss+=v*v;}
    ss=sumwarp(ss);if(threadIdx.x%32==0)sums[threadIdx.x/32]=ss;__syncthreads();
    float total=threadIdx.x<8?sums[threadIdx.x]:0;total=sumwarp(total);
    if(threadIdx.x==0)sums[0]=rsqrtf(total/width+1e-6f);__syncthreads();
    for(int e=threadIdx.x;e<width;e+=256)y[size_t(r)*width+e]=x[size_t(r)*width+e]*sums[0]*w[e];
}
__global__ void conv(const float * x,const float * dynamic,const float * base,float * y,int n,int side) {
    const int i=blockIdx.x*256+threadIdx.x;if(i>=n*D)return;
    const int t=i/D,e=i%D,g=e/16;
    float v=x[i]*(dynamic[t*1280+side*640+g]+base[side*2*D+e]);
    if(t>0)v+=x[i-D]*(dynamic[t*1280+side*640+320+g]+base[side*2*D+D+e]);
    y[i]=v;
}
__global__ void add(const float * a,const float * b,float * y,int n) {
    const int i=blockIdx.x*256+threadIdx.x;if(i<n)y[i]=a[i]+b[i];
}
__global__ void glu(const float * g,const float * u,float * y,int n) {
    const int i=blockIdx.x*256+threadIdx.x;if(i<n)y[i]=g[i]/(1.f+expf(-g[i]))*u[i];
}

// Norm/RoPE are computed in the original (unrotated) coordinate system.
// FP16 caches need no quantization Hadamard rotation. Noise has separate storage.
__global__ void cache_kv(const float * k,const float * v,const float * kn,
                         __half * cache_k,__half * cache_v,int pos,int n,const Control * c,bool noise) {
    const int row=blockIdx.x,t=row/8,h=row%8,e=threadIdx.x;
    if(t>=n||e>=HD)return;
    __shared__ float values[HD];
    const float kv=k[t*KD+h*HD+e];
    float ss=sumwarp(kv*kv);__shared__ float sums[4];
    if(e%32==0)sums[e/32]=ss;__syncthreads();
    if(e==0)sums[0]=rsqrtf((sums[0]+sums[1]+sums[2]+sums[3])/HD+1e-6f);__syncthreads();
    values[e]=kv*sums[0]*kn[e];__syncthreads();
    const int p=(noise?c->position:pos)+t;
    float kr=values[e];
    {const int j=e%64;const float theta=float(p)*powf(10000000.f,-2.f*j/128.f);float sn,cs;sincosf(theta,&sn,&cs);
        kr=e<64?values[e]*cs-values[e+64]*sn:values[e]*cs+values[e-64]*sn;}
    const size_t idx=noise?(size_t(t)*8+h)*HD:(size_t(h)*CACHE+p%CACHE)*HD;
    cache_k[idx+e]=__float2half_rn(kr);cache_v[idx+e]=__float2half_rn(v[t*KD+h*HD+e]);
}
__global__ void query_rope(float * q,const float * qn,int n,const Control * c) {
    const int row=blockIdx.x,t=row/32,h=row%32,e=threadIdx.x;if(t>=n)return;
    __shared__ float values[HD],sums[4];
    float v=q[t*QD+h*HD+e],ss=sumwarp(v*v);
    if(e%32==0)sums[e/32]=ss;__syncthreads();
    if(e==0)sums[0]=rsqrtf((sums[0]+sums[1]+sums[2]+sums[3])/HD+1e-6f);__syncthreads();
    values[e]=v*sums[0]*qn[e];__syncthreads();
    v=values[e];{float sn,cs;sincosf(float(c->position+t)*powf(10000000.f,-2.f*(e%64)/128.f),&sn,&cs);
        v=e<64?values[e]*cs-values[e+64]*sn:values[e]*cs+values[e-64]*sn;}
    q[t*QD+h*HD+e]=v;
}

__global__ void attention(const float * q,const __half * hk,const __half * hv,const __half * nk,const __half * nv,
                          float * partial,float2 * stats,int n,int chunks,const Control * ctl) {
    const int lane=threadIdx.x%32,row=blockIdx.x*8+threadIdx.x/32;
    if(row>=n*32)return;
    const int t=row/32,h=row%32,g=h/4,pos=ctl->position;
    const int first=max(0,pos+t-WINDOW+1),total=pos+n-first;
    const int length=(total+chunks-1)/chunks,begin=first+blockIdx.y*length,end=min(pos+n,begin+length);
    float a[4]={},query[4],m=-INFINITY,s=0;
    #pragma unroll
    for(int j=0;j<4;j++)query[j]=q[t*QD+h*HD+lane+32*j];
    for(int p=begin;p<end;p++) {
        const size_t i=p<pos?(size_t(g)*CACHE+p%CACHE)*HD:(size_t(p-pos)*8+g)*HD;
        const auto * k=p<pos?hk:nk;const auto * v=p<pos?hv:nv;
        float dot=0;
        #pragma unroll
        for(int j=0;j<4;j++)dot=fmaf(query[j],__half2float(k[i+lane+32*j]),dot);
        dot=sumwarp(dot)*(1.f/sqrtf(128.f));
        const float nm=fmaxf(m,dot),alpha=expf(m-nm),beta=expf(dot-nm);
        #pragma unroll
        for(int j=0;j<4;j++)a[j]=a[j]*alpha+beta*__half2float(v[i+lane+32*j]);
        s=s*alpha+beta;m=nm;
    }
    const size_t dest=(size_t(row)*chunks+blockIdx.y)*HD;
    #pragma unroll
    for(int j=0;j<4;j++)partial[dest+lane+32*j]=a[j];
    if(lane==0)stats[size_t(row)*chunks+blockIdx.y]=make_float2(m,s);
}
__global__ void attention_combine(const float * partial,const float2 * stats,float * out,int n,int chunks) {
    const int i=blockIdx.x*256+threadIdx.x;if(i>=n*QD)return;
    const int row=i/HD,e=i%HD;float m=-INFINITY,num=0,den=0;
    for(int c=0;c<chunks;c++)m=fmaxf(m,stats[row*chunks+c].x);
    for(int c=0;c<chunks;c++) {const auto st=stats[row*chunks+c];const float a=st.x==-INFINITY?0:expf(st.x-m);
        num+=a*partial[(size_t(row)*chunks+c)*HD+e];den+=a*st.y;}
    out[i]=num/den;
}

__device__ void best_reduce(float & value,int & id) {
    const int lane=threadIdx.x%32;
    for(int off=16;off;off/=2){const float v=__shfl_down_sync(0xffffffff,value,off);const int k=__shfl_down_sync(0xffffffff,id,off);
        if(lane+off<32&&(v>value||(v==value&&k<id))){value=v;id=k;}}
}
__global__ void top_partial(const float * logits,float2 * partial,int blocks,int vocab) {
    const int id0=blockIdx.x*256+threadIdx.x,t=blockIdx.y;
    float own=id0<vocab?logits[size_t(t)*vocab+id0]:-INFINITY;
    __shared__ float vals[8];__shared__ int ids[8],selected;
    for(int j=0;j<TOP;j++) {
        float v=own;int id=id0;best_reduce(v,id);
        if(threadIdx.x%32==0){vals[threadIdx.x/32]=v;ids[threadIdx.x/32]=id;}__syncthreads();
        if(threadIdx.x<32){v=threadIdx.x<8?vals[threadIdx.x]:-INFINITY;id=threadIdx.x<8?ids[threadIdx.x]:vocab;
            best_reduce(v,id);if(threadIdx.x==0){partial[(size_t(t)*blocks+blockIdx.x)*TOP+j]=make_float2(v,float(id));selected=id;}}
        __syncthreads();if(id0==selected)own=-INFINITY;__syncthreads();
    }
}
__global__ void top_finish(const float2 * partial,float * lattice,float * unary,int blocks,int vocab) {
    const int t=blockIdx.x;__shared__ int used[TOP],ids[8];__shared__ float vals[8];
    for(int j=0;j<TOP;j++) {
        float v=-INFINITY;int id=vocab;
        for(int i=threadIdx.x;i<blocks*TOP;i+=256) {
            const auto p=partial[size_t(t)*blocks*TOP+i];bool skip=false;
            for(int k=0;k<j;k++)skip|=int(p.y)==used[k];
            if(!skip&&(p.x>v||(p.x==v&&int(p.y)<id))){v=p.x;id=int(p.y);}
        }
        best_reduce(v,id);if(threadIdx.x%32==0){vals[threadIdx.x/32]=v;ids[threadIdx.x/32]=id;}__syncthreads();
        if(threadIdx.x<32){v=threadIdx.x<8?vals[threadIdx.x]:-INFINITY;id=threadIdx.x<8?ids[threadIdx.x]:vocab;
            best_reduce(v,id);if(threadIdx.x==0){used[j]=id;lattice[size_t(t+1)*D+j]=float(id);unary[t*TOP+j]=v;}}
        __syncthreads();
    }
}
__global__ void selector(Matrix prev,Matrix next,const float * gate,const float * unary,float * lattice,const Control * c) {
    const int successor=blockIdx.x%TOP,predecessor=blockIdx.x/TOP,t=blockIdx.y+1,e=threadIdx.x;
    const int pid=t==1?c->anchor:int(lattice[size_t(t-1)*D+predecessor]);
    const int nid=int(lattice[size_t(t)*D+successor]);
    float v=weight(prev,pid,e)*gate[t*RANK+e]*weight(next,nid,e);v=sumwarp(v);
    __shared__ float sums[8];if(e%32==0)sums[e/32]=v;__syncthreads();
    v=e<8?sums[e]:0;v=sumwarp(v);
    if(e==0)lattice[size_t(t)*D+TOP+predecessor*TOP+successor]=v+unary[(t-1)*TOP+successor];
}
__global__ void walk(const float * lattice,int * ids,int depth) {
    if(threadIdx.x||blockIdx.x)return;int pred=0;
    for(int t=1;t<=depth;t++){const float * row=lattice+size_t(t)*D;
        const float * scores=row+TOP+pred*TOP;int best=0;
        for(int j=1;j<TOP;j++)if(scores[j]>scores[best])best=j;
        ids[t-1]=int(row[best]);pred=best;}
}
} // namespace dn

struct MKDFlashNative::Impl {
    using Matrix=dn::Matrix;
    std::vector<void *> allocations;
    size_t bytes=0;
    cudaStream_t stream=nullptr;
    cudaEvent_t ready=nullptr;
    cudaEvent_t injection_begin=nullptr,injection_end=nullptr;
    cublasHandle_t blas=nullptr;
    std::array<cudaGraphExec_t,9> graphs{};
    std::array<dn::Layer,5> blocks{};
    std::vector<int> taps{6,20,34,48,62};
    Matrix embed{},head{},fc{},hidden{},prev{},next{};
    const float * enc_norm=nullptr,* out_norm=nullptr;
    int vocab=0;
    int context=0,mask=0,chunk=512,end=0,checkpoint_end=-1,last_n=0;
    bool use_graph=true,use_q8=true;
    bool typed_mmq=true;
    int mmq_rows=4,head_rows=4;
    int8_t * quantized_input=nullptr;float * quantized_scale=nullptr;
    float * feature_input=nullptr,* fused=nullptr,* fused_norm=nullptr;
    float * x=nullptr,* xn=nullptr,* temp=nullptr,* temp2=nullptr,* residual=nullptr;
    float * dynamic=nullptr,* fg=nullptr,* fu=nullptr,* act=nullptr;
    float * q=nullptr,* k=nullptr,* v=nullptr,* attn=nullptr;
    float * logits=nullptr,* gate=nullptr,* unary=nullptr,* packed=nullptr,* partial=nullptr;
    float2 * stats=nullptr,* top_part=nullptr;
    dn::Control * ctl=nullptr;
    int * proposal_ids=nullptr;
    __half * cache_k=nullptr,* cache_v=nullptr,* noise_k=nullptr,* noise_v=nullptr;
    __half * ck_k=nullptr,* ck_v=nullptr,* weight_half=nullptr,* input_half=nullptr;
    static constexpr size_t CACHE_ELEMENTS=size_t(dn::NL)*8*dn::CACHE*dn::HD;
    ~Impl() {
        if(stream)cudaStreamSynchronize(stream);
        for(auto g:graphs)if(g)cudaGraphExecDestroy(g);
        if(blas)cublasDestroy(blas);
        if(ready)cudaEventDestroy(ready);
        if(injection_begin)cudaEventDestroy(injection_begin);
        if(injection_end)cudaEventDestroy(injection_end);
        if(stream)cudaStreamDestroy(stream);
        for(auto p:allocations)cudaFree(p);
    }
    template<class T>T * allocate(size_t n) {
        void * p=nullptr;DC(cudaMalloc(&p,n*sizeof(T)));allocations.push_back(p);bytes+=n*sizeof(T);return static_cast<T *>(p);
    }
    Matrix load_matrix(const gguf::File & file,const std::string & name,int K,int N) {
        const auto & t=file.get(name);
        if(t.ne!=std::vector<uint64_t>{uint64_t(K),uint64_t(N)} || (t.type!=12&&t.type!=14))
            throw std::runtime_error("native draft unsupported matrix: "+name);
        auto * p=allocate<uint8_t>(t.nbytes);DC(cudaMemcpy(p,t.data,t.nbytes,cudaMemcpyHostToDevice));
        return {p,K,N,int(t.type),int(gguf::type_row_bytes(t.type,K))};
    }
    const float * load_float(const gguf::File & file,const std::string & name,size_t count) {
        const auto & t=file.get(name);
        if(t.type!=0 || t.nbytes!=count*4)throw std::runtime_error("native draft unsupported norm/conv: "+name);
        auto * p=allocate<float>(count);DC(cudaMemcpy(p,t.data,t.nbytes,cudaMemcpyHostToDevice));return p;
    }
    template<int T>void small_mm(Matrix w,const float * src,float * dst) {
        if(use_q8){
            dn::quant_input<<<(T*w.K/32+3)/4,128,0,stream>>>(src,quantized_input,quantized_scale,w.K,T);
            if(typed_mmq) {
                const int requested=w.w==head.w ? head_rows : mmq_rows;
                // Condensed vocabularies may be even without being divisible by four.
                const int rows=w.N%4 ? 2 : requested;
                if(rows==4) {
                    if(w.type==12)dn::mmq_typed<T,12,4><<<(w.N+15)/16,128,0,stream>>>(w,quantized_input,quantized_scale,dst);
                    else dn::mmq_typed<T,14,4><<<(w.N+15)/16,128,0,stream>>>(w,quantized_input,quantized_scale,dst);
                } else {
                    if(w.type==12)dn::mmq_typed<T,12,2><<<(w.N+7)/8,128,0,stream>>>(w,quantized_input,quantized_scale,dst);
                    else dn::mmq_typed<T,14,2><<<(w.N+7)/8,128,0,stream>>>(w,quantized_input,quantized_scale,dst);
                }
            } else dn::mmq<T><<<(w.N+7)/8,128,0,stream>>>(w,quantized_input,quantized_scale,dst);
        }else dn::mm<T><<<(w.N+7)/8,128,0,stream>>>(w,src,dst);
    }
    void trace(const char * name,int layer,const float * data,size_t count) {
        const char * dir=std::getenv("MK_DFLASH_TRACE");if(!dir)return;
        DC(cudaStreamSynchronize(stream));std::vector<float> f(count);
        DC(cudaMemcpy(f.data(),data,count*4,cudaMemcpyDeviceToHost));
        std::string key=name;if(layer>=0)key+="-"+std::to_string(layer);
        std::ofstream out(std::string(dir)+"/native-"+key+".f32",std::ios::binary);
        out.write(reinterpret_cast<const char *>(f.data()),count*4);
    }
    void trace_cache(int layer) {
        const char * dir=std::getenv("MK_DFLASH_TRACE");if(!dir)return;
        DC(cudaStreamSynchronize(stream));
        for(int side=0;side<2;side++) {
            const auto * src=(side?cache_v:cache_k)+size_t(layer)*8*dn::CACHE*dn::HD;
            std::vector<__half> h(8*dn::CACHE*dn::HD);DC(cudaMemcpy(h.data(),src,h.size()*2,cudaMemcpyDeviceToHost));
            std::vector<float> f(size_t(end)*dn::KD);
            for(int t=0;t<end;t++)for(int head0=0;head0<8;head0++)for(int e=0;e<dn::HD;e++)
                f[t*dn::KD+head0*dn::HD+e]=__half2float(h[(size_t(head0)*dn::CACHE+t%dn::CACHE)*dn::HD+e]);
            std::ofstream out(std::string(dir)+(side?"/native-history-V-":"/native-history-K-")+std::to_string(layer)+".f32",std::ios::binary);
            out.write(reinterpret_cast<const char *>(f.data()),f.size()*4);
        }
    }
    void multiply(Matrix w,const float * src,float * dst,int n) {
        switch(n) {
        case 1:small_mm<1>(w,src,dst);return;case 2:small_mm<2>(w,src,dst);return;
        case 3:small_mm<3>(w,src,dst);return;case 4:small_mm<4>(w,src,dst);return;
        case 5:small_mm<5>(w,src,dst);return;case 6:small_mm<6>(w,src,dst);return;
        case 7:small_mm<7>(w,src,dst);return;case 8:small_mm<8>(w,src,dst);return;
        }
        if(n<1||n>chunk)throw std::runtime_error("native draft matrix batch outside workspace");
        if(!weight_half) {
            weight_half=allocate<__half>(size_t(25600)*5120);
            input_half=allocate<__half>(size_t(chunk)*25600);
        }
        dn::dequant_half<<<(size_t(w.K)*w.N+2047)/2048,256,0,stream>>>(w,weight_half);
        dn::to_half<<<(size_t(w.K)*n+255)/256,256,0,stream>>>(src,input_half,size_t(w.K)*n);
        const float alpha=1,beta=0;
        auto status=cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,w.N,n,w.K,&alpha,
            weight_half,CUDA_R_16F,w.K,input_half,CUDA_R_16F,w.K,&beta,dst,CUDA_R_32F,w.N,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("native draft cuBLAS failed: "+std::to_string(int(status)));
    }
    void run(int n) {
        using namespace dn;
        cudaMemsetAsync(packed,0,MAXT*D*4,stream);
        embed_noise(n);
        trace("inp_noise_embd",-1,x,n*D);
        for(int i=0;i<NL;i++) {
            const auto & b=blocks[i];
            trace_cache(i);
            norm<<<n,256,0,stream>>>(x,b.an,xn,D,n);
            trace("noise_norm",i,xn,n*D);
            multiply(b.ac,xn,dynamic,n);
            conv<<<(n*D+255)/256,256,0,stream>>>(xn,dynamic,b.ab,temp,n,0);
            trace("attn_conv_in",i,temp,n*D);
            multiply(b.q,temp,q,n);multiply(b.k,temp,k,n);multiply(b.v,temp,v,n);
            query_rope<<<n*32,HD,0,stream>>>(q,b.qn,n,ctl);
            cache_kv<<<n*8,HD,0,stream>>>(k,v,b.kn,noise_k,noise_v,0,n,ctl,true);
            trace("Qcur",i,q,n*QD);trace("Vcur",i,v,n*KD);
            // Fixed split count makes graphs reusable for any position; empty
            // chunks have (-inf,0) stats and do not affect the combine.
            constexpr int chunks=8;
            attention<<<dim3((n*32+7)/8,chunks),256,0,stream>>>(q,
                cache_k+size_t(i)*8*CACHE*HD,cache_v+size_t(i)*8*CACHE*HD,
                noise_k,noise_v,partial,stats,n,chunks,ctl);
            attention_combine<<<(n*QD+255)/256,256,0,stream>>>(partial,stats,attn,n,chunks);
            trace("native_attention",i,attn,n*QD);
            multiply(b.o,attn,temp2,n);
            trace("attn_out",i,temp2,n*D);
            conv<<<(n*D+255)/256,256,0,stream>>>(temp2,dynamic,b.ab,temp,n,1);
            trace("attn_conv_out",i,temp,n*D);
            add<<<(n*D+255)/256,256,0,stream>>>(x,temp,residual,n*D);
            trace("ffn_inp",i,residual,n*D);
            norm<<<n,256,0,stream>>>(residual,b.fn,xn,D,n);
            multiply(b.fc,xn,dynamic,n);
            conv<<<(n*D+255)/256,256,0,stream>>>(xn,dynamic,b.fb,temp,n,0);
            multiply(b.gate,temp,fg,n);multiply(b.up,temp,fu,n);
            trace("ffn_gate",i,fg,n*FF);trace("ffn_up",i,fu,n*FF);
            glu<<<(n*FF+255)/256,256,0,stream>>>(fg,fu,act,n*FF);
            trace("ffn_swiglu",i,act,n*FF);
            multiply(b.down,act,temp2,n);
            conv<<<(n*D+255)/256,256,0,stream>>>(temp2,dynamic,b.fb,temp,n,1);
            add<<<(n*D+255)/256,256,0,stream>>>(residual,temp,x,n*D);
            trace("l_out",i,x,n*D);
        }
        norm<<<n,256,0,stream>>>(x,out_norm,xn,D,n);
        trace("result_norm",-1,xn,n*D);
        multiply(head,xn+D,logits,n-1);multiply(hidden,xn,gate,n);
        const int nb=(vocab+255)/256;
        top_partial<<<dim3(nb,n-1),256,0,stream>>>(logits,top_part,nb,vocab);
        top_finish<<<n-1,256,0,stream>>>(top_part,packed,unary,nb,vocab);
        selector<<<dim3(TOP*TOP,n-1),256,0,stream>>>(prev,next,gate,unary,packed,ctl);
        walk<<<1,1,0,stream>>>(packed,proposal_ids,n-1);
    }
    void embed_noise(int n) { dn::embed<<<(n*dn::D+255)/256,256,0,stream>>>(embed,x,n,mask,ctl); }
};

MKDFlashNative::MKDFlashNative(MKEngine & target,const std::string & path,int context,int mask,int chunk):p_(new Impl) {
    using namespace dn;
    const int vocab = target.vocab_size();
    p_->vocab=vocab;
    if(context<16||mask<0||mask>=vocab||chunk<8||chunk>2048)throw std::runtime_error("invalid native draft configuration");
    p_->context=context;p_->mask=mask;p_->chunk=chunk;
    if(const char * s=std::getenv("MK_DFLASH_GRAPH"))p_->use_graph=std::strcmp(s,"0")!=0;
    if(std::getenv("MK_DFLASH_TRACE"))p_->use_graph=false;
    if(const char * s=std::getenv("MK_DFLASH_MMQ_TYPED"))p_->typed_mmq=std::strcmp(s,"0")!=0;
    for(auto setting:{std::pair<const char *,int *>{"MK_DFLASH_MMQ_ROWS",&p_->mmq_rows},
                     {"MK_DFLASH_MMQ_HEAD_ROWS",&p_->head_rows}}) {
        if(const char * s=std::getenv(setting.first)) {
            if(std::strcmp(s,"2")&&std::strcmp(s,"4"))throw std::runtime_error(std::string(setting.first)+" must be 2 or 4");
            *setting.second=std::atoi(s);
        }
    }
    if(const char * s=std::getenv("MK_DFLASH_MATH")) {
        if(std::strcmp(s,"q8")&&std::strcmp(s,"f32"))throw std::runtime_error("MK_DFLASH_MATH must be q8 or f32");
        p_->use_q8=std::strcmp(s,"q8")==0;
    }
    gguf::File file(path);
    if(file.strs.at("general.architecture")!="dflash"||file.tensors.size()!=81)
        throw std::runtime_error("native draft expects the validated 81-tensor DFlash2 schema");
    if(file.array_sizes.at("tokenizer.ggml.tokens")!=size_t(vocab))
        throw std::runtime_error("native draft vocabulary size differs; prepare a remapped condensed draft first");
    for(auto pair:{std::pair<const char *,uint64_t>{"dflash.embedding_length",D},
        {"dflash.block_count",NL},{"dflash.feed_forward_length",FF},{"dflash.block_size",8},
        {"dflash.attention.head_count",32},{"dflash.attention.head_count_kv",8},
        {"dflash.attention.key_length",HD},{"dflash.attention.value_length",HD},
        {"dflash.conv_kernel_size",2},{"dflash.conv_group_size",16},
        {"dflash.selector_rank",RANK},{"dflash.selector_top_k",TOP},{"dflash.attention.sliding_window",WINDOW}})
        if(file.ints.at(pair.first)!=pair.second)throw std::runtime_error(std::string("native draft metadata mismatch: ")+pair.first);
    if(file.arrays.at("dflash.target_layers")!=std::vector<uint64_t>{6,20,34,48,62} ||
       file.arrays.at("dflash.attention.sliding_window_pattern")!=std::vector<uint64_t>{1,1,1,1,1} ||
       std::abs(file.reals.at("dflash.attention.layer_norm_rms_epsilon")-1e-6)>1e-12 ||
       file.arrays.at("dflash.rope.dimension_sections")!=std::vector<uint64_t>{64,0,0,0} ||
       file.reals.at("dflash.rope.freq_base")!=10000000. || file.ints.at("dflash.attention.causal")!=0)
        throw std::runtime_error("native draft RoPE/tap/mask metadata mismatch");
    DC(cudaStreamCreateWithFlags(&p_->stream,cudaStreamNonBlocking));
    DC(cudaEventCreateWithFlags(&p_->ready,cudaEventDisableTiming));
    DC(cudaEventCreate(&p_->injection_begin));DC(cudaEventCreate(&p_->injection_end));
    if(cublasCreate(&p_->blas)!=CUBLAS_STATUS_SUCCESS||cublasSetStream(p_->blas,p_->stream)!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("native draft cuBLAS initialization failed");
    const auto shared=target.share_weights();
    if(shared.embedding->type!=GGML_TYPE_IQ2_S||shared.output->type!=GGML_TYPE_Q4_K)
        throw std::runtime_error("native draft unsupported shared weights");
    p_->embed={static_cast<const uint8_t *>(shared.embedding->data),D,vocab,22,int(gguf::type_row_bytes(22,D))};
    p_->head={static_cast<const uint8_t *>(shared.output->data),D,vocab,12,int(gguf::type_row_bytes(12,D))};
    p_->fc=p_->load_matrix(file,"fc.weight",5*D,D);
    p_->hidden=p_->load_matrix(file,"selector_hidden.weight",D,RANK);
    p_->prev=p_->load_matrix(file,"selector_predecessor.weight",RANK,vocab);
    p_->next=p_->load_matrix(file,"selector_successor.weight",RANK,vocab);
    p_->enc_norm=p_->load_float(file,"enc.output_norm.weight",D);
    p_->out_norm=p_->load_float(file,"output_norm.weight",D);
    for(int i=0;i<NL;i++) {
        auto name=[i](const char * s){return "blk."+std::to_string(i)+"."+s;};auto & b=p_->blocks[i];
        b.q=p_->load_matrix(file,name("attn_q.weight"),D,QD);
        b.k=p_->load_matrix(file,name("attn_k.weight"),D,KD);b.v=p_->load_matrix(file,name("attn_v.weight"),D,KD);
        b.o=p_->load_matrix(file,name("attn_output.weight"),QD,D);
        b.gate=p_->load_matrix(file,name("ffn_gate.weight"),D,FF);b.up=p_->load_matrix(file,name("ffn_up.weight"),D,FF);
        b.down=p_->load_matrix(file,name("ffn_down.weight"),FF,D);
        b.ac=p_->load_matrix(file,name("attn_conv_proj.weight"),D,1280);b.fc=p_->load_matrix(file,name("ffn_conv_proj.weight"),D,1280);
        b.an=p_->load_float(file,name("attn_norm.weight"),D);b.fn=p_->load_float(file,name("ffn_norm.weight"),D);
        b.qn=p_->load_float(file,name("attn_q_norm.weight"),HD);b.kn=p_->load_float(file,name("attn_k_norm.weight"),HD);
        b.ab=p_->load_float(file,name("attn_conv_base"),4*D);b.fb=p_->load_float(file,name("ffn_conv_base"),4*D);
    }
    auto small=[&](int width){return p_->allocate<float>(MAXT*width);};
    p_->fused=p_->allocate<float>(size_t(chunk)*D);p_->fused_norm=p_->allocate<float>(size_t(chunk)*D);
    p_->x=small(D);p_->xn=small(D);p_->temp=small(D);p_->temp2=small(D);p_->residual=small(D);
    p_->dynamic=small(1280);p_->fg=small(FF);p_->fu=small(FF);p_->act=small(FF);
    p_->q=small(QD);p_->k=p_->allocate<float>(size_t(chunk)*KD);p_->v=p_->allocate<float>(size_t(chunk)*KD);p_->attn=small(QD);
    p_->logits=small(vocab);p_->gate=small(RANK);p_->unary=small(TOP);p_->packed=small(D);
    p_->partial=p_->allocate<float>(MAXT*32*8*HD);p_->stats=p_->allocate<float2>(MAXT*32*8);
    p_->top_part=p_->allocate<float2>(size_t(MAXT)*((vocab+255)/256)*TOP);
    p_->proposal_ids=p_->allocate<int>(7);p_->ctl=p_->allocate<Control>(1);
    p_->cache_k=p_->allocate<__half>(Impl::CACHE_ELEMENTS);p_->cache_v=p_->allocate<__half>(Impl::CACHE_ELEMENTS);
    p_->noise_k=p_->allocate<__half>(MAXT*KD);p_->noise_v=p_->allocate<__half>(MAXT*KD);
    p_->quantized_input=p_->allocate<int8_t>(MAXT*25600);p_->quantized_scale=p_->allocate<float>(MAXT*25600/32);
    fprintf(stderr,"[mk-dflash-native] CUDA-only draft %.3f GiB allocations; sliding KV FP16 %d rows; graphs %s; math %s\n",
        p_->bytes/double(1ull<<30),CACHE,p_->use_graph?"on":"off",p_->use_q8?"DP4A/Q8":"F32");
}
MKDFlashNative::~MKDFlashNative()=default;
const std::vector<int> & MKDFlashNative::layers() const{return p_->taps;}
size_t MKDFlashNative::allocated_bytes() const{return p_->bytes;}
void MKDFlashNative::reset(){p_->end=0;p_->checkpoint_end=-1;p_->last_n=0;injection_seconds=0;draft_seconds=0;injection_gpu_seconds=0;}
void MKDFlashNative::inject(int pos,int n,const std::vector<float> & f) {
    if(n<1||f.size()!=size_t(n)*5*dn::D)throw std::runtime_error("native draft invalid host features");
    if(!p_->feature_input)p_->feature_input=p_->allocate<float>(size_t(p_->chunk)*5*dn::D);
    for(int off=0;off<n;off+=p_->chunk) {
        const int count=std::min(p_->chunk,n-off);
        DC(cudaMemcpy(p_->feature_input,f.data()+size_t(off)*5*dn::D,size_t(count)*5*dn::D*4,cudaMemcpyHostToDevice));
        inject_device(pos+off,count,p_->feature_input);
    }
}
void MKDFlashNative::inject_device(int pos,int n,const float * f) {
    using namespace dn;
    if(pos<0||n<1||!f||pos>p_->end||pos+n>p_->context||p_->end-pos>MAXT)
        throw std::runtime_error("native draft invalid feature position/gap");
    const auto start=std::chrono::steady_clock::now();
    DC(cudaEventRecord(p_->ready,0));DC(cudaStreamWaitEvent(p_->stream,p_->ready,0));
    DC(cudaEventRecord(p_->injection_begin,p_->stream));
    for(int off=0;off<n;off+=p_->chunk) {
        const int count=std::min(p_->chunk,n-off);
        p_->multiply(p_->fc,f+size_t(off)*5*D,p_->fused,count);
        norm<<<count,256,0,p_->stream>>>(p_->fused,p_->enc_norm,p_->fused_norm,D,count);
        p_->trace("inp_g_embeddings",-1,p_->fused_norm,size_t(count)*D);
        for(int i=0;i<NL;i++) {
            const auto & b=p_->blocks[i];
            p_->multiply(b.k,p_->fused_norm,p_->k,count);p_->multiply(b.v,p_->fused_norm,p_->v,count);
            cache_kv<<<count*8,HD,0,p_->stream>>>(p_->k,p_->v,b.kn,
                p_->cache_k+size_t(i)*8*CACHE*HD,p_->cache_v+size_t(i)*8*CACHE*HD,pos+off,count,p_->ctl,false);
        }
    }
    DC(cudaEventRecord(p_->injection_end,p_->stream));
    DC(cudaGetLastError());DC(cudaStreamSynchronize(p_->stream));p_->end=pos+n;
    float ms=0;DC(cudaEventElapsedTime(&ms,p_->injection_begin,p_->injection_end));injection_gpu_seconds+=ms*.001;
    injection_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
std::vector<int> MKDFlashNative::draft(int pos,int anchor,int depth) {
    if(pos!=p_->end||pos<0||depth<1||depth>7||pos+depth>=p_->context||anchor<0||anchor>=p_->vocab)
        throw std::runtime_error("native draft invalid position/token/depth");
    const auto start=std::chrono::steady_clock::now();const int n=depth+1;
    const dn::Control ctl{pos,anchor};DC(cudaMemcpyAsync(p_->ctl,&ctl,sizeof(ctl),cudaMemcpyHostToDevice,p_->stream));
    if(p_->use_graph) {
        if(!p_->graphs[n]) {
            cudaGraph_t graph=nullptr;DC(cudaStreamBeginCapture(p_->stream,cudaStreamCaptureModeThreadLocal));
            p_->run(n);DC(cudaStreamEndCapture(p_->stream,&graph));
            auto err=cudaGraphInstantiate(&p_->graphs[n],graph,0);cudaGraphDestroy(graph);DC(err);
        }
        DC(cudaGraphLaunch(p_->graphs[n],p_->stream));
    } else p_->run(n);
    std::vector<int> ids(depth);DC(cudaMemcpyAsync(ids.data(),p_->proposal_ids,depth*4,cudaMemcpyDeviceToHost,p_->stream));
    DC(cudaGetLastError());DC(cudaStreamSynchronize(p_->stream));
    for(int id:ids)if(id<0||id>=p_->vocab)throw std::runtime_error("native draft selector returned invalid ID");
    p_->last_n=n;draft_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return ids;
}
std::vector<float> MKDFlashNative::lattice() const {
    std::vector<float> f(size_t(p_->last_n)*dn::D);if(!f.empty())DC(cudaMemcpy(f.data(),p_->packed,f.size()*4,cudaMemcpyDeviceToHost));return f;
}
void MKDFlashNative::checkpoint() {
    if(!p_->ck_k){p_->ck_k=p_->allocate<__half>(Impl::CACHE_ELEMENTS);p_->ck_v=p_->allocate<__half>(Impl::CACHE_ELEMENTS);}
    DC(cudaMemcpyAsync(p_->ck_k,p_->cache_k,Impl::CACHE_ELEMENTS*2,cudaMemcpyDeviceToDevice,p_->stream));
    DC(cudaMemcpyAsync(p_->ck_v,p_->cache_v,Impl::CACHE_ELEMENTS*2,cudaMemcpyDeviceToDevice,p_->stream));
    DC(cudaStreamSynchronize(p_->stream));p_->checkpoint_end=p_->end;
}
bool MKDFlashNative::restore_checkpoint() {
    if(p_->checkpoint_end<0)return false;
    DC(cudaMemcpyAsync(p_->cache_k,p_->ck_k,Impl::CACHE_ELEMENTS*2,cudaMemcpyDeviceToDevice,p_->stream));
    DC(cudaMemcpyAsync(p_->cache_v,p_->ck_v,Impl::CACHE_ELEMENTS*2,cudaMemcpyDeviceToDevice,p_->stream));
    DC(cudaStreamSynchronize(p_->stream));p_->end=p_->checkpoint_end;return true;
}
void MKDFlashNative::truncate(int pos) {
    if(pos<0||p_->end-pos>dn::MAXT)throw std::runtime_error("native draft truncate outside preserved history");
    p_->end=std::min(p_->end,pos);
}
