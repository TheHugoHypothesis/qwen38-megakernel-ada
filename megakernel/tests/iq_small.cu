// T=4 ordered gate/up fusion: actual model bytes, legacy path, CPU GGML/double.
#ifndef MK_IQ2_DIRECT
#define MK_IQ2_DIRECT 1
#endif
#include "../native_iq_mma.cuh"
#include "../gguf.h"
#include "ggml.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>

#define CHECK(x) do { auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); } while(0)
constexpr int TOKENS=4,UNITS=160,K=5120,ROWS=16;
template<int GT,int UT,bool FUSED>
__global__ void dots(const uint8_t * g,int gs,const uint8_t * u,int us,const int4 * a,const int4 * b,const uint2 * m,float * out) {
    __shared__ int4 sa[TOKENS*UNITS],sb[TOKENS*UNITS];
    __shared__ uint2 sm[TOKENS*UNITS];
    for(int i=threadIdx.x;i<TOKENS*UNITS;i+=blockDim.x){sa[i]=a[i];sb[i]=b[i];sm[i]=m[i];}
    __syncthreads();
    const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    float gate[TOKENS]{},up[TOKENS]{};
    g+=size_t(row)*gs;u+=size_t(row)*us;
    if constexpr(FUSED)iq_dp4a_dual<TOKENS,UNITS,GT,UT,true,true,true>(g,u,sa,sb,sm,gate,up);
    else {
        // Original per-matrix ordered dot: 16-value path for IQ3/4,
        // 8-value path for IQ2 encodings, without gate/up load sharing.
        for(int side=0;side<2;side++) {
            const uint8_t * w=side?u:g;const int kind=side?UT:GT;
            float * acc=side?up:gate;
            #pragma unroll 1
            for(int i=0;i<UNITS/32;i++) {
                const int unit=lane+32*i,e=256*(unit/8)+64*((unit/2)&3)+16*(unit&1);
                for(int part=0;part<2;part++) {
                    if(kind==18||kind==21||kind==23) {
                        Quant16 w16;
                        if(side){w16=native_iq16<UT>(w,e+32*part);}else{w16=native_iq16<GT>(w,e+32*part);}
                        for(int t=0;t<TOKENS;t++) {
                            int ix=t*UNITS+unit;const int4 x=part?sb[ix]:sa[ix];
                            const float2 scale=__half22float2(*reinterpret_cast<const __half2 *>(&sm[ix].x));
                            const int d=__dp4a(w16.q.w,x.w,__dp4a(w16.q.z,x.z,__dp4a(w16.q.y,x.y,__dp4a(w16.q.x,x.x,0))));
                            acc[t]+=(part?scale.y:scale.x)*w16.d*d;
                        }
                    } else for(int half=0;half<2;half++) {
                        const Quant8 w8=quant8(kind,w,e+32*part+8*half);
                        for(int t=0;t<TOKENS;t++) {
                            int ix=t*UNITS+unit;const int4 x=part?sb[ix]:sa[ix];
                            const int x0=half?x.z:x.x,x1=half?x.w:x.y;
                            const float2 scale=__half22float2(*reinterpret_cast<const __half2 *>(&sm[ix].x));
                            const int sum=__dp4a(x0,0x01010101,__dp4a(x1,0x01010101,0));
                            const float dot=__fmaf_rn(w8.d,float(__dp4a(w8.q.x,x0,__dp4a(w8.q.y,x1,0))),w8.offset*sum);
                            acc[t]=__fadd_rn(acc[t],__fmul_rn(part?scale.y:scale.x,dot));
                        }
                    }
                }
            }
        }
    }
    for(int t=0;t<TOKENS;t++) {
        for(int off=16;off;off/=2){gate[t]+=__shfl_xor_sync(0xffffffff,gate[t],off);up[t]+=__shfl_xor_sync(0xffffffff,up[t],off);}
        if(lane==0){out[(t*ROWS+row)*2]=gate[t];out[(t*ROWS+row)*2+1]=up[t];}
    }
}
template<int GT>void dispatch_up(int ut,bool fused,const uint8_t * g,int gs,const uint8_t * u,int us,const int4 * a,const int4 * b,const uint2 * m,float * out) {
#define CASE(UT) case UT: if(fused)dots<GT,UT,true><<<ROWS/4,128>>>(g,gs,u,us,a,b,m,out);else dots<GT,UT,false><<<ROWS/4,128>>>(g,gs,u,us,a,b,m,out);break;
    switch(ut){CASE(16) CASE(17) CASE(18) CASE(21) CASE(22) CASE(23)}
#undef CASE
}
void launch(int gt,int ut,bool fused,const uint8_t * g,int gs,const uint8_t * u,int us,const int4 * a,const int4 * b,const uint2 * m,float * out) {
#define CASE(GT) case GT:dispatch_up<GT>(ut,fused,g,gs,u,us,a,b,m,out);break;
    switch(gt){CASE(16) CASE(17) CASE(18) CASE(21) CASE(22) CASE(23)}
#undef CASE
}
int main(int argc,char ** argv) {
 try {
    if(argc!=2)throw std::runtime_error("usage: iq_small_test model.gguf");
    gguf::File file(argv[1]);
    struct Weight{int kind,stride;uint8_t * device;std::vector<float> decoded;};
    std::vector<Weight> weights;
    for(int kind:{16,17,18,21,22,23}) {
        const gguf::Tensor * tensor=nullptr;
        for(const auto & x:file.tensors)if(x.second.type==unsigned(kind)&&x.second.ne.size()==2&&x.second.ne[0]==K&&x.second.ne[1]>=ROWS){tensor=&x.second;break;}
        if(!tensor)throw std::runtime_error("source lacks required real quantization format");
        const int stride=gguf::type_row_bytes(kind,K);Weight w{kind,stride,nullptr,std::vector<float>(ROWS*K)};
        CHECK(cudaMalloc(&w.device,ROWS*stride));CHECK(cudaMemcpy(w.device,tensor->data,ROWS*stride,cudaMemcpyHostToDevice));
        ggml_get_type_traits(ggml_type(kind))->to_float(tensor->data,w.decoded.data(),ROWS*K);weights.push_back(std::move(w));
    }
    int4 * a,* b;uint2 * m;float * out;
    CHECK(cudaMalloc(&a,TOKENS*UNITS*16));CHECK(cudaMalloc(&b,TOKENS*UNITS*16));CHECK(cudaMalloc(&m,TOKENS*UNITS*8));CHECK(cudaMalloc(&out,TOKENS*ROWS*2*4));
    size_t exact=0;double worst=0,max_abs=0;
    for(int trial=0;trial<3;trial++) {
        std::vector<int4> ha(TOKENS*UNITS),hb(ha.size());std::vector<uint2> hm(ha.size());
        std::vector<float> decoded(TOKENS*K);
        for(int t=0;t<TOKENS;t++)for(int unit=0;unit<UNITS;unit++) {
            const int ix=t*UNITS+unit,e=256*(unit/8)+64*((unit/2)&3)+16*(unit&1);
            const __half2 dh=__floats2half2_rn(trial?.003f*(1+(unit+t)%7):0,trial?.007f*(1+(unit+2*t)%5):0);
            std::memcpy(&hm[ix].x,&dh,4);hm[ix].y=0;
            const float2 scale=__half22float2(dh);
            for(int j=0;j<16;j++) {
                const int8_t va=trial?int8_t((e+j*11+t*3+trial)%255-127):0,vb=trial?int8_t((e+j*7+t*5+trial)%255-127):0;
                reinterpret_cast<int8_t *>(&ha[ix])[j]=va;reinterpret_cast<int8_t *>(&hb[ix])[j]=vb;
                decoded[t*K+e+j]=scale.x*va;decoded[t*K+e+32+j]=scale.y*vb;
            }
        }
        CHECK(cudaMemcpy(a,ha.data(),ha.size()*16,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,hb.data(),hb.size()*16,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(m,hm.data(),hm.size()*8,cudaMemcpyHostToDevice));
        for(const auto & g:weights)for(const auto & u:weights) {
            std::vector<float> legacy(TOKENS*ROWS*2),actual(legacy.size());
            launch(g.kind,u.kind,false,g.device,g.stride,u.device,u.stride,a,b,m,out);CHECK(cudaGetLastError());CHECK(cudaMemcpy(legacy.data(),out,legacy.size()*4,cudaMemcpyDeviceToHost));
            launch(g.kind,u.kind,true,g.device,g.stride,u.device,u.stride,a,b,m,out);CHECK(cudaGetLastError());CHECK(cudaMemcpy(actual.data(),out,actual.size()*4,cudaMemcpyDeviceToHost));
            double err=0,ref2=0;
            for(int t=0;t<TOKENS;t++)for(int r=0;r<ROWS;r++)for(int side=0;side<2;side++) {
                const int ix=(t*ROWS+r)*2+side;
                if(!std::isfinite(actual[ix])||actual[ix]!=legacy[ix])throw std::runtime_error("T4 fused dot differs from ordered legacy");
                exact++;
                double ref=0;const auto & w=side?u:g;
                for(int k=0;k<K;k++)ref+=double(w.decoded[r*K+k])*decoded[t*K+k];
                const double delta=actual[ix]-ref;err+=delta*delta;ref2+=ref*ref;max_abs=std::max(max_abs,std::abs(delta));
                if(std::abs(delta)>1e-4+2e-5*std::abs(ref))throw std::runtime_error("T4 dot exceeds CPU GGML/double tolerance");
            }
            worst=std::max(worst,ref2?std::sqrt(err/ref2):std::sqrt(err));
        }
    }
    std::printf("PASS %zu exact ordered T4 dots, 36 IQ pairs x 3 inputs; max NRMSE %.9g abs %.9g\n",exact,worst,max_abs);
    for(auto & w:weights)CHECK(cudaFree(w.device));CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(m));CHECK(cudaFree(out));
 }catch(const std::exception & e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
