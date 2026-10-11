// Independent GGML CPU dequantization and double-precision matmul oracle.
#include "../dflash_native.cu"
#include "ggml-impl.h"
#include <fstream>

__global__ void inspect_weights(dn::Matrix w,float * y) {
    const int i=blockIdx.x*256+threadIdx.x;if(i<w.K*w.N)y[i]=dn::weight(w,i/w.K,i%w.K);
}
static void require(bool good,const char * msg){if(!good)throw std::runtime_error(msg);}
template<int T,int ROWS> static void typed_q8(dn::Matrix m,const int8_t * q,const float * scale,float * out) {
    if(m.type==12)dn::mmq_typed<T,12,ROWS><<<(m.N+4*ROWS-1)/(4*ROWS),128>>>(m,q,scale,out);
    else dn::mmq_typed<T,14,ROWS><<<(m.N+4*ROWS-1)/(4*ROWS),128>>>(m,q,scale,out);
}
int main(int argc,char ** argv) {
 try {
    require(argc==2,"usage: dflash_math_test draft.gguf");gguf::File file(argv[1]);
    size_t values=0,dots=0,typed_values=0;double max_weight=0,max_mm=0;
    for(const auto & item:file.tensors) {
        const auto & t=item.second;if(t.ne.size()!=2||(t.type!=12&&t.type!=14))continue;
        const int K=t.ne[0],N=t.ne[1],rb=gguf::type_row_bytes(t.type,K);
        std::vector<uint8_t> bytes(size_t(4)*rb);std::vector<float> decoded(size_t(4)*K);
        const int rows[4]={0,N/3,2*N/3,N-1};
        for(int r=0;r<4;r++) {
            std::memcpy(bytes.data()+size_t(r)*rb,t.data+size_t(rows[r])*rb,rb);
            ggml_get_type_traits(ggml_type(t.type))->to_float(bytes.data()+size_t(r)*rb,decoded.data()+size_t(r)*K,K);
        }
        uint8_t * w=nullptr;float * y=nullptr,*x=nullptr;
        DC(cudaMalloc(&w,bytes.size()));DC(cudaMemcpy(w,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
        DC(cudaMalloc(&y,std::max(size_t(4)*K,size_t(32))*4));DC(cudaMalloc(&x,size_t(8)*K*4));
        dn::Matrix m{w,K,4,int(t.type),rb};inspect_weights<<<(4*K+255)/256,256>>>(m,y);
        std::vector<float> actual(size_t(4)*K);DC(cudaMemcpy(actual.data(),y,actual.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<actual.size();i++) {
            max_weight=std::max(max_weight,double(std::abs(actual[i]-decoded[i])));
            require(std::isfinite(actual[i])&&std::abs(actual[i]-decoded[i])<=1e-5f+1e-5f*std::abs(decoded[i]),"weight reader differs from CPU GGML");
        }
        __half * halfweights;DC(cudaMalloc(&halfweights,decoded.size()*2));
        dn::dequant_half<<<(decoded.size()+2047)/2048,256>>>(m,halfweights);
        std::vector<__half> halfcpu(decoded.size());DC(cudaMemcpy(halfcpu.data(),halfweights,halfcpu.size()*2,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<decoded.size();i++)require(__half2float(halfcpu[i])==__half2float(__float2half(decoded[i])),"vector dequant differs from CPU FP16 staging");
        cudaFree(halfweights);
        values+=actual.size();std::vector<float> inputs(size_t(8)*K);
        for(size_t i=0;i<inputs.size();i++)inputs[i]=std::sin(double(i)*.017)*.3;
        DC(cudaMemcpy(x,inputs.data(),inputs.size()*4,cudaMemcpyHostToDevice));
        for(int n:{1,4,8}) {
            if(n==1)dn::mm<1><<<1,128>>>(m,x,y);
            if(n==4)dn::mm<4><<<1,128>>>(m,x,y);
            if(n==8)dn::mm<8><<<1,128>>>(m,x,y);
            std::vector<float> result(n*4);DC(cudaMemcpy(result.data(),y,result.size()*4,cudaMemcpyDeviceToHost));
            for(int t0=0;t0<n;t0++)for(int r=0;r<4;r++) {
                double expected=0;for(int k=0;k<K;k++)expected+=double(decoded[size_t(r)*K+k])*inputs[size_t(t0)*K+k];
                const double error=std::abs(result[t0*4+r]-expected);max_mm=std::max(max_mm,error);
                require(std::isfinite(result[t0*4+r])&&error<3e-4+1e-4*std::abs(expected),"draft matmul differs from CPU oracle");dots++;
            }
        }
        int8_t * qx;float * qs;DC(cudaMalloc(&qx,size_t(8)*K));DC(cudaMalloc(&qs,size_t(8)*(K/32)*4));
        dn::quant_input<<<(8*K/32+3)/4,128>>>(x,qx,qs,K,8);
        std::vector<uint8_t> cpuq(size_t(8)*(K/32)*34);std::vector<float> cpux(inputs.size());
        auto traits=ggml_get_type_traits(GGML_TYPE_Q8_0);
        traits->from_float_ref(inputs.data(),cpuq.data(),inputs.size());traits->to_float(cpuq.data(),cpux.data(),inputs.size());
        for(int n:{1,4,7,8}) {
            if(n==1)dn::mmq<1><<<1,128>>>(m,qx,qs,y);
            if(n==4)dn::mmq<4><<<1,128>>>(m,qx,qs,y);
            if(n==7)dn::mmq<7><<<1,128>>>(m,qx,qs,y);
            if(n==8)dn::mmq<8><<<1,128>>>(m,qx,qs,y);
            std::vector<float> result(n*4);DC(cudaMemcpy(result.data(),y,result.size()*4,cudaMemcpyDeviceToHost));
            for(int t0=0;t0<n;t0++)for(int r=0;r<4;r++) {
                double expected=0;for(int k=0;k<K;k++)expected+=double(decoded[size_t(r)*K+k])*cpux[size_t(t0)*K+k];
                if(std::abs(result[t0*4+r]-expected)>=.002+5e-4*std::abs(expected)) {
                    fprintf(stderr,"Q8 %s t=%d r=%d actual=%g CPU=%g error=%g\n",t.name.c_str(),t0,r,result[t0*4+r],expected,std::abs(result[t0*4+r]-expected));
                    throw std::runtime_error("DP4A matmul differs from CPU GGML Q8 oracle");
                }
            }
            for(int rows:{2,4}) {
                if(rows==2) {
                    if(n==1)typed_q8<1,2>(m,qx,qs,y);
                    if(n==4)typed_q8<4,2>(m,qx,qs,y);
                    if(n==7)typed_q8<7,2>(m,qx,qs,y);
                    if(n==8)typed_q8<8,2>(m,qx,qs,y);
                } else {
                    if(n==1)typed_q8<1,4>(m,qx,qs,y);
                    if(n==4)typed_q8<4,4>(m,qx,qs,y);
                    if(n==7)typed_q8<7,4>(m,qx,qs,y);
                    if(n==8)typed_q8<8,4>(m,qx,qs,y);
                }
                std::vector<float> typed(n*4);
                DC(cudaMemcpy(typed.data(),y,typed.size()*4,cudaMemcpyDeviceToHost));
                for(size_t i=0;i<typed.size();i++) {
                    require(typed[i]==result[i],"typed/R4 MMQ changes the legacy DP4A result");
                    typed_values++;
                }
            }
        }
        cudaFree(qx);cudaFree(qs);
        cudaFree(w);cudaFree(x);cudaFree(y);
    }
    printf("PASS %zu typed R2/R4 Q8 values exactly match legacy DP4A, T1/4/7/8\n",typed_values);
    // Dynamic conv: includes left padding and both independently learned sides.
    std::vector<float> h(8*dn::D),dynamic(8*1280),base(4*dn::D);
    for(size_t i=0;i<h.size();i++)h[i]=std::sin(i*.001);
    for(size_t i=0;i<dynamic.size();i++)dynamic[i]=std::cos(i*.002)*.1;
    for(size_t i=0;i<base.size();i++)base[i]=std::sin(i*.003)*.2;
    float * gh,*gd,*gb,*gy;
    DC(cudaMalloc(&gh,h.size()*4));DC(cudaMalloc(&gd,dynamic.size()*4));DC(cudaMalloc(&gb,base.size()*4));DC(cudaMalloc(&gy,h.size()*4));
    DC(cudaMemcpy(gh,h.data(),h.size()*4,cudaMemcpyHostToDevice));DC(cudaMemcpy(gd,dynamic.data(),dynamic.size()*4,cudaMemcpyHostToDevice));DC(cudaMemcpy(gb,base.data(),base.size()*4,cudaMemcpyHostToDevice));
    for(int side=0;side<2;side++) {
        dn::conv<<<(8*dn::D+255)/256,256>>>(gh,gd,gb,gy,8,side);
        std::vector<float> got(h.size());DC(cudaMemcpy(got.data(),gy,got.size()*4,cudaMemcpyDeviceToHost));
        for(int t=0;t<8;t++)for(int e=0;e<dn::D;e++) {
            float v=h[t*dn::D+e]*(dynamic[t*1280+side*640+e/16]+base[side*2*dn::D+e]);
            if(t)v+=h[(t-1)*dn::D+e]*(dynamic[t*1280+side*640+320+e/16]+base[side*2*dn::D+dn::D+e]);
            require(std::abs(v-got[t*dn::D+e])<1e-6f,"dynamic conv differs from CPU oracle");
        }
    }
    cudaFree(gh);cudaFree(gd);cudaFree(gb);cudaFree(gy);
    // Cache writer independently checks all channels, both ring and noise.
    std::vector<float> rk(8*dn::KD),rv(rk.size()),rn(dn::HD);
    for(size_t i=0;i<rk.size();i++){rk[i]=std::sin(i*.003)+.1;rv[i]=std::cos(i*.013)*.2;}
    for(int e=0;e<dn::HD;e++)rn[e]=.9+e*.001;
    float * crk,*crv,*crn;__half * cck,*ccv;dn::Control * cc;
    const size_t cachecount=8*dn::CACHE*dn::HD;
    DC(cudaMalloc(&crk,rk.size()*4));DC(cudaMalloc(&crv,rv.size()*4));DC(cudaMalloc(&crn,rn.size()*4));DC(cudaMalloc(&cck,cachecount*2));DC(cudaMalloc(&ccv,cachecount*2));DC(cudaMalloc(&cc,sizeof(dn::Control)));
    DC(cudaMemcpy(crk,rk.data(),rk.size()*4,cudaMemcpyHostToDevice));DC(cudaMemcpy(crv,rv.data(),rv.size()*4,cudaMemcpyHostToDevice));DC(cudaMemcpy(crn,rn.data(),rn.size()*4,cudaMemcpyHostToDevice));
    const dn::Control ccpu{2050,123};DC(cudaMemcpy(cc,&ccpu,sizeof(ccpu),cudaMemcpyHostToDevice));double cache_error=0;
    for(bool noise:{false,true}) {
        dn::cache_kv<<<64,dn::HD>>>(crk,crv,crn,cck,ccv,2050,8,cc,noise);
        std::vector<__half> ak(cachecount),av(cachecount);DC(cudaMemcpy(ak.data(),cck,cachecount*2,cudaMemcpyDeviceToHost));DC(cudaMemcpy(av.data(),ccv,cachecount*2,cudaMemcpyDeviceToHost));
        for(int t=0;t<8;t++)for(int h0=0;h0<8;h0++) {
            double ss=0;for(int e=0;e<dn::HD;e++)ss+=double(rk[t*dn::KD+h0*dn::HD+e])*rk[t*dn::KD+h0*dn::HD+e];
            const double scale=1/std::sqrt(ss/dn::HD+1e-6);
            for(int e=0;e<dn::HD;e++) {
                const int j=e%64;double theta=(2050+t)*std::pow(1e7,-2.*j/128);
                const int other=e<64?e+64:e-64;double x=rk[t*dn::KD+h0*dn::HD+e]*rn[e]*scale,y=rk[t*dn::KD+h0*dn::HD+other]*rn[other]*scale;
                double expected=x*std::cos(theta)+(e<64?-1:1)*y*std::sin(theta);
                const size_t idx=(noise?(size_t(t)*8+h0)*dn::HD:(size_t(h0)*dn::CACHE+(2050+t)%dn::CACHE)*dn::HD)+e;
                double error=std::abs(__half2float(ak[idx])-expected);cache_error=std::max(cache_error,error);
                require(error<.004,"KV norm/RoPE writer differs from CPU oracle");
                require(__half2float(av[idx])==__half2float(__float2half(rv[t*dn::KD+h0*dn::HD+e])),"KV writer lost a value channel");
            }
        }
    }
    for(void * ptr:{(void*)crk,(void*)crv,(void*)crn,(void*)cck,(void*)ccv,(void*)cc})cudaFree(ptr);
    printf("PASS KV norm/RoPE and all channels, ring/noise, max FP16 error %.9g\n",cache_error);
    // Non-causal sliding attention across the ring boundary, all 128 channels.
    constexpr int nt=8,pos=2050,chunks=8;
    std::vector<float> query(nt*dn::QD);
    std::vector<__half> hk(8*dn::CACHE*dn::HD),hv(hk.size()),nk(nt*dn::KD),nv(nk.size());
    for(size_t i=0;i<query.size();i++)query[i]=std::sin(i*.003)*.2;
    for(size_t i=0;i<hk.size();i++){hk[i]=__float2half(std::sin(i*.017)*.3);hv[i]=__float2half(std::cos(i*.023)*.4);}
    for(size_t i=0;i<nk.size();i++){nk[i]=__float2half(std::sin(i*.019)*.3);nv[i]=__float2half(std::cos(i*.029)*.4);}
    float * gq,*gp,*go;float2 * gs;__half * gk,*gv,*gnk,*gnv;dn::Control * gc;
    DC(cudaMalloc(&gq,query.size()*4));DC(cudaMalloc(&go,query.size()*4));DC(cudaMalloc(&gp,nt*32*chunks*dn::HD*4));DC(cudaMalloc(&gs,nt*32*chunks*sizeof(float2)));
    DC(cudaMalloc(&gk,hk.size()*2));DC(cudaMalloc(&gv,hv.size()*2));DC(cudaMalloc(&gnk,nk.size()*2));DC(cudaMalloc(&gnv,nv.size()*2));DC(cudaMalloc(&gc,sizeof(dn::Control)));
    const dn::Control control{pos,123};DC(cudaMemcpy(gc,&control,sizeof(control),cudaMemcpyHostToDevice));
    DC(cudaMemcpy(gq,query.data(),query.size()*4,cudaMemcpyHostToDevice));DC(cudaMemcpy(gk,hk.data(),hk.size()*2,cudaMemcpyHostToDevice));DC(cudaMemcpy(gv,hv.data(),hv.size()*2,cudaMemcpyHostToDevice));DC(cudaMemcpy(gnk,nk.data(),nk.size()*2,cudaMemcpyHostToDevice));DC(cudaMemcpy(gnv,nv.data(),nv.size()*2,cudaMemcpyHostToDevice));
    dn::attention<<<dim3(nt*4,chunks),256>>>(gq,gk,gv,gnk,gnv,gp,gs,nt,chunks,gc);
    dn::attention_combine<<<(nt*dn::QD+255)/256,256>>>(gp,gs,go,nt,chunks);
    std::vector<float> out(query.size());DC(cudaMemcpy(out.data(),go,out.size()*4,cudaMemcpyDeviceToHost));
    double attention_error=0;
    for(int t=0;t<nt;t++)for(int h0=0;h0<32;h0++) {
        const int first=std::max(0,pos+t-dn::WINDOW+1),group=h0/4;double sum=0;std::vector<double> expect(dn::HD);
        for(int p0=first;p0<pos+nt;p0++) {
            const size_t idx=p0<pos?(size_t(group)*dn::CACHE+p0%dn::CACHE)*dn::HD:(size_t(p0-pos)*8+group)*dn::HD;
            const auto & keys=p0<pos?hk:nk;const auto & vals=p0<pos?hv:nv;double score=0;
            for(int e=0;e<dn::HD;e++)score+=query[(t*32+h0)*dn::HD+e]*double(__half2float(keys[idx+e]));
            double z=std::exp(score/std::sqrt(128.));sum+=z;
            for(int e=0;e<dn::HD;e++)expect[e]+=z*__half2float(vals[idx+e]);
        }
        for(int e=0;e<dn::HD;e++) {double error=std::abs(out[(t*32+h0)*dn::HD+e]-expect[e]/sum);attention_error=std::max(attention_error,error);
            if(error>1e-5){fprintf(stderr,"attention t=%d h=%d e=%d actual=%g expected=%g\n",t,h0,e,out[(t*32+h0)*dn::HD+e],expect[e]/sum);throw std::runtime_error("sliding attention differs from CPU oracle");}}
    }
    for(void * p0:{(void*)gq,(void*)gp,(void*)go,(void*)gs,(void*)gk,(void*)gv,(void*)gnk,(void*)gnv,(void*)gc})cudaFree(p0);
    printf("PASS sliding attention 2050+8, max error %.9g\n",attention_error);
    printf("PASS native draft: %zu weights, %zu F32 dots, both conv sides; max weight error %.9g, max matmul error %.9g\n",values,dots,max_weight,max_mm);
 } catch(const std::exception & e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
 return 0;
}
