// Exercise the actual prefill and split-K decode kernels against double CPU attention.
// Including the implementation lets this test use private kernels without a production test API.
#include "../qwen38_mk.cu"
#include <memory>

static void need(bool ok,const char * msg) { if(!ok)throw std::runtime_error(msg); }
struct DeviceArena {
    std::vector<void *> ptrs;
    template<class U> U * alloc(size_t n) { U * p;CK(cudaMalloc(&p,n*sizeof(U)));ptrs.push_back(p);return p; }
    template<class U> U * upload(const std::vector<U> & a) { auto p=alloc<U>(a.size());CK(cudaMemcpy(p,a.data(),a.size()*sizeof(U),cudaMemcpyHostToDevice));return p; }
    ~DeviceArena() { for(auto p:ptrs)cudaFree(p); }
};
template<int MODE> __global__ void history(const float * in,int n,int slot) {
    const int head=blockIdx.y,pos=blockIdx.x,lane=threadIdx.x;
    float v[8];const size_t row=(size_t(slot)*NKV+head)*cB.ctx+pos;
    for(int k=0;k<8;k++)v[k]=in[(head*n+pos)*HD+lane*8+k];
    kv_store<MODE>(cB.kc,cB.ksc,row,v);
    for(int k=0;k<8;k++)v[k]=in[(NKV*n+head*n+pos)*HD+lane*8+k];
    kv_store<MODE>(cB.vc,cB.vsc,row,v);
}
template<int MODE,int T> __global__ void __launch_bounds__(NT) decode_attention(LayerW w,int n,int slot) { attn_core_t<T,MODE>(w,slot,n); }

static std::vector<float> cache_cpu(const Bufs & b,bool value,int slot,int positions) {
    const size_t rb=kv_payload_bytes(b.kv_mode);
    std::vector<float> result(size_t(NKV)*positions*HD);
    std::vector<uint8_t> bytes(size_t(positions)*rb);
    std::vector<__half> scales(size_t(positions)*8);
    for(int g=0;g<NKV;g++) {
        const size_t row=(size_t(slot)*NKV+g)*b.ctx;
        CK(cudaMemcpy(bytes.data(),static_cast<const uint8_t *>(value?b.vc:b.kc)+row*rb,bytes.size(),cudaMemcpyDeviceToHost));
        if(b.kv_mode)CK(cudaMemcpy(scales.data(),(value?b.vsc:b.ksc)+row*8,scales.size()*2,cudaMemcpyDeviceToHost));
        for(int p=0;p<positions;p++)for(int c=0;c<HD;c++) {
            float v;
            if(b.kv_mode==2) {const int nibble=(bytes[p*rb+c/2]>>(4*(c&1)))&15;v=((nibble^8)-8)*__half2float(scales[p*8+c/32]);}
            else if(b.kv_mode==1)v=int8_t(bytes[p*rb+c])*__half2float(scales[p*8+c/32]);
            else v=__half2float(reinterpret_cast<const __half *>(bytes.data())[p*HD+c]);
            result[(size_t(g)*positions+p)*HD+c]=__half2float(__float2half(v));
        }
    }
    return result;
}

template<int MODE,int T,bool DECODE> static void run(int n0,int capacity) {
    const int slot=16,positions=n0+T;
    DeviceArena a;Bufs b{};PBufs p{};LayerW w{};
    b.ctx=capacity;b.kv_mode=MODE;
    const size_t rows=size_t(N_ATT_SLOTS)*NKV*capacity;
    b.kc=a.alloc<uint8_t>(rows*kv_payload_bytes(MODE));b.vc=a.alloc<uint8_t>(rows*kv_payload_bytes(MODE));
    if constexpr(MODE) {b.ksc=a.alloc<__half>(rows*8);b.vsc=a.alloc<__half>(rows*8);}
    std::vector<float> h(size_t(2)*NKV*n0*HD);
    for(size_t i=0;i<h.size();i++)h[i]=std::sin(float(i)*.00371f)*.8f;
    std::vector<float> q(size_t(T)*QG),k(size_t(T)*KVD),v(k.size()),kv(size_t(T)*2*KVD);
    for(int t=0;t<T;t++) {
        for(int j=0;j<QG;j++)q[size_t(t)*QG+j]=std::cos(float(t*QG+j)*.0171f)*.65f;
        for(int j=0;j<KVD;j++) {
            k[size_t(t)*KVD+j]=std::sin(float(t*KVD+j)*.0131f)*.7f;
            v[size_t(t)*KVD+j]=std::cos(float(t*KVD+j)*.00571f)*.9f;
            kv[size_t(t)*2*KVD+j]=k[size_t(t)*KVD+j];kv[size_t(t)*2*KVD+KVD+j]=v[size_t(t)*KVD+j];
        }
    }
    b.qbuf=a.upload(q);b.kbuf=a.upload(k);b.vbuf=a.upload(v);
    std::vector<float2> rope(size_t(capacity)*32,make_float2(1,0));b.rope=a.upload(rope);
    std::vector<float> norms(HD,1);w.qn=a.upload(norms);w.kn=w.qn;
    b.apart=a.alloc<float>(size_t(NKV)*MAX_CH*AR*HD);b.aml=a.alloc<float2>(size_t(NKV)*MAX_CH*AR);
    p.qh=a.alloc<__half>(size_t(T)*NH*HD);p.y0=b.qbuf;p.y1=a.upload(kv);p.y2=a.alloc<float>(size_t(T)*VD);
    CK(cudaMemcpyToSymbol(cB,&b,sizeof(b)));CK(cudaMemcpyToSymbol(cP,&p,sizeof(p)));
    if(n0)history<MODE><<<dim3(n0,NKV),32>>>(a.upload(h),n0,slot);
    pk_attn_prep<<<T,256>>>(n0,slot,w.qn,w.kn,1);
    std::vector<__half> qh(size_t(T)*NH*HD);CK(cudaMemcpy(qh.data(),p.qh,qh.size()*2,cudaMemcpyDeviceToHost));
    std::vector<float> actual(size_t(T)*VD);
    if constexpr(DECODE) {
        CK(cudaFuncSetAttribute(decode_attention<MODE,T>,cudaFuncAttributeMaxDynamicSharedMemorySize,SM_TOTAL));
        decode_attention<MODE,T><<<34,NT,SM_TOTAL>>>(w,n0,slot);
        const int nch=std::max(1,std::min({MAX_CH,34/NKV,(positions+63)/64}));
        std::vector<float> partial(size_t(NKV)*nch*AR*HD);std::vector<float2> ml(size_t(NKV)*nch*AR);
        CK(cudaMemcpy(partial.data(),b.apart,partial.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(ml.data(),b.aml,ml.size()*8,cudaMemcpyDeviceToHost));
        for(int g=0;g<NKV;g++)for(int hq=0;hq<6;hq++)for(int t=0;t<T;t++) {
            float mx=-INFINITY;for(int ch=0;ch<nch;ch++)mx=std::max(mx,ml[(g*nch+ch)*AR+hq*T+t].x);
            double den=0;for(int ch=0;ch<nch;ch++){auto z=ml[(g*nch+ch)*AR+hq*T+t];den+=std::exp(double(z.x-mx))*z.y;}
            for(int c=0;c<HD;c++) {double num=0;for(int ch=0;ch<nch;ch++)num+=std::exp(double(ml[(g*nch+ch)*AR+hq*T+t].x-mx))*partial[((size_t(g)*nch+ch)*AR+hq*T+t)*HD+c];actual[size_t(t)*VD+(g*6+hq)*HD+c]=num/den;}
        }
    } else {
        CK(cudaFuncSetAttribute(pk_attn<MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,AT_SMEM));
        pk_attn<MODE><<<dim3((T+AT_Q-1)/AT_Q,NKV),192,AT_SMEM>>>(T,n0,slot);
        CK(cudaMemcpy(actual.data(),p.y2,actual.size()*4,cudaMemcpyDeviceToHost));
    }
    const auto kc=cache_cpu(b,false,slot,positions),vc=cache_cpu(b,true,slot,positions);
    double max_error=0,sse=0,norm=0;
    for(int t=0;t<T;t++)for(int head=0;head<NH;head++) {
        const int g=head/6;std::vector<double> scores(n0+t+1);double mx=-INFINITY;
        for(int pos=0;pos<=n0+t;pos++) {double dot=0;for(int c=0;c<HD;c++)dot+=__half2float(qh[(size_t(t)*NH+head)*HD+c])*kc[(size_t(g)*positions+pos)*HD+c];scores[pos]=dot;mx=std::max(mx,dot);}
        double den=0;for(auto & score:scores){score=std::exp(score-mx);den+=score;}
        for(int c=0;c<HD;c++) {
            double expected=0;for(int pos=0;pos<=n0+t;pos++)expected+=scores[pos]*vc[(size_t(g)*positions+pos)*HD+c];expected/=den;
            const float got=actual[size_t(t)*VD+head*HD+c];const double err=std::abs(got-expected);
            need(std::isfinite(got) && err<.003+.005*std::abs(expected),"attention differs from independent CPU oracle");
            max_error=std::max(max_error,err);sse+=err*err;norm+=expected*expected;
        }
    }
    printf("PASS mode %d %s T=%d n0=%d ctx=%d slot=16: max error %.6g NRMSE %.6g\n",MODE,DECODE?"split-K":"prefill",T,n0,capacity,max_error,std::sqrt(sse/std::max(norm,1e-20)));
}
int main() { try {
    run<0,17,false>(67,65536);run<1,17,false>(67,65536);run<2,17,false>(67,65536);
    run<2,1,true>(67,65536);run<2,8,true>(67,65536);
    run<2,17,false>(4093,65536);run<2,8,true>(4093,65536);
    run<2,1,false>(65535,65536);run<2,1,true>(65535,65536);
} catch(const std::exception & e) {fprintf(stderr,"FAIL: %s\n",e.what());return 1;} }
