// Isolated calls to actual production phase functions, without compiling the full decoder.
#define MK_PHASE_ONLY
#include "../qwen38_mk.cu"
#include <fstream>
#include <set>

struct Arena {
    std::vector<void *> ptrs;
    template<class X>X * alloc(size_t n){X * p;CK(cudaMalloc(&p,n*sizeof(X)));ptrs.push_back(p);return p;}
    ~Arena(){for(void * p:ptrs)cudaFree(p);}
};
static QBuf buffer(Arena & a,int units,int tokens){
    return {a.alloc<int8_t>(size_t(tokens)*units*16),a.alloc<int8_t>(size_t(tokens)*units*16),
            a.alloc<uint2>(size_t(tokens)*units),a.alloc<float>(size_t(tokens)*units),units};
}
static Mat weight(Arena & a,const gguf::Tensor & t){
    auto p=a.alloc<uint8_t>(t.nbytes);CK(cudaMemcpy(p,t.data,t.nbytes,cudaMemcpyHostToDevice));
    return {p,nullptr,int(t.type),int(t.ne[0]),int(t.ne[1]),int(gguf::type_row_bytes(t.type,t.ne[0])),0};
}
template<int T>__global__ void quant_input(const float * x,const float * norm,QBuf q,int k){
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,t=blockIdx.x;
    for(int j=warp;j<k/32;j+=NW){const float v=x[t*k+32*j+lane];q32_store(q,t,j,v*norm[32*j+lane],v*v,true);}
}
__global__ void __launch_bounds__(NT,1) gu_phase(LayerW w){ffn_gu<4>(w,smem);}
__global__ void __launch_bounds__(NT,1) down_phase(LayerW w,const float * norm){resid32<8,17,T_ANY>(w.fd,cB.qa,cB.x,false,norm,smem);}

// Pair adjacent iterations of the existing block-cyclic row assignment. This
// shares staging/barriers, keeps two independent ordered row-pair accumulators,
// and retains the original per-SM row count (including a final unpaired tile).
__global__ void __launch_bounds__(NT,1) down_tiles_phase(LayerW w,const float * norm){
    constexpr int U0=256,U1=32,T=8;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    SX sx0{reinterpret_cast<int4 *>(smem),reinterpret_cast<int4 *>(smem+T*U0*16),reinterpret_cast<uint2 *>(smem+T*U0*32),U0};
    SX sx1{reinterpret_cast<int4 *>(smem),reinterpret_cast<int4 *>(smem+T*U1*16),reinterpret_cast<uint2 *>(smem+T*U1*32),U1};
    auto finish=[&](int j,float (&acc)[2][T]){
        wsum_acc<T,2>(acc);float * s=misc(smem)+8;
        if(lane==0)for(int t=0;t<T;t++){s[t*32+2*warp]=acc[0][t];s[t*32+2*warp+1]=acc[1][t];}
        __syncthreads();
        if(warp<T){const int row=32*j+lane;const float v=s[warp*32+lane];cB.x[warp*D+row]=v;q32_store(cB.qn,warp,j,v*norm[row],v*v,true);}
        __syncthreads();
    };
    for(int j=blockIdx.x;j<D/32;j+=2*gridDim.x){
        const int second=j+gridDim.x;const bool paired=second<D/32;
        float first[2][T],other[2][T];zero_acc<T,2>(first);zero_acc<T,2>(other);
        for(int k=0;k<2;k++){
            fill_xs_part<T,U0>(cB.qa,k*U0,sx0);
            mrows<T,U0/32,2,T_ANY>(w.fd,32*j+2*warp,sx0,first,k*U0);
            if(paired)mrows<T,U0/32,2,T_ANY>(w.fd,32*second+2*warp,sx0,other,k*U0);
        }
        fill_xs_part<T,U1>(cB.qa,2*U0,sx1);
        mrows<T,U1/32,2,T_ANY>(w.fd,32*j+2*warp,sx1,first,2*U0);
        if(paired)mrows<T,U1/32,2,T_ANY>(w.fd,32*second+2*warp,sx1,other,2*U0);
        finish(j,first);if(paired)finish(second,other);
    }
}
__global__ void scrub_l2(uint4 * p,int n){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)p[i]=make_uint4(i,i+1,i+2,i+3);}
static std::vector<uint8_t> payload(QBuf q,int t){
    std::vector<uint8_t> r(size_t(t)*q.U*40);size_t s=size_t(t)*q.U;
    CK(cudaMemcpy(r.data(),q.xa,s*16,cudaMemcpyDeviceToHost));CK(cudaMemcpy(r.data()+s*16,q.xb,s*16,cudaMemcpyDeviceToHost));CK(cudaMemcpy(r.data()+s*32,q.meta,s*8,cudaMemcpyDeviceToHost));return r;
}
int main(int argc,char ** argv){
 try{
    if(argc<2||argc>3)throw std::runtime_error("usage: ffn_phases_test model [layer-input-trace.f32]");
    gguf::File file(argv[1]);cudaDeviceProp props;CK(cudaGetDeviceProperties(&props,0));Arena a;Bufs b{};
    b.qn=buffer(a,U_D,8);b.qa=buffer(a,U_FF,8);b.x=a.alloc<float>(8*D);
    b.gu_probe=a.alloc<float>(size_t(4)*FF*5);
    b.ffn_iq_fused=true;b.iq_paired=true;b.iq_inline=true;
    auto x=a.alloc<float>(8*FF),norm=a.alloc<float>(FF);
    auto scrub=a.alloc<uint4>((64<<20)/sizeof(uint4));
    std::vector<float> values(8*FF),ones(FF,1.f),real(8*D);
    for(size_t i=0;i<values.size();i++)values[i]=std::sin(float(i)*.00371f)*.73f+std::cos(float(i)*.0719f)*.21f;
    std::copy_n(values.begin(),real.size(),real.begin());
    if(argc==3){std::ifstream trace(argv[2],std::ios::binary);for(int t=0;t<4;t++){trace.seekg((size_t(58+t)*64+5)*D*4);trace.read(reinterpret_cast<char *>(real.data()+t*D),D*4);}if(!trace)throw std::runtime_error("cannot read captured inputs");}
    CK(cudaMemcpy(x,real.data(),real.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(norm,ones.data(),FF*4,cudaMemcpyHostToDevice));
    CK(cudaFuncSetAttribute(gu_phase,cudaFuncAttributeMaxDynamicSharedMemorySize,SM_TOTAL));
    CK(cudaFuncSetAttribute(down_phase,cudaFuncAttributeMaxDynamicSharedMemorySize,SM_TOTAL));
    CK(cudaFuncSetAttribute(down_tiles_phase,cudaFuncAttributeMaxDynamicSharedMemorySize,SM_TOTAL));
    for(int l=0;l<64;l++){
        const auto p=std::string("blk.")+std::to_string(l)+".ffn_";Arena layer;LayerW w{};
        w.fg=weight(layer,file.get(p+"gate.weight"));w.fu=weight(layer,file.get(p+"up.weight"));
        std::vector<float> hn(D);const auto & n=file.get("blk."+std::to_string(l)+".post_attention_norm.weight");
        if(n.type==0)std::memcpy(hn.data(),n.data,D*4);else ggml_get_type_traits(ggml_type(n.type))->to_float(n.data,hn.data(),D);
        CK(cudaMemcpy(norm,hn.data(),D*4,cudaMemcpyHostToDevice));
        b.ffn_iq_small=false;CK(cudaMemcpyToSymbol(cB,&b,sizeof(b)));quant_input<4><<<4,NT>>>(x,norm,b.qn,D);gu_phase<<<props.multiProcessorCount,NT,SM_TOTAL>>>(w);
        auto ref=payload(b.qa,4);
        std::vector<float> raw_ref(size_t(4)*FF*5),raw_actual(raw_ref.size());
        CK(cudaMemcpy(raw_ref.data(),b.gu_probe,raw_ref.size()*4,cudaMemcpyDeviceToHost));
        b.ffn_iq_small=true;CK(cudaMemcpyToSymbol(cB,&b,sizeof(b)));gu_phase<<<props.multiProcessorCount,NT,SM_TOTAL>>>(w);auto got=payload(b.qa,4);
        CK(cudaMemcpy(raw_actual.data(),b.gu_probe,raw_actual.size()*4,cudaMemcpyDeviceToHost));
        int different=0;size_t first=0;for(size_t i=0;i<ref.size();i++)if(ref[i]!=got[i]){if(!different)first=i;different++;}
        printf("GU {\"layer\":%d,\"gate_type\":%d,\"up_type\":%d,\"different_bytes\":%d,\"first_byte\":%zu}\n",l,w.fg.type,w.fu.type,different,first);fflush(stdout);
        for(int field=0;field<5;field++){
            int count=0;size_t index=0;double maximum=0;
            for(size_t i=field;i<raw_ref.size();i+=5)if(raw_ref[i]!=raw_actual[i]){if(!count)index=i;count++;maximum=std::max(maximum,std::abs(double(raw_ref[i])-raw_actual[i]));}
            if(count){printf("RAW {\"layer\":%d,\"field\":%d,\"different\":%d,\"max_abs\":%.9g,\"first_row\":%zu,\"legacy\":%.9g,\"fused\":%.9g}\n",l,field,count,maximum,index/5,raw_ref[index],raw_actual[index]);throw std::runtime_error("T4 raw accumulation/epilogue differs");}
        }fflush(stdout);
        if(different)throw std::runtime_error("T4 quantized payload/metadata differs");
    }
    // Every original down-projection format, at its actual 17408 x 5120 shape.
    std::set<int> seen;CK(cudaMemcpy(x,values.data(),values.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(norm,ones.data(),FF*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(cB,&b,sizeof(b)));quant_input<8><<<8,NT>>>(x,norm,b.qa,FF);
    for(int l=0;l<64;l++){
        const auto & t=file.get("blk."+std::to_string(l)+".ffn_down.weight");if(!seen.insert(t.type).second)continue;
        Arena layer;LayerW w{};w.fd=weight(layer,t);
        std::vector<float> reference(8*D),actual(reference.size());
        auto launch=[&](bool tiles){if(tiles)down_tiles_phase<<<props.multiProcessorCount,NT,SM_TOTAL>>>(w,norm);else down_phase<<<props.multiProcessorCount,NT,SM_TOTAL>>>(w,norm);};
        launch(false);CK(cudaMemcpy(reference.data(),b.x,reference.size()*4,cudaMemcpyDeviceToHost));auto qr=payload(b.qn,8);
        launch(true);CK(cudaMemcpy(actual.data(),b.x,actual.size()*4,cudaMemcpyDeviceToHost));auto qa=payload(b.qn,8);
        const bool exact=!std::memcmp(reference.data(),actual.data(),reference.size()*4)&&qr==qa;
        std::vector<float> activation(8*FF),weights(16*FF);auto encoded=payload(b.qa,8);
        const auto * am=encoded.data()+size_t(8)*U_FF*32;
        for(int token=0;token<8;token++)for(int unit=0;unit<U_FF;unit++)for(int part=0;part<2;part++){
            uint16_t bits;std::memcpy(&bits,am+(size_t(token)*U_FF+unit)*8+2*part,2);
            const float scale=__half2float(__ushort_as_half(bits));
            const int e=256*(unit/8)+64*((unit/2)&3)+16*(unit&1)+32*part;
            for(int j=0;j<16;j++)activation[token*FF+e+j]=scale*int8_t(encoded[size_t(part)*8*U_FF*16+(size_t(token)*U_FF+unit)*16+j]);
        }
        ggml_get_type_traits(ggml_type(t.type))->to_float(t.data,weights.data(),weights.size());
        double cpu_err=0,cpu_norm=0,cpu_max=0;
        for(int token=0;token<8;token++)for(int row=0;row<16;row++){
            double ref=0;for(int k=0;k<FF;k++)ref+=double(activation[token*FF+k])*weights[row*FF+k];
            const double delta=reference[token*D+row]-ref;cpu_err+=delta*delta;cpu_norm+=ref*ref;cpu_max=std::max(cpu_max,std::abs(delta));
            if(!std::isfinite(delta)||std::abs(delta)>1e-4+2e-5*std::abs(ref))throw std::runtime_error("down projection exceeds CPU tolerance");
        }
        double err=0,scale=0;for(size_t i=0;i<actual.size();i++){const double d=actual[i]-reference[i];err+=d*d;scale+=double(reference[i])*reference[i];}
        cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));double times[2]{};
        for(int rep=-2;rep<10;rep++)for(int k=0;k<2;k++){
            const int variant=rep%2?k:1-k;scrub_l2<<<256,256>>>(scrub,(64<<20)/sizeof(uint4));CK(cudaEventRecord(start));launch(variant);CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,start,end));if(rep>=0)times[variant]+=ms/10;
        }
        printf("DOWN {\"layer\":%d,\"type\":%d,\"legacy_ms\":%.6f,\"tiles_ms\":%.6f,\"exact\":%s,\"nrmse\":%.9g,\"cpu_nrmse\":%.9g,\"cpu_max_abs\":%.9g}\n",l,w.fd.type,times[0],times[1],exact?"true":"false",sqrt(err/std::max(scale,1e-30)),sqrt(cpu_err/std::max(cpu_norm,1e-30)),cpu_max);fflush(stdout);
        CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
        if(!exact)throw std::runtime_error("paired-tile down changed production results");
    }
    CK(cudaGetLastError());
 }catch(const std::exception & e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}return 0;
}
