// Independent scalar CPU oracle for the production KV packing and dequantizers.
#include "../kv_quant.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
static void require(bool ok,const char * s) { if(!ok)throw std::runtime_error(s); }
template<int MODE> __global__ void pack(const float * x,void * q,__half * s,int rows) {
    if(blockIdx.x>=rows)return;
    float v[8];for(int j=0;j<8;j++)v[j]=x[blockIdx.x*256+threadIdx.x*8+j];
    kv_store<MODE>(q,s,blockIdx.x+2,v); // guard rows around the writes
}
template<int MODE> __global__ void unpack(const uint8_t * q,const __half * s,__half * y,int rows) {
    const int row=blockIdx.x,c=threadIdx.x;
    if(row>=rows)return;
    __half2 h[4];
    const float d=MODE?__half2float(s[(row+2)*8+(c>>2)]):1;
    if constexpr(MODE==2)q4_to_f16(*reinterpret_cast<const unsigned *>(q+(row+2)*128+4*c),d,h);
    else if constexpr(MODE==1)q8_to_f16(*reinterpret_cast<const uint2 *>(q+(row+2)*256+8*c),d,h);
    else { for(int j=0;j<4;j++) h[j]=reinterpret_cast<const __half2 *>(q+(row+2)*512)[c*4+j]; }
    *reinterpret_cast<int4 *>(y+row*256+c*8)=*reinterpret_cast<int4 *>(h);
}
template<int MODE> static void run() {
    constexpr int rows=37;
    std::vector<float> x(rows*256);
    for(int r=0;r<rows;r++)for(int c=0;c<256;c++) {
        float v=std::sin(float(r*256+c)*.013f)*(1+r%7);
        if(r==0 || (r==1 && c<32))v=0;
        if(r==2)v=(c&1)?-1:1;
        if(r==3)v*=1e-5f;
        if(r==4)v*=100;
        x[r*256+c]=v;
    }
    float * dx;uint8_t * dq;__half * ds,*dy;
    const size_t rb=kv_payload_bytes(MODE),bytes=(rows+4)*rb;
    CHECK(cudaMalloc(&dx,x.size()*4));CHECK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&dq,bytes));CHECK(cudaMemset(dq,0xa5,bytes));
    CHECK(cudaMalloc(&ds,(rows+4)*8*2));CHECK(cudaMemset(ds,0xa5,(rows+4)*8*2));CHECK(cudaMalloc(&dy,x.size()*2));
    pack<MODE><<<rows,32>>>(dx,dq,ds,rows);unpack<MODE><<<rows,32>>>(dq,ds,dy,rows);
    std::vector<uint8_t> q(bytes);std::vector<__half> s((rows+4)*8),y(x.size());
    CHECK(cudaMemcpy(q.data(),dq,bytes,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(s.data(),ds,s.size()*2,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(y.data(),dy,y.size()*2,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<2*rb;i++)require(q[i]==0xa5 && q[bytes-1-i]==0xa5,"payload guard overwritten");
    for(size_t i=0;i<16*2;i++) {
        auto p=reinterpret_cast<const uint8_t *>(s.data());
        require(p[i]==0xa5 && p[s.size()*2-1-i]==0xa5,"scale guard overwritten");
    }
    double max_error=0;
    for(int r=0;r<rows;r++)for(int b=0;b<8;b++) {
        float am=0;for(int k=0;k<32;k++)am=std::max(am,std::abs(x[r*256+b*32+k]));
        const float limit=MODE==2?7:127;
        const float inv=am>0?limit/am:0;
        const __half expected_scale=__float2half(am/limit);
        if constexpr(MODE!=0)require(__half2float(s[(r+2)*8+b])==__half2float(expected_scale),"scale differs from CPU oracle");
        for(int k=0;k<32;k++) {
            const int c=b*32+k;const float value=x[r*256+c];
            const int quant=int(std::nearbyint(value*inv));
            float expected;
            if constexpr(MODE==2) {
                const int nibble=(q[(r+2)*rb+c/2]>>(4*(c&1)))&15;
                const int signed_value=(nibble^8)-8;
                require(signed_value==quant && signed_value>=-7 && signed_value<=7,"Q4 nibble differs from CPU oracle");
                expected=__half2float(expected_scale)*quant;
            } else if constexpr(MODE==1) {
                require(int(int8_t(q[(r+2)*rb+c]))==quant,"Q8 byte differs from CPU oracle");
                expected=__half2float(expected_scale)*quant;
            } else expected=value;
            require(__half2float(y[r*256+c])==__half2float(__float2half(expected)),"dequantized value differs from CPU FP16 oracle");
            max_error=std::max(max_error,double(std::abs(__half2float(y[r*256+c])-value)));
        }
    }
    CHECK(cudaFree(dx));CHECK(cudaFree(dq));CHECK(cudaFree(ds));CHECK(cudaFree(dy));
    printf("PASS KV mode %d: %zu values, CPU bytes/scales/FP16 oracle, guards; max abs quantization error %.6g\n",MODE,x.size(),max_error);
}
int main() { try {
    static_assert(kv_cache_bytes(1,17*4)==36992);
    static_assert(kv_cache_bytes(2,17*4)==19584);
    static_assert(kv_cache_bytes(1,17ull*4*65536)-kv_cache_bytes(2,17ull*4*65536)==1140850688);
    run<0>();run<1>();run<2>();
} catch(const std::exception & e) {fprintf(stderr,"FAIL: %s\n",e.what());return 1;} }
