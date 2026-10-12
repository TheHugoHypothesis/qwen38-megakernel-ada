// Full-vocabulary legacy comparison plus sampled CPU GGML/double oracle.
#include "../dflash_native.cu"
#include <numeric>

template<int ROWS> void launch_head(dn::Matrix w,const int8_t * q,const float * s,float * y) {
    dn::mmq_typed<7,12,ROWS><<<(w.N+4*ROWS-1)/(4*ROWS),128>>>(w,q,s,y);
}
int main(int argc,char ** argv) {
 try {
    if(argc!=2)throw std::runtime_error("usage: dflash_head_test target.gguf");
    gguf::File file(argv[1]);const auto & tensor=file.get("output.weight");
    const int vocab = int(tensor.ne.at(1));
    if(vocab%4 || tensor.type!=12||tensor.ne!=std::vector<uint64_t>{dn::D,uint64_t(vocab)})
        throw std::runtime_error("expected original Q4_K vocabulary head");
    uint8_t * weights;float * x,* scale,* out;int8_t * q;
    DC(cudaMalloc(&weights,tensor.nbytes));DC(cudaMemcpy(weights,tensor.data,tensor.nbytes,cudaMemcpyHostToDevice));
    dn::Matrix w{weights,dn::D,vocab,12,int(gguf::type_row_bytes(12,dn::D))};
    DC(cudaMalloc(&x,7*dn::D*4));DC(cudaMalloc(&q,7*dn::D));
    DC(cudaMalloc(&scale,7*(dn::D/32)*4));DC(cudaMalloc(&out,7*vocab*4));
    double max_error=0,max_nrmse=0;size_t exact=0;
    for(int trial=0;trial<3;trial++) {
        std::vector<float> input(7*dn::D);
        if(trial)for(size_t i=0;i<input.size();i++)input[i]=std::sin(i*(trial==1?.017:.031))*(trial==1?.3:2.1);
        DC(cudaMemcpy(x,input.data(),input.size()*4,cudaMemcpyHostToDevice));
        dn::quant_input<<<(7*dn::D/32+3)/4,128>>>(x,q,scale,dn::D,7);
        dn::mmq<7><<<(w.N+7)/8,128>>>(w,q,scale,out);
        std::vector<float> legacy(7*vocab);
        DC(cudaMemcpy(legacy.data(),out,legacy.size()*4,cudaMemcpyDeviceToHost));
        for(int rows:{2,4}) {
            if(rows==2)launch_head<2>(w,q,scale,out);else launch_head<4>(w,q,scale,out);
            std::vector<float> actual(legacy.size());DC(cudaMemcpy(actual.data(),out,actual.size()*4,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<actual.size();i++) {
                if(!std::isfinite(actual[i])||actual[i]!=legacy[i])throw std::runtime_error("typed vocabulary head changes a legacy score");
                exact++;
            }
        }
        std::vector<uint8_t> packed(size_t(7)*(dn::D/32)*34);std::vector<float> decoded_input(input.size()),decoded_weight(dn::D);
        auto traits=ggml_get_type_traits(GGML_TYPE_Q8_0);
        traits->from_float_ref(input.data(),packed.data(),input.size());traits->to_float(packed.data(),decoded_input.data(),input.size());
        double e2=0,r2=0;
        for(int row:{0,1,7,31,123,997,8191,65535,82773,124160,165547,200001,248319,vocab-2,vocab-1}) {
            if(row>=vocab)continue;
            ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(tensor.data+size_t(row)*w.rb,decoded_weight.data(),dn::D);
            for(int t=0;t<7;t++) {
                double ref=0;for(int k=0;k<dn::D;k++)ref+=double(decoded_weight[k])*decoded_input[t*dn::D+k];
                const double delta=legacy[t*vocab+row]-ref;
                e2+=delta*delta;r2+=ref*ref;max_error=std::max(max_error,std::abs(delta));
                if(std::abs(delta)>.002+5e-4*std::abs(ref))throw std::runtime_error("head differs from CPU GGML/double");
            }
        }
        max_nrmse=std::max(max_nrmse,r2?std::sqrt(e2/r2):std::sqrt(e2));
    }
    cudaEvent_t begin,end;DC(cudaEventCreate(&begin));DC(cudaEventCreate(&end));
    for(int mode=0;mode<3;mode++) {
        auto launch=[&](){if(!mode)dn::mmq<7><<<(w.N+7)/8,128>>>(w,q,scale,out);else if(mode==1)launch_head<2>(w,q,scale,out);else launch_head<4>(w,q,scale,out);};
        for(int i=0;i<5;i++)launch();DC(cudaEventRecord(begin));
        for(int i=0;i<30;i++)launch();DC(cudaEventRecord(end));DC(cudaEventSynchronize(end));
        float ms;DC(cudaEventElapsedTime(&ms,begin,end));printf("HEAD mode%d %.6f ms\n",mode,ms/30);
    }
    printf("PASS %zu vocabulary scores equal legacy; sampled CPU/double dots, max NRMSE %.9g abs %.9g\n",exact,max_nrmse,max_error);
    DC(cudaEventDestroy(begin));DC(cudaEventDestroy(end));
    for(void * p:{static_cast<void *>(weights),static_cast<void *>(x),static_cast<void *>(q),static_cast<void *>(scale),static_cast<void *>(out)})DC(cudaFree(p));
 }catch(const std::exception & e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
