#include "../qwen38_mk.h"
#include "../gguf.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

static void require(bool ok,const char * message) { if (!ok) throw std::runtime_error(message); }
static std::vector<int> read_ids(const char * path) {
    std::ifstream f(path);std::vector<int> v;int id;
    while (f>>id) v.push_back(id);
    require(f.eof() && !v.empty(),"invalid prompt IDs");return v;
}

int main(int argc,char ** argv) {
    try {
        require(argc==4,"usage: external_verify_test model short.ids long.ids");
        setenv("MK_PREFILL_CHUNK","128",1);
        auto prompt=read_ids(argv[2]);
        gguf::File file(argv[1]);
        MKEngine engine(argv[1],4096,0,"",1);
        std::vector<int> reference{engine.prefill(prompt)};
        while (reference.size()<80) engine.step(reference);
        engine.reset();
        auto weights=engine.share_weights();
        require(engine.share_weights().output==weights.output,"shared weight identity changed");
        for (auto entry:{std::pair<ggml_tensor *,const char *>{weights.embedding,"token_embd.weight"},
                        {weights.output,"output.weight"}}) {
            const auto & original=file.get(entry.second);
            require(ggml_nbytes(entry.first)==original.nbytes,"shared weight shape changed");
            std::vector<uint8_t> bytes(16384);
            for (size_t offset:{size_t(0),size_t(original.nbytes/2),size_t(original.nbytes-bytes.size())}) {
                ggml_backend_tensor_get(entry.first,bytes.data(),offset,bytes.size());
                require(!std::memcmp(bytes.data(),original.data+offset,bytes.size()),"shared weights changed bytes");
            }
        }
        std::vector<int> expected=prompt;
        expected.insert(expected.end(),reference.begin(),reference.end());
        int next_position=0,callbacks=0;
        engine.enable_features({0,6,20,34,48,62},[&](int pos,int n,const std::vector<float> & features) {
            require(pos==next_position,"feature positions are not contiguous");
            require(features.size()==size_t(n)*6*5120,"feature shape mismatch");
            require(pos+n<=int(expected.size()),"feature position outside known sequence");
            for (float v:features) require(std::isfinite(v),"non-finite feature");
            std::vector<float> embedding(5120);
            const auto & tensor=file.get("token_embd.weight");
            for (int t=0;t<n;t++) {
                ggml_get_type_traits(ggml_type(tensor.type))->to_float(
                    tensor.data+size_t(expected[pos+t])*gguf::type_row_bytes(tensor.type,5120),embedding.data(),5120);
                for (int e=0;e<5120;e++) if(std::abs(embedding[e]-features[size_t(t)*6*5120+e])>1e-6f) {
                    fprintf(stderr,"feature mismatch pos=%d count=%d token=%d e=%d expected=%g actual=%g\n",
                        pos+t,n,expected[pos+t],e,embedding[e],features[size_t(t)*6*5120+e]);
                    throw std::runtime_error("layer-zero feature does not match CPU embedding");
                }
            }
            next_position+=n;callbacks++;
        });
        for (int depth:{0,1,2,3,4,5,6,7}) {
        for (int rejected=0;rejected<=depth;rejected++) {
            engine.reset();next_position=0;
            std::vector<int> got{engine.prefill(prompt)};
            std::vector<int> proposals(reference.begin()+1,reference.begin()+1+depth);
            if (rejected<depth) proposals[rejected]=(proposals[rejected]+1)%248320;
            engine.verify_external(proposals,got);
            require(engine.stats.accepted==rejected,"unexpected first rejection position");
            require(got.size()==size_t(rejected+2),"incorrect accepted prefix plus bonus");
            while (got.size()<64) {
                proposals.assign(reference.begin()+got.size(),reference.begin()+got.size()+depth);
                engine.verify_external(proposals,got);
            }
            require(std::equal(got.begin(),got.begin()+64,reference.begin()),"external verifier differs from scalar decode");
            require(engine.position()==int(engine.sequence().size())-1,"sequence position mismatch");
            require(engine.committed()<=engine.position(),"recurrent state committed speculative rows");
            printf("PASS depth %d rejection %d, scalar IDs and continued replay\n",depth,rejected);
        }
        }
        const auto before=engine.sequence();
        for (auto invalid:{std::vector<int>{-1},std::vector<int>{248320},std::vector<int>(8,0)}) {
            bool rejected=false;std::vector<int> out;
            try { engine.verify_external(invalid,out); } catch (const std::runtime_error &) { rejected=true; }
            require(rejected && engine.sequence()==before,"invalid proposal mutated sequence");
        }
        engine.reset();next_position=0;
        expected=read_ids(argv[3]);
        engine.prefill(expected);
        require(next_position==int(expected.size()),"chunked prefill features skipped positions");
        require(expected.size()>128,"long prompt did not cross prefill chunk");
        printf("PASS: shared IQ2_S/Q4_K bytes, invalid proposals, %d feature callbacks and long chunked prefill\n",callbacks);
    } catch (const std::exception & e) { fprintf(stderr,"FAIL: %s\n",e.what());return 1; }
    return 0;
}
