// Bounds and scalar-conversion tests independent of CUDA and the large model.
#include "../gguf.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>
int main() {
    const std::string path="/tmp/qwen38-gguf-parser-"+std::to_string(getpid())+".gguf";
    for (const std::string bytes : {std::string(), std::string("GGUF"), std::string(24,'\0')}) {
        { std::ofstream f(path,std::ios::binary); f.write(bytes.data(),bytes.size()); }
        bool rejected=false;
        try { gguf::File f(path); } catch(const std::runtime_error &) {rejected=true;}
        if(!rejected) return 1;
    }
    unlink(path.c_str());
    uint16_t h[]={0,0x8000,1,0x3c00,0xc000,0x7c00};
    gguf::Tensor t; t.name="fp16";t.type=gguf::F16;t.data=reinterpret_cast<uint8_t*>(h);t.nbytes=sizeof(h);
    const auto v=gguf::float_values(t);
    if(v[0]!=0 || !std::signbit(v[1]) || v[2]!=std::ldexp(1.f,-24) || v[3]!=1 || v[4]!=-2 || !std::isinf(v[5])) return 2;
    uint16_t b[]={0x3f80,0xc000,0x7fc0};t.type=gguf::BF16;t.data=reinterpret_cast<uint8_t*>(b);t.nbytes=sizeof(b);
    const auto z=gguf::float_values(t);
    if(z[0]!=1 || z[1]!=-2 || !std::isnan(z[2])) return 3;
    std::cout<<"PASS: malformed GGUF rejected and F16/BF16 scalar conversion correct\n";
}
