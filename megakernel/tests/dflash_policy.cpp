#include "../dflash_verify_policy.h"
#include <cstdio>
#include <stdexcept>
static void require(bool ok){if(!ok)throw std::runtime_error("verification policy assertion");}
int main() {
 try {
    DFlashVerifyPolicy full(7),fixed(7,"3"),good(7,"auto"),bad(7,"auto");
    for(int i=0;i<40;i++){require(full.width()==7);full.observe(7,0);require(good.width()==7);good.observe(7,7);}
    require(fixed.width()==3);
    for(int i=0;i<12&&bad.width()!=3;i++)bad.observe(7,0);
    require(bad.width()==3);
    for(int i=0;i<8;i++){require(bad.width()==3);bad.observe(3,0);}
    require(bad.width()==7); // probe a potentially changed task
    bad.observe(7,7);bad.reset();require(bad.width()==7&&bad.three_cycles==0&&bad.full_cycles==0);
    DFlashVerifyPolicy shallow(2,"auto");require(shallow.width()==2);
    bool invalid=false;try{DFlashVerifyPolicy nope(7,"wrong");}catch(const std::exception &){invalid=true;}require(invalid);
    std::puts("PASS full/fixed/auto, warmup, acceptance, censoring, probe, reset and validation");
 }catch(const std::exception & e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
