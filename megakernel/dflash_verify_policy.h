#pragma once
#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>

// Only truncate verification; never shorten the noncausal trained draft block.
// Decisions depend on accepted-prefix counts, not noisy wall-clock samples.
class DFlashVerifyPolicy {
    int maximum_, since_full_=0, observations_=0;
    std::string mode_;
    std::array<double,8> survival_{};
public:
    long three_cycles=0,full_cycles=0;
    explicit DFlashVerifyPolicy(int maximum,const char * mode=nullptr):maximum_(maximum),mode_(mode?mode:"full") {
        if(maximum<1||maximum>7||(mode_!="full"&&mode_!="3"&&mode_!="auto"))
            throw std::runtime_error("MK_DFLASH_VERIFY must be full, 3 or auto (draft depth 1..7)");
        reset();
    }
    static const char * environment() {return std::getenv("MK_DFLASH_VERIFY");}
    void reset() {survival_.fill(1.);since_full_=observations_=0;three_cycles=full_cycles=0;}
    int width() const {
        if(mode_=="full"||maximum_<=3)return maximum_;
        if(mode_=="3")return 3;
        if(observations_<4||since_full_>=8)return maximum_;
        double small=1,full=1;
        for(int i=1;i<=maximum_;i++){full+=survival_[i];if(i<=3)small+=survival_[i];}
        // Conservative Ada cost threshold; auto remains opt-in and benchmarked.
        return small>=.85*full?3:maximum_;
    }
    void observe(int proposed,int accepted) {
        if(proposed<1||proposed>maximum_||accepted<0||accepted>proposed)
            throw std::runtime_error("invalid accepted-prefix observation");
        if(proposed==3)three_cycles++;
        if(proposed==maximum_){full_cycles++;since_full_=0;}else since_full_++;
        observations_++;
        for(int i=1;i<=maximum_;i++) {
            // Rejection reveals all later prefix events are false. A fully
            // accepted short verification censors only the unverified suffix.
            if(i<=proposed||accepted<proposed)survival_[i]+=.25*((i<=accepted?1.:0.)-survival_[i]);
        }
    }
};
