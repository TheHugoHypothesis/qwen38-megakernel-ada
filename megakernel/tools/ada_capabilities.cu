// Read-only hardware facts for deciding which Ada experiments are applicable.
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
static void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
int main() {
    try {
        int device=0,driver=0,runtime=0;check(cudaGetDevice(&device));
        cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,device));
        check(cudaDriverGetVersion(&driver));check(cudaRuntimeGetVersion(&runtime));
        size_t current=0;check(cudaDeviceGetLimit(&current,cudaLimitPersistingL2CacheSize));
        constexpr size_t state_layer=size_t(48)*128*128*4;
        constexpr size_t state_total=48*state_layer;
        std::printf("{\n  \"device\": %d, \"name\": \"%s\", \"compute_capability\": \"%d.%d\",\n"
            "  \"sms\": %d, \"registers_per_sm\": %d, \"max_threads_per_sm\": %d,\n"
            "  \"shared_per_sm_bytes\": %zu, \"shared_block_optin_bytes\": %zu,\n"
            "  \"l2_bytes\": %d, \"persisting_l2_max_bytes\": %d, \"access_policy_window_max_bytes\": %d,\n"
            "  \"persisting_l2_current_bytes\": %zu, \"cooperative_launch\": %s,\n"
            "  \"gdn_state_one_layer_bytes\": %zu, \"gdn_state_all_layers_bytes\": %zu,\n"
            "  \"driver_api\": %d, \"runtime_api\": %d, \"settings_changed\": false\n}\n",
            device,p.name,p.major,p.minor,p.multiProcessorCount,p.regsPerMultiprocessor,p.maxThreadsPerMultiProcessor,
            p.sharedMemPerMultiprocessor,p.sharedMemPerBlockOptin,p.l2CacheSize,p.persistingL2CacheMaxSize,
            p.accessPolicyMaxWindowSize,current,p.cooperativeLaunch?"true":"false",state_layer,state_total,driver,runtime);
    } catch(const std::exception & e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
