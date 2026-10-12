// Full GGUF schema validation without CUDA allocations.
#include "../model_spec.h"
#include <cstdio>
int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: model_schema_test model.gguf");
        gguf::File file(argv[1]);
        const auto main_bytes = gguf::validate_qwen38(file, false);
        const auto all_bytes = gguf::validate_qwen38(file, true);
        std::printf("PASS vocab=%llu tensors=%zu main_device_weight_bytes=%zu all_device_weight_bytes=%zu\n",
            (unsigned long long)file.get("token_embd.weight").ne[1], file.tensors.size(), main_bytes, all_bytes);
    } catch (const std::exception & e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
