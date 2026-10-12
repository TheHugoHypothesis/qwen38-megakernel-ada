LLAMA ?= $(firstword $(wildcard ../llama.cpp ../../llama.cpp))
override LLAMA := $(shell realpath --relative-to=. "$(LLAMA)")
NVCC ?= /usr/local/cuda/bin/nvcc
# the first GPU's architecture (sm_86 RTX 30, sm_89 RTX 40, sm_120 RTX 50); sm_86 if nvidia-smi is missing
ARCH ?= $(or $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. '),86)
override ARCH := sm_$(patsubst sm_%,%,$(ARCH))
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas -dlcm=cg -Xptxas -v -Xcompiler -fPIC -I$(LLAMA)/ggml/src -I$(LLAMA)/ggml/include

all: qwen38_mk libqwen38mk.so tools/vocab_freq

dflash_native.o: dflash_native.cu dflash_native.h native_quant.cuh gguf.h qwen38_mk.h
	$(NVCC) $(NVFLAGS) -c dflash_native.cu -o $@ 2> dflash_ptxas.log || (cat dflash_ptxas.log; false)

dflash_math_test: tests/dflash_math.cu dflash_native.cu dflash_native.h native_quant.cuh qwen38_mk.o
	$(NVCC) $(NVFLAGS) tests/dflash_math.cu qwen38_mk.o -L$(LLAMA)/build/bin -lggml-base -lggml-cuda -lcublas \
		-Xlinker -rpath -Xlinker '$$ORIGIN/$(LLAMA)/build/bin' -o $@ 2> dflash_math_ptxas.log

dflash_head_test: tests/dflash_head.cu dflash_native.cu dflash_native.h native_quant.cuh qwen38_mk.o
	$(NVCC) $(NVFLAGS) tests/dflash_head.cu qwen38_mk.o -L$(LLAMA)/build/bin -lggml-base -lggml-cuda -lcublas \
		-Xlinker -rpath -Xlinker '$$ORIGIN/$(LLAMA)/build/bin' -o $@ 2> dflash_head_ptxas.log

target_features_test: tests/target_features.cpp qwen38_mk.o qwen38_mk.h
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -I$(LLAMA)/include -I$(LLAMA)/src -I$(LLAMA)/ggml/include tests/target_features.cpp qwen38_mk.o \
		-L$(LLAMA)/build/bin -lllama -lggml-base -lggml-cuda -lcublas -Xlinker -rpath -Xlinker '$$ORIGIN/$(LLAMA)/build/bin' -o $@

# ptxas register / spill report for every phase function goes to ptxas.log
qwen38_mk.o: qwen38_mk.cu qwen38_mk.h gguf.h kv_quant.cuh native_quant.cuh native_iq_mma.cuh native_prefill.cuh model_spec.h prefill.cuh prefill_gemm.cuh $(LLAMA)/ggml/src/ggml-common.h
	$(NVCC) $(NVFLAGS) -c qwen38_mk.cu -o $@ 2> ptxas.log || (cat ptxas.log; false)

qwen38_mk: qwen38_mk.o main.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -I$(LLAMA)/include -I$(LLAMA)/ggml/include main.cu qwen38_mk.o \
		-L$(LLAMA)/build/bin -lllama -lggml-base -lggml-cuda -lcublas -Xlinker -rpath -Xlinker '$$ORIGIN/$(LLAMA)/build/bin' -o $@

# engine + tokenizer as a shared library with a C API (capi.cpp), used by server/mk_server.py
libqwen38mk.so: qwen38_mk.o dflash_native.o capi.cpp qwen38_mk.h dflash_native.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -shared -Xcompiler -fPIC -I$(LLAMA)/include -I$(LLAMA)/ggml/include capi.cpp qwen38_mk.o dflash_native.o \
		-L$(LLAMA)/build/bin -lllama -lggml-base -lggml-cuda -lcublas -Xlinker -rpath -Xlinker '$$ORIGIN/$(LLAMA)/build/bin' -o $@

tools/vocab_freq: tools/vocab_freq.cpp
	g++ -O2 -std=c++17 -I$(LLAMA)/include -I$(LLAMA)/ggml/include $< -L$(LLAMA)/build/bin -lllama \
		-Wl,-rpath,'$$ORIGIN/../$(LLAMA)/build/bin' -o $@

clean:
	rm -f kv_quant_test kv_attention_test kv_attention_ptxas.log qwen38_mk libqwen38mk.so qwen38_mk.o dflash_native.o ptxas.log dflash_ptxas.log dflash_math_ptxas.log dflash_head_ptxas.log iq_mma_ptxas.log iq_ffn_ptxas.log tools/vocab_freq native_quant_test gguf_parser_test model_schema_test native_gemm_test iq_mma_test bench/iq_ffn_bench qwen38_dflash external_verify_test dflash_math_test dflash_head_test target_features_test

# Native DFlash2 plus a selectable GGML reference for numerical validation.
qwen38_dflash: qwen38_mk.o dflash_native.o dflash_main.cpp dflash_bridge.cpp dflash_bridge.h qwen38_mk.h $(LLAMA)/src/llama-model.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -I$(LLAMA)/include -I$(LLAMA)/src -I$(LLAMA)/ggml/include -I$(LLAMA)/ggml/src \
		dflash_main.cpp dflash_bridge.cpp qwen38_mk.o dflash_native.o -L$(LLAMA)/build/bin -lllama -lggml-base -lggml-cuda -lcublas \
		-Xlinker -rpath -Xlinker '$$ORIGIN/$(LLAMA)/build/bin' -o $@

external_verify_test: qwen38_mk.o tests/external_verify.cpp qwen38_mk.h gguf.h
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -I$(LLAMA)/ggml/include tests/external_verify.cpp qwen38_mk.o \
		-L$(LLAMA)/build/bin -lggml-base -lggml-cuda -lcublas -Xlinker -rpath -Xlinker '$$ORIGIN/$(LLAMA)/build/bin' -o $@

iq_mma_test: tests/iq_mma.cu native_iq_mma.cuh native_iq_pipeline.cuh native_quant.cuh gguf.h
	$(NVCC) $(NVFLAGS) $< -L$(LLAMA)/build/bin -lggml-base \
		-Xlinker -rpath -Xlinker '$$ORIGIN/$(LLAMA)/build/bin' -o $@ 2> iq_mma_ptxas.log

bench/iq_ffn_bench: bench/iq_ffn_bench.cu native_iq_mma.cuh native_iq_pipeline.cuh native_quant.cuh gguf.h
	$(NVCC) $(NVFLAGS) $< -o $@ 2> iq_ffn_ptxas.log

# Numerical oracle for the native formats, using actual model tensor blocks.
native_quant_test: tests/native_quant.cu native_quant.cuh gguf.h $(LLAMA)/ggml/src/ggml-common.h
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -I. -I$(LLAMA)/ggml/src -I$(LLAMA)/ggml/include $< \
		-L$(LLAMA)/build/bin -lggml-base -Xlinker -rpath -Xlinker '$$ORIGIN/$(LLAMA)/build/bin' -o $@

gguf_parser_test: tests/gguf_parser.cpp gguf.h
	$(CXX) -O2 -std=c++17 $< -o $@

model_schema_test: tests/model_schema.cpp model_spec.h gguf.h
	$(CXX) -O2 -std=c++17 $< -o $@

native_gemm_test: tests/native_gemm.cu native_quant.cuh native_prefill.cuh prefill_gemm.cuh gguf.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -I. -I$(LLAMA)/ggml/src -I$(LLAMA)/ggml/include $< \
		-L$(LLAMA)/build/bin -lggml-base -lcublas -Xlinker -rpath -Xlinker '$$ORIGIN/$(LLAMA)/build/bin' -o $@

kv_quant_test: tests/kv_quant.cu kv_quant.cuh
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) $< -o $@

kv_attention_test: tests/kv_attention.cu qwen38_mk.cu qwen38_mk.h gguf.h prefill.cuh prefill_gemm.cuh kv_quant.cuh native_quant.cuh native_iq_mma.cuh native_iq_pipeline.cuh native_prefill.cuh model_spec.h
	$(NVCC) $(NVFLAGS) -Xptxas -O1 $< -L$(LLAMA)/build/bin -lggml-base -lggml-cuda -lcublas \
		-Xlinker -rpath -Xlinker '$$ORIGIN/$(LLAMA)/build/bin' -o $@ 2> kv_attention_ptxas.log
