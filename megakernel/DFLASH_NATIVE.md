# Native DFlash2 in the MK backend: RTX 4060 Ti 16 GB

The target and verifier run through the cooperative megakernel. The full DFlash2 decoder runs as native CUDA kernels captured in CUDA Graphs; draft inference does not call `llama_decode`, `llama_process`, or execute a GGML graph. GGML is retained for shared-weight descriptors/buffers, and libllama provides tokenization. The CLI keeps a GGML reference backend for numerical oracles.

**The draft has not been fused into one persistent megakernel.** “Native” here means the draft computation uses the custom CUDA implementation and CUDA Graph orchestration. A monolithic draft kernel and further IMMA fusion remain future experiments, not delivered features.

## Build and launch

From the workspace root:

```sh
make -C open-jet/megakernel ARCH=89 all qwen38_dflash
python3 open-jet/megakernel/serve_ada.py \
  --engine megakernel-dflash2 --ctx 4096 --draft 7
```

Open-Jet can select the megakernel backend with `OPENJET_MK_DFLASH_MODEL` pointing at the draft GGUF. `OPENJET_MK_DRAFT=7` is optional. The build cache includes the native draft sources. The server is serial and handles one sequence at a time.

The Python server and C API provide sampling, checkpoint, rewind, truncation, reset, streaming, and prefix reuse. The draft is destroyed before the target because it borrows target embedding/head storage. The CLI accepts a GGML reference backend for tests; production native DFlash2 uses CUDA Graphs. `Makefile`, `serve_ada.py`, `src/llama_server.py`, and `src/megakernel_build.py` contain the build and launcher integration.

## Unsloth Studio request compatibility

The HTTP adapter accepts sampling options supported by the megakernel and returns a clear error for unsupported options. `presence_penalty` support was added to avoid the Studio request rejection. Reasoning effort is read from the top-level field as well as supported aliases, because some clients send it outside `chat_template_kwargs`. `high` maps to the available `xhigh` setting; `none` disables reasoning. Unspecified values retain the model's GGUF default. See [stage 23](bench/dflash-port/23-UNSLOTH-SAMPLING-COMPATIVEL.md) and [stage 24](bench/dflash-port/24-UNSLOTH-REASONING-EFFORT.md).

## Optional Q4 target KV cache

The launcher accepts `--kv q4` for the **target attention cache**. Target weights retain their original mixed GSQ-RCO quantization. The native DFlash2 sliding-window cache stays FP16. Q8 remains the default; `auto` selects FP16 or Q8.

```sh
python3 open-jet/megakernel/serve_ada.py \
  --engine megakernel-dflash2 --kv q4 --ctx 65536 --prefill-chunk 128
```

This cache uses symmetric signed INT4 and one FP16 scale per 32 values, then dequantizes tiles into the existing FP16 attention kernels. It is a custom cache layout, not GGML Q4_0. Target-cache storage decreases from 36,992 to 19,584 bytes per context position across 17 slots, saving 1.0625 GiB at 65,536 tokens. Preflight also includes the draft, workspaces, and CUDA/library reserve. Context length includes prompt and generated output; actual fit depends on available VRAM. Start below the desired maximum when other applications use the GPU.

Q4 is lossy and changes target logits; speculative verification follows that Q4 target distribution and does not reproduce Q8 behavior. Validation and measured trade-offs are in [stage 26](bench/dflash-port/26-Q4-TARGET-KV.md).

## VRAM accounting

| Component | Approximate allocation |
|---|---:|
| Target weights without MTP head | 11 GiB; omitting the MTP weights saves about 0.343 GiB |
| Target Q8 KV | 17 slots × 4 heads × context × 544 bytes; about 144.5 MiB at 4K and 1.129 GiB at 32K |
| DeltaNet state and checkpoint | about 144 MiB each |
| Target convolution ring and checkpoint | about 30 MiB each |
| Feature capture | 5 × 5120 × 4 bytes × prefill capacity; 12.5–200 MiB |
| Initial draft total | about 1.129 GiB including weights, caches, and small buffers |
| Draft FP16 KV and checkpoint | about 40.16 MiB each |
| Encoder staging | largest FP16 matrix about 250 MiB plus about 25 MiB input, allocated on demand |
| CUDA, libraries, and graphs | additional reserve; measure actual device use, not just `cudaMalloc` totals |

These are planning estimates, not a promise that every combination fits. Runtime use depends on prefill workspace and other GPU users. The server prints preflight and runtime allocations; use those readings when setting context.

## Validation commands and limits

Build test oracles with:

```sh
make -C open-jet/megakernel ARCH=89 \
  external_verify_test dflash_math_test target_features_test
```

The external verifier checks depths 0–7, all rejection positions (36 cases), scalar continuation, shared bytes, and feature layout. `dflash_math_test DRAFT` checks CPU GGML weights, F32 and Q8/DP4A against double-precision products, FP16 staging, both draft convolutions, norm/RoPE/cache, and sliding attention past position 2050. CLI `--oracle` compares aligned candidate scores with the GGML FP16 draft. `target_features_test mk|llama TARGET ids output.f32` teacher-forces identical IDs and extracts five taps at three positions. `bench/dflash_port_bench.py`, `dflash_port_quality.py`, and `tests/gpu_server_smoke.py --draft-model DRAFT --draft 7` cover performance, output, sampling, HTTP, and sliding-window attention.

Memcheck uses `--error-exitcode 99`; exact commands and filter scope are in `native-final-validation.json`. Filtered launch counts are not equivalent to sanitizing every context. See the [implementation log](DFLASH_PORT_LOG.md) and [Nsight notes](bench/dflash-port/10-NSIGHT-E-LEITURA-PACKED.md). The llama.cpp reference checkout at `aa94f2086147cd4bfbcff818ce998dffce5fcb39` remains unmodified.

## Results and limits

Seven proposals and packed reads raised native-draft throughput substantially over the initial hybrid prototype. Later target IQ and typed-MMQ work reached 93.21/62.28/93.36 tok/s short/long/sampling across six samples. This was 15.18%/15.40%/15.88% above the prior MK snapshot, but only +1.49% on the short llama.cpp comparison and -0.34% on the long one. There is no demonstrated general win over llama.cpp. The short-case advantage is small and should be rechecked under a larger repeated benchmark.

Q4 KV primarily helps memory headroom; it does not improve throughput consistently and changes some generated outputs. Q8 remains the default. Long teacher-forcing divergence in the target and limited quality-set coverage remain open. See [stage 11](bench/dflash-port/11-ENTREGA-NATIVA-RESULTADOS.md), [stage 22](bench/dflash-port/22-RESULTADO-IQ-E-DRAFT-NATIVO.md), and [stage 26](bench/dflash-port/26-Q4-TARGET-KV.md).
