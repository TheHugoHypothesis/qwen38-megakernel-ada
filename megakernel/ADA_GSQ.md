# RTX 4060 Ti 16 GB and Qwen3.8 GSQ-RCO

This note documents the Ada adaptation for the ISTA-DASLab GSQ-RCO target GGUF. The file is a non-uniform mixed-precision model; it is not the standard Qwen3.8-27B Q4_K_M target. The validated snapshot was `d562806dbafae37109975e970aae91b43e73b440`. The GGUF is 12,120,016,960 bytes (11.288 GiB), with a tensor payload of 12,109,021,184 bytes. Its embedding is IQ2_S, output is Q4_K, and all eight MTP matrices are Q6_K. Only 25 of the 65 FFN gate/up pairs share a dtype.

| Tensor type | Count | Tensor type | Count |
|---|---:|---|---:|
| F32 | 360 | BF16 | 96 |
| IQ3_S | 144 | IQ3_XXS | 78 |
| IQ4_XS | 96 | IQ2_S | 17 |
| IQ2_XS | 9 | IQ2_XXS | 5 |
| IQ1_M | 1 | Q2_K | 13 |
| Q4_K | 39 | Q6_K | 8 |

The loader validates `qwen35`, 64 main blocks plus one MTP block, embedding width 5120, FFN width 17408, 24 query heads, 4 KV heads, and vocabulary size 248,320. It checks metadata, tensor names, shapes, and types across all 866 tensors before upload. A filename alone does not make an unsupported model eligible.

## Build and run

From the package root, build llama.cpp for CUDA and Ada, then build the megakernel with `ARCH=89`. The Makefile consumes local llama.cpp headers and libraries; `libllama` provides tokenization/template support and cuBLAS supports fallback prefill. Rebuild CUDA objects after changing architecture, llama.cpp headers, or source (`make -B`). `ARCH=89` and `ARCH=sm_89` are equivalent. See the top-level README for complete commands.

Example text server (the target model alone uses roughly 13–14 GiB of device memory, so do not run it beside another large GPU process):

```sh
MODEL=models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF/snapshots/d562806dbafae37109975e970aae91b43e73b440/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
python megakernel/server/mk_server.py -m "$MODEL" \
  --port 18080 -c 32768 --kv q8 --draft 4 --prefill-chunk auto
```

Open-Jet's `llama_backend: megakernel` selects the custom engine explicitly. Its `auto` mode checks the local GGUF and GPU, and keeps llama-server as a startup fallback. Ada eligibility requires `sm_89` and at least 15,360 MiB of reported capacity; the CUDA loader separately checks currently free memory against the requested context.

## CUDA changes for mixed quantization

- `native_quant.cuh` decodes IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ1_M, and Q2_K without rewriting the GGUF. It expands eight integer weights with scale and offset for DP4A, shares those decoded weights across verified tokens, and uses vectorized/read-only cache accesses for lookup tables.
- `qwen38_mk.cu` distributes the 48 DeltaNet value heads by the actual CTA count, including GPUs with fewer than 48 SMs. Projection, embedding, gate/up, output, and MTP kernels dispatch on each tensor's real type. Gate and up matrices may have different types and row strides. BF16/F16 alpha/beta values are widened to F32 before the existing norm folding.
- IQ3_S, IQ3_XXS, and IQ4_XS use compile-time-specialized helpers to avoid repeated type dispatch. The existing Q6_K path is reused for MTP, whose eight matrices all use that type. IQ accumulators stay in registers until reduction completes, avoiding repeated local-memory traffic.
- The cooperative grid uses `cooperative_groups::this_grid().sync()`. Attention initializes matrix padding and softmax factors; split-K is capped at the SM count. `NT=512` is retained because the layout depends on 16 warps.
- `prefill.cuh` retains existing specialized GEMMs. New formats are dequantized one matrix at a time to FP16 and multiplied with cuBLAS using F32 accumulation and reusable buffers. Mixed gate/up types use two GEMMs and the same SiLU operation. Workspace size does not grow with layer count.
- Five-token down projections use two K slices and register partial sums. Residual projections stop on non-finite values, preventing NaNs from silently becoming token IDs.
- `MK_PREFILL_CHUNK` / `--prefill-chunk` accepts multiples of 128 from 128 to 2048. `auto` selects a chunk based on free VRAM (the former default was 512). Logit-scoring microbatches are capped by the actual `y0` capacity to avoid overflow at small chunks.

Q6_K and Q8_0 retain their existing repack, with values unchanged. The global `general.file_type` is not a substitute for per-tensor inspection. `mmproj` is not loaded: this backend is text-only and single-sequence.

## VRAM budget

The following measurements used this GGUF, prefill chunk 512, and a draft vocabulary of 32,768 plus 4,096 additional positions.

| Component | Memory |
|---|---:|
| Weights, including alpha/beta conversion and Q6 repack | 11.343 GiB |
| Restricted output copy for draft | 0.099 GiB |
| DeltaNet F32 state | 144 MiB |
| Convolution ring | 15 MiB |
| DeltaNet replay | 24.28 MiB |
| Checkpoint | 159.31 MiB |
| Prefill buffers | about 161.7 MiB |
| IQ fallback: weights + input + output | 170 + 17 + 34 MiB |
| KV F16, 17 slots | 69,632 bytes/token |
| KV Q8, including scales | 36,992 bytes/token |
| RoPE | 256 bytes/token |

At 4,096 context with Q8, the logger measured **12.297 GiB** of engine allocations and about **13.020 GiB** of CUDA-visible total usage, including runtime, cuBLAS, and other occupants. Values vary with active processes. Approximate totals were 14.01–14.03 GiB at 32K/Q8, 15.016 GiB at 32K/F16, and 15.14–15.16 GiB at 65K/Q8. Auto chose F16 at 32K and Q8 at 65K. The 65K run left about 0.50 GiB free and required an otherwise idle GPU. A final HTTP run reported 14,350/16,380 MiB. The 65K test allocated the full cache but generated from a short prompt; it does **not** validate quality for a 65K-token prompt. A 262,144/Q8 request was rejected before upload because its estimate was 21.75 GiB.

`--kv auto` compares free memory with the estimated budget and selects F16 or Q8 before loading weights. Requests that do not fit are rejected; the engine does not silently offload layers or reduce context. The estimate includes runtime, checkpoint, and cuBLAS fallback reserves. Disabling MTP skips draft computation and its output copy, but keeps MTP block weights loaded to preserve the engine layout.

## Reproducible validation

```sh
make -C megakernel ARCH=89 native_quant_test gguf_parser_test
megakernel/native_quant_test "$MODEL"
megakernel/gguf_parser_test
python megakernel/bench/ada_bench.py -m "$MODEL" \
  --llama-server llama.cpp/build/bin/llama-server \
  --output /tmp/ada-benchmark --ctx 4096 --tokens 128 --reps 2 \
  --drafts 0,1,2,3,4 --draft-vocab megakernel/draft_vocab_32k.bin
```

The CUDA test compares 8,945,664 values from real rows with llama.cpp's independent CPU dequantizers, including IQ16 dot products and the repacked Q6 path. The earlier version checked 5,038,080 values. The backend/GGUF/build suite passed 34/34 tests. `gpu_server_smoke.py` checked seven real HTTP/CUDA cases, including streaming, seed, a second turn, and a 1,351-token prompt. Compute Sanitizer memcheck reported zero errors over three instrumented decode cycles in a 16-token test; this is not exhaustive. Host tests cover truncated files and F16/BF16 subnormal conversion.

The benchmark warms llama-server, sends the same prompt IDs to both engines, records sequences/repetitions/timing, and teacher-forces the reference continuation. `--temperature 1 --top-k 20 --top-p .95` enables sampling. Load and prefill are excluded from decode throughput and should be compared separately from acceptance.

`MK_PROFILE=1` reports per-phase times; `MK_PF_PROFILE=1` measures prefill. `MK_CHECK_FINITE=1` identifies the first phase producing non-finite values. These profiling modes alter timing and must not be used for throughput claims. Keep Compute Sanitizer runs separate from benchmarks.

## Quality and limits

Server defaults match this model's GGUF metadata: temperature 1, top-k 20, top-p 0.95, and min-p 0. The engine supports top-k up to 32. Unsupported penalties and larger top-k values return errors rather than silently changing sampling. Use temperature 0 and neutral penalties for greedy comparisons.

Weight dequantization preserves the stored weights, but decode's Q8 activations and the original engine's FP16 prefill remain approximate. A matching continuation does not establish general quality. Perplexity corpora and long prompts require evaluation before treating this engine as a universal llama.cpp replacement. No training or GGUF modification was performed.

Local regression coverage included greedy, seeded sampling, streaming, a second turn, and prefill across chunks. These are functional/numerical checks, not a broad perplexity or quality evaluation.

## RTX 4060 Ti performance and stability

The historical initial baseline used llama.cpp MTP 2. A later MTP 4 comparison reached 62.96 tok/s for short greedy, 61.93 for sampling, and 45.51 for long greedy. The megakernel was 23.49%, 23.73%, and 12.27% slower respectively; long continuations differed. The early 4.6%/2.7% gains therefore do not establish an advantage over tuned llama.cpp.

On the long MTP 4 greedy test, llama.cpp matched its no-speculation reference and diverged from the megakernel/MTP 2 reference. Teacher forcing agreed at 127/128, with the first disagreement at index 41. Cause isolation and broad quality evaluation remain open. The newer profile assigns about 59% of decode to the FFN and under 1% to attention for these prompts. Chunk 2048 improved prefill from 530.5 to 630.2 tok/s, used about 0.629 GiB more, and kept decode near 40 tok/s. See [`OPTIMIZATION_ANALYSIS.md`](OPTIMIZATION_ANALYSIS.md) and the benchmark notes.

The reference was llama.cpp `aa94f2086` (build 11549), CUDA 13.4, `sm_89`, context 4096/Q8, single sequence. Clocks and voltage boost had no offsets in LACT; the original fan curve was retained. Rows below use two 128-token repetitions except the llama.cpp greedy MTP baseline, which uses three. Decode excludes load and prefill; megakernel's first token is produced by prefill.

| Case | llama-server | Megakernel | Note |
|---|---:|---:|---|
| Greedy, 24-token code prompt, no MTP | 20.92 tok/s | 16.42 tok/s | Plain decode favored llama.cpp |
| Same prompt, MTP | 46.04 tok/s, depth 2 | 48.17 tok/s, depth 4 | About +4.6% |
| Sampling, temp 1, top-k 20, top-p 0.95, seed 123 | 46.01 tok/s, depth 2 | 47.24 tok/s, depth 4 | About +2.7% |
| Greedy, 1,365-token prompt, MTP | 40.91 tok/s, depth 2 | 39.93 tok/s, depth 4 | llama.cpp about 2.5% faster |
| Prefill, 1,365-token prompt | about 740 tok/s | about 529 tok/s | IQ fallback limited by workspace/dequantization |

For the short prompt, depths 0/1/2/3/4 reached 16.42 / 28.67 / 39.33 / 46.64 / 48.17 tok/s. All ten greedy generations exactly reproduced the reference's 128 IDs. The four long greedy generations also matched the 128 reference IDs. Teacher forcing agreed 128/128 on both prompts, including long prefill at chunk 128. Sampling uses different RNG implementations, so identical sampled IDs are not a valid distribution-equivalence criterion. Draft acceptance fell from about 89% on the short prompt to 69% on the long prompt.

The final kernel uses 512 threads/CTA, 128 registers/thread, and 92,544 bytes dynamic shared memory. It launches 34 cooperative CTAs (one per SM), with 16 resident warps per SM (about 33% occupancy). These constraints make increasing CTA count or changing NT without changing layouts unsafe. Projections and FFN dominate; compile-time-specialized IQ helpers and Q6_K specialization produced real gains. Some compile spills remain. Lowering register/spill pressure or writing workspace-free IQ prefill GEMMs are future work, not measured gains.

## Clocks and stability

The initial LACT profile used +190 core and +1,200 memory offsets, voltage boost 8, and explicit clock limits. Illegal/misaligned accesses and corrupted generations appeared with that profile, including in llama.cpp MTP. With offsets reset to zero, repetitions were stable. An A/B test with the same binary passed three times at stock settings and failed once in two repetitions under the prior profile. This associates the issue with the clock configuration but does not isolate a single setting or certify general hardware stability.

The user requested stock clocks. They were restored and confirmed through the [LACT API](https://github.com/ilya-zlobintsev/LACT/blob/master/docs/API.md); the fan curve and PowerMizer Adaptive setting were preserved. The prior-profile failures are not established as an inherent llama.cpp MTP limitation: with stock clocks, it passed all three repeats.
