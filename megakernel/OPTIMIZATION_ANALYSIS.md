# Performance analysis and optimization priorities: RTX 4060 Ti 16 GB

This report records the investigation before the later implementation work. The authorized IQ16, prefill, and automatic-chunk changes were subsequently implemented and measured at roughly +25% MK decode throughput; DFlash2 Q4 through llama.cpp reached 91.35 tok/s on the short prompt. The tiled GEMM prototype was slower and remains opt-in. For current results, see [IMPLEMENTATION_RESULTS.md](IMPLEMENTATION_RESULTS.md) and the [intermediate implementation notes](bench/optimization-implementation/README.md). Statements below about work not yet implemented describe the investigation date, not the current code.

The investigation and measurements were performed on 2026-10-10 at stock clocks with CUDA 13.4, `sm_89`, and the existing GSQ-RCO IQ3_S-MTP target. That stage added profiles, benchmarks, and reports; it did not modify inference kernels or binaries.

## Baseline and interpretation

The megakernel fit the memory budget but initially trailed tuned llama.cpp MTP. An earlier +4.6% comparison used two llama.cpp proposals versus four in MK, so it did not establish an advantage over the best available baseline.

| 128 generated tokens, 4096 context, Q8 KV | llama.cpp MTP4 | MK MTP4 | MK difference |
|---|---:|---:|---:|
| Greedy, 24-token prompt | 62.956 tok/s | 48.170 tok/s | -23.49% |
| Sampling, temp 1/top-k 20/top-p 0.95/seed 123 | 61.931 tok/s | 47.235 tok/s | -23.73% |
| Greedy, 1365-token prompt | 45.509 tok/s | 39.925 tok/s | -12.27% |

The new llama.cpp short greedy baseline averaged three repetitions; the other cases averaged two. MK values came from earlier runs using the same binary, settings, and clocks, not an interleaved controlled test. Decode throughput excludes model load and prefill. No confidence interval was available. Prompt token IDs were matched to avoid template/tokenization differences. Long continuations differed, so that row compares observed generations rather than a fixed continuation; speculative acceptance depends on the generated text.

The llama.cpp proposal-depth sweep measured MTP3 54.10, MTP4 62.96, MTP5 63.02, MTP6 62.90, and MTP7 63.21 tok/s. Differences below 0.5% across MTP4–7 do not establish that seven is faster; MTP4 is the comparable reference depth. Previous non-MTP measurements were 20.92 tok/s for llama.cpp and 16.42 for MK.

Raw data and chart inputs are under `bench/optimization-analysis/`, `bench/ada-report/chart_data.json`, and the archived report. The old chart was corrected to identify its baseline and numerical divergence.

## Profile findings and priorities

The main opportunity was the target feed-forward network and its quantized matrix-vector paths, not DeltaNet alone. Focus on the operations that dominate complete decode time and preserve direct decoding of the original mixed-IQ GGUF. Candidate work included packed IQ dequantization, gate/up fusion, reducing intermediate memory traffic, and tuning row/warp tiling for the 34-SM Ada GPU. Keep exact-format fallbacks for rare types such as IQ1_M.

The prefill path was a separate performance problem from token decode. `pf_init`, `gemm`, `gemm_glu`, and `prefill.cuh` were identified for profiling. Reusing llama.cpp MMQ loaders/tiles could reduce redundant work, but an activation-Q8 path would change current FP16 numerical behavior and needs its own validation. FP16 tile dequantization is a lower-risk alternative. Double-buffering a temporary matrix would add roughly 170 MiB and should be considered only when measurements show useful overlap and memory permits.

Speculative decoding should be tuned by accepted tokens per second, not acceptance rate alone. The correct draft depth depends on per-position acceptance and the measured draft cost. Draft-only vocabulary pruning may reduce draft output cost, but the target must still verify the full vocabulary. It must be compared against the full head cost, memory use, and changes to proposals.

The official [Qwen3.8 DFlash2 GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) provides Q4_K_M (about 1.14 GB) and Q8_0 (about 2.06 GB). At the time of this investigation it had not yet been downloaded or tested, so compatibility and acceptance with the GSQ-RCO target were hypotheses. Its configuration described five draft blocks, hidden size 5120, block size eight, two-tap convolution, selector, and target taps `[5,19,33,47,61]`; the local converter stores corresponding GGUF layer inputs at `[6,20,34,48,62]`. Always check the actual draft metadata before porting.

The existing llama.cpp speculative implementation (`common/speculative.cpp`, `src/models/dflash.cpp`, and `common/arg.cpp`) was the preferred reference before writing new code. Comparisons should use the same target, GPU, prompt IDs, context, and proposal depth. Do not compare a published speedup against plain autoregressive decoding with a local baseline that already uses MTP.

The draft Q4 file alone was about 1.06 GiB. A rough draft KV estimate was 40 MiB FP16 or 21 MiB Q8 for five layers, eight KV heads, dimension 128, and a 2048-token window, before features and workspace. At 32K the MK target used about 14 GiB, leaving little room; Q8 was estimated not to fit the observed 15.65 GiB CUDA budget with all extras. These were estimates, not runtime DFlash measurements.

A full DFlash2 port required target feature taps, draft execution, selector, proposal verification, and DeltaNet rollback. The existing `MAXT=5` and attention padding also constrained larger blocks. Merely increasing the CLI proposal count would not implement this path and could exceed shared memory. The later native path is documented in [DFLASH_NATIVE.md](DFLASH_NATIVE.md).

## Architecture and lower-priority ideas

If fused microkernels cannot beat the baseline, separating large GEMMs allows independent grid, tile, and register tuning while preserving useful normalization, residual, and DeltaNet fusion. CUDA Graphs already reduce launch overhead in llama.cpp; adding graphs around the current one-kernel-per-cycle MK path would not address the measured dominant cost by itself.

DeltaNet distributes 48 heads over 34 SMs, leaving 14 SMs with two heads and the rest with one. This imbalance was visible but the core represented only about 2.2% of the measured profile. Q4 KV or compressed recurrent state primarily address long-context memory pressure and introduce numerical error; they do not target the main decode cost. FP16 copies of all weights do not fit. TMA/WGMMA and native NVFP4 features from later GPU generations are not direct solutions for `sm_89`.

## Validation plan and success criteria

1. Fix baselines for non-speculative and MTP4 with identical prompt IDs, context, output length, and sampling settings.
2. Build microbenchmarks around real target weights for FFN and projections at T=1–5.
3. Measure candidates without instrumentation after warm-up, with at least five repetitions and interleaved ordering.
4. Test both allocated and populated context at 4K, 8K, and 32K; allocated context alone does not represent long-context decode cost.
5. Evaluate perplexity/NLL on the same corpus and tasks with known answers.

Success means beating tuned llama.cpp on representative workloads by more than measurement variability, fitting the memory budget, and preserving agreed quality limits. A local kernel improvement must also improve complete request latency without material regression elsewhere. The roughly 63 tok/s short-prompt value is a specific reference, not a throughput guarantee for all text.

Proposed initial quality limits for approximate paths were less than 0.5% relative perplexity increase and less than one percentage point of accuracy loss, measured with adequate sample sizes and confidence intervals. These were proposed criteria, not measured results. Weight-layout changes should preserve exact dequantization of the source format.

Evidence is retained under `bench/optimization-analysis/`; `bench/ada_summary.json` contains original result hashes and this follow-up. The archived `bench/ada-report/README.md` explains the baseline and divergence. External primary sources were consulted on 2026-10-10.
