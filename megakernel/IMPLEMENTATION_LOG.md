# Ada optimization implementation log

## 2026-10-10 — work begins

The authorized scope was to implement the proposed changes and keep an intermediate Markdown record. GPU clocks were to remain at stock settings. The GSQ-RCO weights and llama.cpp reference checkout were left unchanged. The GPU was idle at the start, with 41 MiB in use.

The baseline was preserved under `bench/optimization-implementation/baseline/`, including sources, objects, binaries, and hashes. This supports A/B comparisons and recovery of candidates that fail to improve real performance.

Initial reference measurements were about 62.96 tok/s for llama.cpp MTP4 on a short prompt and 48.17 tok/s for megakernel MTP4. Long teacher forcing matched 127/128 tokens, with the first divergence at index 41. General numerical equivalence had not been established.

### Work sequence

1. Pack IQ3_S, IQ3_XXS, and IQ4_XS weights in groups of 16 while preserving DP4A and the original GGUF; compare dequantization and dot products against an independent oracle.
2. Select prefill chunk size automatically within a VRAM budget, while preserving explicit overrides and existing KV-precision preferences.
3. Run A/B measurements and investigate numerical divergence; do not infer quality from a small number of argmax comparisons.
4. Evaluate resource use, tiled prefill, and MTP adaptation based on measured gains. Experimental candidates become defaults only after validation.
5. Obtain the official DFlash2 Q4 draft and test its support in the existing llama.cpp checkout before deciding whether to port it into the megakernel.

Per-stage records are in `bench/optimization-implementation/`. The main success criterion is measured generation performance, with memory and correctness documented; adding flags or kernels alone does not count as an improvement.

## 2026-10-10 — implementation completed and validated

IQ16 support is active for IQ3_XXS, IQ3_S, and IQ4_XS. Automatic chunk sizing, a cuBLAS prefill path with fewer copies, MTP4 by default, and a logits probe were delivered. The final interleaved A/B measured +25.54% short, +24.81% long, and +25.05% sampling throughput, with the same IDs as the original engine. Long-prefill latency fell by 22.78%. A tiled GEMM passed validation but was slower, so it remains experimental and disabled.

The official DFlash2 Q4 draft was downloaded and compatibility was verified with the real target. It reuses local llama.cpp: 91.35/62.29 tok/s greedy short/long and 92.56 tok/s sampling. `serve_ada.py` selects this path by default. Both launchers passed real API checks; DFlash2 also passed with 32K allocated context and fit disabled. No draft port into `mk_cycle` was completed at this stage.

Validation covered 8,945,664 real values in the oracle, GEMMs against CPU/cuBLAS, 34 host tests, and four-cycle memcheck with no errors. All three engines solved 10/10 numeric tasks. DFlash2 produced 17/20 generations identical to the non-speculative baseline; MK produced 16/20. The long MK teacher-forcing comparison still matched 127/128 and remains unresolved. Perplexity and broad quality were not measured.

See the [six-stage index](bench/optimization-implementation/README.md) and [implementation results](IMPLEMENTATION_RESULTS.md). Clocks remained at stock settings and test servers were stopped afterward.

## Follow-up: DFlash2 port

The feasibility analysis is recorded in [stage 07](bench/optimization-implementation/07-VIABILIDADE-PORTE-DFLASH2.md). The existing verifier was limited to MAXT=5 and the real draft did not own separate embedding/head weights. A hybrid needed a weight/feature bridge, not merely two API calls. The recommendation was to profile and validate a prototype before attempting the full port. No kernels or benchmarks were changed in that analysis.

The hybrid target-MK plus real GGML/CUDA draft was subsequently implemented and run. It shared weights, captured features, and verified up to four proposals. The sm_89 build, 34 host tests, rejection/replay cases, 20-output suite, and MK A/B passed within their recorded limits. It fit in 16 GB and improved on MK MTP4, but remained below llama.cpp DFlash2 4. See [DFLASH_PORT_LOG.md](DFLASH_PORT_LOG.md) and [DFLASH_PORT.md](DFLASH_PORT.md).

## Native draft and later optimization stages

The native CUDA DFlash2 decoder, target T=8 verification, GPU features, DP4A/Q8 math, packed reads, 32K API/checkpoints, and real tests were added later. The draft is captured in CUDA Graphs; it is not fused into one persistent kernel. See [DFLASH_NATIVE.md](DFLASH_NATIVE.md) and [stage 11 results](bench/dflash-port/11-ENTREGA-NATIVA-RESULTADOS.md). Numerical and quality limitations remain documented.

Stages 13–21 profiled the FFN, fused gate/up for six IQ formats, tested 36 format pairs, specialized/inlined paths and two-row IQ layouts, and rejected slower IMMA/pipeline prototypes. Typed Q4_K/Q6_K MMQ with four rows per warp was added, including output-tail handling. An expanded HTTP test exposed an incorrect checkpoint on the second turn; it was fixed and validated again. The prior snapshot and failing stage were preserved.

The selected candidate measured 93.21/62.28/93.36 tok/s for short/long/sampling across six samples, at stock clocks. Gains over the previous snapshot were 15.18%/15.40%/15.88%; against fresh llama.cpp it was +1.49% short and -0.34% long, so no overall win was demonstrated. Three 128-token fixture IDs were preserved; 10/10 known-answer tasks passed and 17/20 generations matched the previous native version. Thirty-six external cases, 13 HTTP cases, 34 CPU tests, and bounded CUDA memchecks passed. See [stage 22](bench/dflash-port/22-RESULTADO-IQ-E-DRAFT-NATIVO.md).

The Unsloth compatibility change added top-level `reasoning_effort` and aliases `high`→`xhigh` and `none`→reasoning off, preserving other request fields and the GGUF default. No Studio installation or saved settings were edited. See [stage 24](bench/dflash-port/24-UNSLOTH-REASONING-EFFORT.md).

The later state-of-the-art review found the result competitive but not a general win: +1.49% on the short llama.cpp comparison, without demonstrated overall superiority. The report recommends prioritizing target FFN work and representative validation; isolated draft fusion has limited upside because of its measured time share. No new benchmark, CUDA change, clock change, or interruption of the active server was performed during that review. See [stage 25](bench/dflash-port/25-ESTADO-DA-ARTE-E-MARGEM.md).

## Q4 target KV cache

Stage 26 added an optional packed symmetric INT4 target cache to prefill, decode, and T=8 verification. Native DFlash2 and model weights are unchanged; Q8 remains the default. At 65,536 cache capacity and chunk 128, Q4 saves 1.0625 GiB compared with Q8. Measured CUDA use was 15.050 GiB with 0.597 GiB free. Short throughput was 93.23 tok/s versus 93.36 for the preserved Q8 run at 38K; sampling was 93.38 versus 93.47. A 1,365-token fixture fell from 62.34 to 57.72 tok/s, with a changed continuation and 27 rather than 25 cycles.

Both formats passed 10/10 small tasks. Q4 completed 20/20 generations, with 16/20 IDs matching Q8; rebuilt Q8 matched the prior IDs 20/20. Thirty-six verifier cases per format, nine attention cases including position 65,535, the packing oracle, 13+13 HTTP checks, a 14,530-token retrieval test, MTP smoke test, 44 CPU tests/12 subtests, and bounded CUDA memchecks passed. These checks do not establish broad perplexity or full-65K quality equivalence. Raw data and preserved pre-change binaries are in `bench/dflash-port/q4-target/`. No clocks were changed. See [stage 26](bench/dflash-port/26-Q4-TARGET-KV.md).
