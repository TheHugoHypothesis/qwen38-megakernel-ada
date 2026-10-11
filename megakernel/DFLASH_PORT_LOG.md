# DFlash2 port: implementation log

## Stage 1: MK target with a GGML/CUDA draft

Authorized on 2026-10-10. GPU clocks were to remain at stock settings. The original target and draft GGUFs were used; the llama.cpp checkout remained an unmodified library and reference. The IQ16 baseline and hashes are preserved under `bench/dflash-port/baseline/`.

The first deliverable was an experimental CLI supporting up to four proposals, target-feature capture at layers `[6,20,34,48,62]`, external verification, and a real DFlash2 draft. Existing MAXT and T=5 layouts were retained. Seven proposals and draft-kernel fusion were deferred pending validation.

The IQ2_S embedding and Q4_K LM head were migrated once into standard GGML CUDA buffers and borrowed by the draft. Bytes were copied without requantization; MK pointers were updated and old allocations released. The draft/context must be destroyed before the target that owns those buffers. This avoided a second target model and private CUDA-layout wrappers.

Features were captured into F32 buffers per chunk and delivered through callbacks after prefill and verification. Only the anchor and accepted proposals were injected into draft KV; speculative noise was cleared before each injection. The implementation did not retain five hidden-state streams for the whole context. The bridge uses the private `llama-model.h` from the local llama.cpp checkout; that dependency is explicit in the build notes.

The validation plan covered correct and incorrect proposals, rejection at each position, non-speculative token IDs, chunked capture, shared bytes/pointers, real DFlash2 generation, memory, and throughput against llama.cpp DFlash2 4. No gain was assumed before measurement.

## Executable result and measurements

The `make ARCH=89 all qwen38_dflash external_verify_test` build completed. `qwen38_dflash` runs the MK target with the real GGUF draft and up to four proposals. It does not load a second target or requantize shared weights. The draft graph, F32 chunk features, and shared embedding/head are described in [DFLASH_PORT.md](DFLASH_PORT.md).

1. [Verifier and features](bench/dflash-port/01-VERIFICADOR-E-FEATURES.md): rejection positions 0–3, full acceptance, continuation IDs matching scalar decode, shared bytes, and chunked prefill passed.
2. [Hybrid and memory](bench/dflash-port/02-HIBRIDO-E-MEMORIA.md): 67.05 tok/s short, 55.65 long, 64.41 sampling; sampled NVML peak 14,945 MiB.
3. [Quality and limits](bench/dflash-port/03-QUALIDADE-E-LIMITES.md): 10/10 numeric answers correct; 19/20 texts matched the previous MK version and 15/20 matched non-speculative llama. This is not a broad accuracy evaluation.
4. [MK regression](bench/dflash-port/04-REGRESSAO-DO-MK.md): no observed regression; variation below 0.1% and IDs matched the preserved baseline.
5. [Final checks](bench/dflash-port/05-CHECKS-FINAIS.md): external test repeated on the final build and memcheck of four MK launches reported zero errors.

Fresh llama.cpp DFlash2 4 measured 74.13/59.52 tok/s. The hybrid was about 10.5%/11.2% faster than MK MTP4 in the two cases, but about 9.6%/6.5% slower than llama.cpp DFlash2 4. Different long continuations limit direct comparisons. The chart-generation source is `bench/plot_dflash_port.py`; raw per-run data and build hashes are retained where available in the associated benchmark records.

The hybrid measurements came from the first integrated build. The final build added multiple prompts, excluded text conversion from timed throughput, and made weight migration rollback safe; decode math kernels were unchanged. The quality suite, MK A/B, and complete external test were rerun on the final build.

## Native DFlash2 and target optimization

Subsequent stages implemented a native CUDA draft decoder captured in CUDA Graphs, T=8 target verification, GPU-resident features, DP4A/Q8 math, packed reads, checkpoint/API support, and tests. The draft is not fused into a single persistent kernel. See [DFLASH_NATIVE.md](DFLASH_NATIVE.md), [stage 11](bench/dflash-port/11-ENTREGA-NATIVA-RESULTADOS.md), and the later IQ/MMQ work in [stage 22](bench/dflash-port/22-RESULTADO-IQ-E-DRAFT-NATIVO.md).

Target-side stages later optimized IQ FFN paths and typed draft MMQ. The selected result reached 93.21/62.28/93.36 tok/s short/long/sampling across six samples. It improved on the prior MK snapshot by 15.18%/15.40%/15.88%, but compared with fresh llama.cpp it was +1.49% short and -0.34% long: no general win was demonstrated. Clocks remained at stock settings. Full details, limitations, and next steps are in [stage 22](bench/dflash-port/22-RESULTADO-IQ-E-DRAFT-NATIVO.md).

The reasoning-effort compatibility update for Unsloth is recorded in [stage 24](bench/dflash-port/24-UNSLOTH-REASONING-EFFORT.md). The state-of-the-art review and Amdahl-based follow-up priorities are in [stage 25](bench/dflash-port/25-ESTADO-DA-ARTE-E-MARGEM.md).

## Optional Q4 target KV

[Stage 26](bench/dflash-port/26-Q4-TARGET-KV.md) documents the optional packed INT4 target KV implementation. Q8 remains the default. At 65,536 cache capacity and chunk 128, Q4 saves 1.0625 GiB versus Q8. Quality checks found 16/20 generations with IDs matching Q8; a long fixture also had lower throughput and a changed continuation. The checks do not establish broad quality equivalence. See the stage note for the complete test list, memory readings, and raw artifacts.
