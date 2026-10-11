# Optimization implementation and benchmark notes

This directory records the staged Ada optimization work. The consolidated summary is [IMPLEMENTATION_RESULTS.md](../../IMPLEMENTATION_RESULTS.md). Measurements used stock clocks, the original target GGUF, CUDA 13.4/`sm_89`, and one GPU used serially.

| Stage | Record |
|---|---|
| 1 | [IQ16 kernels and automatic chunk sizing](01-IQ16.md) |
| 2 | [Interleaved IQ16 results](02-IQ16-RESULTADOS.md) |
| 3 | [DFlash2 and prefill decisions](03-DFLASH-E-PREFILL.md) |
| 4 | [Expanded 20-prompt validation](04-VALIDACAO-AMPLIADA.md) |
| 5 | [Final A/B and numerical diagnosis](05-CHECKS-FINAIS.md) |
| 6 | [API, sampling, and 32K allocation test](06-API-SAMPLING-E-32K.md) |
| 7 | [DFlash2 port feasibility](07-VIABILIDADE-PORTE-DFLASH2.md) |

Keep raw data, hashes, and archived binaries with their stage. A reported speedup only applies to the recorded engine, context, prompts, sampling, and repetitions.
