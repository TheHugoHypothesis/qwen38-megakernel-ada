# 12 — Remaining work to beat llama.cpp DFlash2

2026-10-10. This is a code/profile review with two additional RTX 4060 Ti measurements. Clocks remained at stock settings. No implementation change was made in this analysis.

The measured gap was driven mainly by target T=8 verification, especially target FFN and quantized matrix-vector work; native draft execution had already been reduced substantially. Amdahl's law limits the benefit of further draft-only fusion when draft computation is a small fraction of total cycle time. The report prioritizes profiling each target phase, reducing IQ decode/FFN cost, improving target tiling and memory reuse, and validating complete request throughput against a fresh llama.cpp DFlash2 baseline.

Any candidate should use identical prompt IDs, context, output length, sampling settings, and clocks; interleaved repetitions; and representative short/long prompts. Track acceptance, tokens per cycle, latency, and memory together. The short-case edge later measured at +1.49% is too small to call a general win; long throughput was -0.34% against llama.cpp in that comparison. See [stage 22](22-RESULTADO-IQ-E-DRAFT-NATIVO.md).

The report's Amdahl scenarios are projections, not benchmarks. Preserve snapshots and reject optimizations that do not improve end-to-end throughput beyond measurement variance or that introduce material quality changes.
