# 06 — Plan for the complete native DFlash2 port

2026-10-10. Stock clocks. The four-proposal hybrid baseline was preserved under `full-native-baseline/`, including sources, binaries, and SHA-256 hashes.

The next stage was to replace GGML draft execution with native CUDA while retaining target verification in MK. The design needed T=8 verification for seven proposals, GPU-resident feature transfer, correct sliding-window draft attention, state reset/checkpoint behavior, and independent numerical oracles. The existing hybrid remained the rollback point until all checks passed.
