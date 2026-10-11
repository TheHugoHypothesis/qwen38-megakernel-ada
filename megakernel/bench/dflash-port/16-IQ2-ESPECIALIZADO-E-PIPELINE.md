# 16 — Specialized IQ2 and IMMA pipeline investigation

2026-10-10, stock clocks. This stage added a specialized IQ2 candidate and investigated IMMA/pipelined alternatives. Candidate binaries and evidence were preserved.

The implementation focused on reducing quantized-weight decoding overhead while maintaining the original GGUF formats. Pipeline/IMMA ideas were measured rather than assumed beneficial; only variants that passed independent dequantization/dot-product oracles and complete generation tests could be selected.
