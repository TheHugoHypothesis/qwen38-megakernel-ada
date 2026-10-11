# 26 — Optional Q4 target KV cache

This journal documents packed symmetric INT4 KV for the target. It is separate from target weight quantization and from the DFlash2 draft cache. Q8 remains the default.

The implementation covers target prefill, decode, and T=8 verification. At 65,536 capacity with prefill chunk 128, Q4 saves 1.0625 GiB compared with Q8. Measured CUDA use was 15.050 GiB with 0.597 GiB free. Short throughput was 93.23 tok/s versus 93.36 for the preserved Q8 run at 38K; sampling was 93.38 versus 93.47. A 1,365-token fixture fell from 62.34 to 57.72 tok/s, with a changed continuation and 27 rather than 25 cycles.

Both formats passed 10/10 small tasks. Q4 completed 20/20 generations and matched Q8 token IDs on 16/20; rebuilt Q8 matched prior IDs on 20/20. Validation also covered 36 verifier cases per format, nine attention cases including position 65,535, a packing oracle, 13+13 HTTP checks, a 14,530-token retrieval prompt, MTP smoke, 44 CPU tests/12 subtests, and bounded CUDA memchecks.

These checks do not provide a broad perplexity or full-65K quality guarantee. Raw/aggregate data and preserved pre-change binaries are under `q4-target/`. No clocks were changed. Q4 is a memory-capacity option with a measurable quality/performance trade-off, not a default speed optimization.
