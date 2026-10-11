# 10 — Packed reads and Nsight profiling

Q4_K and Q6_K readers process eight packed values and matched the CPU oracle across 1,271,808 weights. Q6_K uses aligned `uint16` loads consistent with its 210-byte block layout. Eight-values-per-thread FP16 dequantization matched CPU results. Memcheck of all oracle kernels reported zero errors.

Before adding further event instrumentation, three-run native7 throughput measured 80.47/53.64/80.07 tok/s short/long/sampling. IDs matched the earlier Q8 and F32 builds on all three fixtures. Draft cost fell to about 15–16 ms/cycle; target T=8 remained about 77–79 ms/cycle. Seven proposals beat four in all three measured cases.
