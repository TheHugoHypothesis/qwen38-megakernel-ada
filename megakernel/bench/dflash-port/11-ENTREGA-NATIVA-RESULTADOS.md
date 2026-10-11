# 11 — Native DFlash2 delivery and results

2026-10-10. RTX 4060 Ti 16 GB, `sm_89`, stock clocks. Earlier baselines and stages were preserved.

The delivered path runs the target in the cooperative MK and the complete DFlash2 draft in custom CUDA kernels captured by CUDA Graphs. It supports seven proposals, shared target embedding/head storage, GPU feature staging, FP16 sliding-window draft KV, DP4A/Q8 math, HTTP/C API integration, and checkpoint/reset behavior. The draft is native CUDA but is not fused into a single persistent kernel.

The report records throughput, acceptance, memory, output checks, and oracle coverage. Results are tied to exact binaries and fixtures; later target-side IQ/MMQ improvements are recorded in [stage 22](22-RESULTADO-IQ-E-DRAFT-NATIVO.md). No broad quality guarantee or overall victory over llama.cpp was established.
