# DFlash2 and prefill: decisions based on measurements

## DFlash2 Q4 through llama.cpp

The official DFlash2 Q4 draft loaded and generated with the real GSQ-RCO target. Three repetitions measured 91.35 tok/s on the short greedy case, 62.29 tok/s long, and 92.56 tok/s sampling. It became the fastest tested path at that stage. The draft is separate Q4_K_M; the target remains GSQ-RCO mixed precision.

## Prefill

Automatic chunking and a cuBLAS path with fewer copies reduced long-prefill latency. The tiled GEMM candidate passed correctness tests but was slower, so it remained experimental and disabled. Decode and prefill require separate benchmarks because their dominant costs differ.
