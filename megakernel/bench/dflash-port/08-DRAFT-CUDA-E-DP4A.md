# 08 — Full CUDA draft decoder and DP4A optimization

A missing head-channel term in native KV-cache indexing was fixed. An independent oracle covered norm, RoPE, writes to all 128 channels, historical KV, and masked noise; maximum FP16 error was 5.18e-4. CPU sliding attention passed with maximum error 1.26e-9.

The F32 selector matched 110–111 of 112 candidates against the GGML FP16 reference, with 0.39–0.49% NRMSE and identical proposals on the short fixture. Three-run throughput was 66.22 tok/s short, 44.41 long, and 65.81 sampling. Draft work still cost about 34 ms per cycle, so F32 was not selected as the final path.
