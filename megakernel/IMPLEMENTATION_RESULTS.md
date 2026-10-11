# Implementation results: RTX 4060 Ti 16 GB

Measured on 2026-10-10 with CUDA 13.4, `sm_89`, and stock GPU clocks. The megakernel improved by about **25% in decode** and **29.5% in long-prompt prefill throughput**. The fastest measured path was DFlash2 Q4 through llama.cpp: **91.35 tok/s greedy short-prompt** and **92.56 tok/s sampling**. The target remained the original ISTA-DASLab GSQ-RCO mixed-precision GGUF; its weights were not modified or requantized, and the llama.cpp checkout was not changed.

Measured results and validation are summarized below; raw benchmark records are retained in the stage notes where available.

## Running the tested configurations

From the workspace root, with Python, NumPy, and Jinja2 available:

```sh
# Fastest measured configuration: DFlash2 Q4, draft depth 7, 4K/Q8.
python open-jet/megakernel/serve_ada.py

# Other built and tested paths.
python open-jet/megakernel/serve_ada.py --engine megakernel
python open-jet/megakernel/serve_ada.py --engine mtp

# Check files and print the launch command without loading the GPU.
python open-jet/megakernel/serve_ada.py --check

# DFlash2 also passed an allocation/generation check with 32K context.
python open-jet/megakernel/serve_ada.py --ctx 32768
```

The launcher binds to `127.0.0.1:8080`; chat completions use `/v1/chat/completions`. Override `--port`, `--model`, `--draft-model`, `--ctx`, and `--draft` as needed. `--fit off`, full offload, and one sequence make the configuration explicit; context is not silently reduced to fit. Run one server at a time on this GPU. The GGUF chat template and reasoning mode are retained; the reported benchmarks explicitly disabled thinking.

The default target is `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` from ISTA-DASLab. Its `IQ3_S` file label does not mean all tensors use IQ3_S: the 866 tensors use mixed IQ, K-quant, F32, and BF16 types. The separate DFlash2 draft is Q4_K_M. `--kv q4` is a third, optional, lossy target-cache format; it does not change target weights. See [CONCEPTS.md](../CONCEPTS.md) and [ADA_GSQ.md](ADA_GSQ.md).

## Performance measurements

Interleaved A/B measurements used identical token IDs, 128 output tokens, 4K context, Q8 KV, and stock clocks. The IQ16 path improved MK throughput by 25.54% on short prompts, 24.81% on long prompts, and 25.05% in sampling. Long-prefill latency decreased by 22.78%. A tiled GEMM passed correctness checks but was slower and remains disabled.

The locally available llama.cpp DFlash2 path measured 91.35/62.29 tok/s greedy short/long and 92.56 tok/s sampling. The megakernel's later native DFlash2 path improved substantially over earlier MK snapshots, but did **not** establish an overall win over llama.cpp: the selected candidate was +1.49% on the short case and -0.34% on the long case in one controlled comparison. See the later [native DFlash2 and IQ/MMQ results](bench/dflash-port/22-RESULTADO-IQ-E-DRAFT-NATIVO.md).

## Validation and limits

The IQ oracle covered 8,945,664 real values. GEMM results were checked against CPU/cuBLAS, 34 host tests passed, and four-cycle CUDA memcheck reported no errors. All three engines solved 10/10 numeric tasks. In the 20-prompt output comparison, DFlash2 matched the non-speculative baseline on 17/20 full generations and MK on 16/20. Long teacher forcing for MK still matched 127/128 tokens, first diverging at index 41. This is unresolved; a small task set and token-match counts do not establish broad quality or perplexity equivalence.

DFlash2 passed a 32K allocated-context generation check. The optional Q4 target KV cache later saved memory at 65,536 capacity but changed some generated IDs and reduced throughput on a long fixture. Q8 remains the conservative default. See [Q4 target KV results](bench/dflash-port/26-Q4-TARGET-KV.md).

Intermediate evidence is indexed in [optimization implementation notes](bench/optimization-implementation/README.md) and [DFlash2 port notes](DFLASH_PORT_LOG.md). Clocks stayed at stock settings; test servers were shut down after validation.
