# How the Qwen3.8 megakernel works

This guide introduces the inference concepts used by the code in this repository. It is intended to explain the design, not to replace the implementation and validation notes under `megakernel/`.

## The model execution problem

Autoregressive generation predicts one token at a time. For every token, the model reads its weights, updates its recurrent or attention state, computes a vocabulary-sized set of logits, and selects the next token. A conventional framework expresses this as many tensor operations. Each operation is usually a separate GPU kernel launch, with intermediate tensors passed between them.

For a large model doing batch-one decoding, the GPU often spends more time moving weights and waiting for launches than performing arithmetic. This project specializes the execution path for one known architecture and tensor schema: Qwen3.8-27B. Its weights remain in the GGUF's mixed IQ/K formats, and the implementation dispatches according to each tensor's actual type.

## What “megakernel” means here

A megakernel is a large, persistent CUDA kernel that performs much of one decoding cycle in a single cooperative launch. In this engine, a cycle can process the current token together with speculative candidates, run the target model's projections and state updates, compute attention and the feed-forward network, form logits, verify candidates, and commit only accepted state. CUDA cooperative-grid synchronization separates phases inside that launch.

The benefit is lower launch and intermediate-buffer overhead, and the chance to reuse weights across several token positions in the same cycle. The cost is a rigid shape and resource contract: all participating blocks must support cooperative synchronization; shared memory, registers, occupancy, and grid size constrain the design. That is why this source targets a validated Qwen3.8 shape instead of acting as a general model runtime.

Prompt ingestion uses a separate batched prefill path. It processes prompt chunks with CUDA kernels and cuBLAS, creates the same recurrent and attention state that decoding expects, then hands off the final prompt token to the megakernel. Prefill and generation therefore have different bottlenecks and should be benchmarked separately.

## Qwen3.8 state: DeltaNet and attention

Qwen3.8 is a hybrid architecture. Most layers use Gated DeltaNet, a recurrent sequence model that maintains a compact state and updates it as tokens arrive. Some layers use full attention, which stores keys and values for prior positions in a KV cache. This combination is why the engine must manage both recurrent state and an attention cache.

The target's DeltaNet state is held in FP32 for numerical stability. During speculative execution, the engine cannot irrevocably commit state for every proposed token: later proposals may be rejected. The megakernel uses accepted-token bookkeeping and replay so that recurrent state corresponds to the accepted prefix. Checkpoints support rollback and multi-turn server behavior.

The target KV cache stores attention keys and values. Q8 uses eight-bit values plus scales; the experimental signed Q4 cache stores four-bit values plus scales and uses less memory, but adds quantization error and can change logits. It is distinct from weight quantization: `--kv q4` does not convert the GSQ-RCO target weights. The DFlash2 draft has its own short sliding-window KV cache, kept in FP16 in this implementation.

## Mixed-precision GSQ-RCO weights

The target file used here is `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` from ISTA-DASLab. “IQ3_S” is the GGUF's overall file-type label, not a claim that every tensor uses that type. Its 866 tensors use a mixture of IQ, K-quant, F32, and BF16 formats. This non-uniform precision allocation is designed to retain quality at a low average bit rate by assigning different precision to different tensors.

The loader checks architecture metadata, tensor names, shapes, and supported types before uploading weights. CUDA kernels decode the stored formats directly as needed; the target is not requantized to Q4_K_M. Q4_K_M in the launch examples names the separate DFlash2 draft GGUF, while Q4 KV names the optional attention-cache format.

## Speculative decoding and DFlash2

Speculative decoding uses a smaller, faster draft model to propose several future tokens. The large target model evaluates those proposals together and decides which prefix is valid. Accepted proposals reduce the number of expensive target-model cycles needed to emit a sequence. Rejected proposals are corrected by the target, preserving the target's decoding distribution when verification and sampling are implemented correctly.

DFlash2 is a trained draft model that predicts token blocks using hidden features taken from selected target layers. The DFlash2 GGUF in this project has five draft blocks, while the runtime can ask it for up to seven proposals per verification cycle. Draft depth is a performance tradeoff: a deeper draft can produce more accepted tokens per target pass, but costs more computation and may have lower acceptance on a particular prompt.

This repository includes a native CUDA DFlash2 path. It runs the draft's feature encoder, sliding-window attention, feed-forward layers, selector, and output logic using CUDA kernels captured in CUDA Graphs. The target then verifies the proposed tokens using its own logits and state. The word “native” means the draft compute runs through this CUDA implementation; the draft is **not** fused into the target's persistent megakernel. A llama.cpp/GGML reference path is retained for validation.

Acceptance rate depends on the draft, prompt content, sampling settings, and target numerics. Higher acceptance is useful only if the added draft work is smaller than the target work it saves. Therefore benchmark output reports throughput and acceptance together, and comparisons should use representative prompts and equivalent settings.

## Why the 16 GB Ada port needs memory planning

The RTX 4060 Ti has 16 GB of VRAM and 34 Ada streaming multiprocessors (`sm_89`). The model weights, CUDA libraries/runtime, recurrent state, checkpoints, prefill workspace, attention KV, DFlash2 weights, and draft cache all compete for that budget. Some allocations depend on the configured context even before the prompt fills it.

Q4 target KV reduces the cache allocation so more context can fit, but it does not make the model weights smaller. The server performs a preflight against currently free VRAM; display processes and other GPU applications reduce what is available. A context that allocates successfully is not proof that an equally long prompt was processed or that output quality is unchanged.

## Where to look in the source

| Component | Main source |
|---|---|
| Target loader, persistent decode, and engine state | `megakernel/qwen38_mk.cu`, `megakernel/qwen38_mk.h` |
| Target prefill | `megakernel/prefill.cuh`, `megakernel/prefill_gemm.cuh` |
| GGUF quantized-weight decoding | `megakernel/native_quant.cuh`, `megakernel/native_iq_mma.cuh` |
| Q4/Q8 target KV | `megakernel/kv_quant.cuh`, attention code in `qwen38_mk.cu` |
| Native DFlash2 draft | `megakernel/dflash_native.cu`, `megakernel/dflash_native.h` |
| Target/draft integration and CLI | `megakernel/dflash_bridge.cpp`, `megakernel/dflash_main.cpp` |
| C API and OpenAI-compatible server | `megakernel/capi.cpp`, `megakernel/server/mk_server.py` |
| Ada launcher | `port_ada.py`, `megakernel/serve_ada.py` |

For implementation details and measured limitations, start with [`megakernel/ADA_GSQ.md`](megakernel/ADA_GSQ.md), [`megakernel/DFLASH_NATIVE.md`](megakernel/DFLASH_NATIVE.md), and [`megakernel/IMPLEMENTATION_RESULTS.md`](megakernel/IMPLEMENTATION_RESULTS.md).
