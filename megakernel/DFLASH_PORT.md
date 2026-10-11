# DFlash2 with an MK target: hybrid prototype

This note describes the initial hybrid implementation. For the current native CUDA draft path, seven proposals, C API, and HTTP server, see [DFLASH_NATIVE.md](DFLASH_NATIVE.md). The constraints below apply to the preserved prototype; the current CLI defaults to `--backend native`.

The target GSQ-RCO model runs through the megakernel, while the real DFlash2 draft runs through the existing llama.cpp GGML/CUDA graph. The prototype captures target features at layers `[6,20,34,48,62]`, shares the IQ2_S embedding and Q4_K output head, and verifies external proposals. Draft execution remains separate from the persistent target kernel. This stage supports at most four proposals (`MAXT=5`).

## Build and run

Build against a local llama.cpp checkout compatible with its private `llama-model.h` and `llama-ext.h` headers. The recorded checkout is `aa94f2086147cd4bfbcff818ce998dffce5fcb39`; it was not modified.

```sh
make -C open-jet/megakernel ARCH=89 qwen38_dflash

open-jet/megakernel/qwen38_dflash \
  -m models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF/snapshots/d562806dbafae37109975e970aae91b43e73b440/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --draft-model models-dflash2-qwen38/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --prompt-ids open-jet/megakernel/bench/optimization-analysis/llama-mtp4-short/prompt.ids \
  --backend ggml -c 4096 -d 4 -n 128 --reps 3
```

The CLI accepts pre-tokenized IDs and does not apply a chat template. Supply IDs tokenized with the target GGUF's tokenizer. `--prompt-ids` may be repeated to reuse one model load across prompts. Each prompt resets target and draft state, warms up with 16 tokens, then runs measured repetitions. `--temp 0.7 --seed 123` enables target sampling; the draft remains deterministic. Default prefill chunk is 512 and default target KV is Q8.

`IDS:` prints generated token IDs, `TEXT_HEX:` prints UTF-8 output as hex, and `RESULT` reports timing, acceptance, and prompt index. Model load, prefill, first-token latency, and text conversion are excluded from decode throughput. `verify_with_injection_seconds` already includes `injection_decode_seconds`; do not add them together.

## Code and lifecycle contracts

- `qwen38_mk.cu/.h` captures features, verifies external proposals, and exposes borrowed tensors. Only the anchor and accepted proposals become draft features; the existing target rejection/replay logic remains responsible for target state.
- `dflash_bridge.cpp/.h` loads the draft, validates vocabulary/taps/dimensions, injects feature chunks, and extracts selector top-16 proposals. It clears speculative noise from the draft KV before each new injection.
- `dflash_main.cpp` coordinates prompts, resets, sampling, and metrics.
- `tests/external_verify.cpp` tests a scalar token-ID oracle, rejection positions, post-rejection state, shared bytes, layouts, and chunked prefill.
- `bench/dflash_port_bench.py` and `dflash_port_quality.py` contain benchmark and output-quality harnesses with recorded commands and results.

The target owns the shared weight buffers. Destroy the draft bridge/context before destroying the target. Call `enable_features` once before the first prefill; `reset` retains allocations and callbacks. Reset the draft whenever the target is reset. If a callback throws after a forward pass, reset both models before continuing. Checkpointing, truncation, multi-turn reuse, and server concurrency are not integrated in this prototype.

## Limits and follow-up

Validation covered 4K context with Q8 KV on the RTX 4060 Ti. Larger contexts, other GPUs, draft models, and embedding/head quantizations require separate validation. Captured features are F32 and use 50 MiB per 512-token chunk; the full context is not retained on the CPU. The prototype copies features device-to-host and then to the draft; this cost is measured.

Initial results were below llama.cpp/DFlash2 with four proposals. The production launcher was not changed. Seven proposals require T=8 and review of attention, reductions, and shared-memory layouts. Before further draft fusion, measure T=8 verification and compare per-layer logits/features with llama.cpp, including the known long-run numerical divergence in the MK. Any additional draft fusion should be profile-driven.

See the [implementation log](DFLASH_PORT_LOG.md) and [intermediate notes](bench/dflash-port/01-VERIFICADOR-E-FEATURES.md).
