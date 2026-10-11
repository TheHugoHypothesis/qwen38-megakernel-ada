<div align="center">

# Qwen3.8-27B CUDA Megakernel · Ada Port

**Open-Jet's Qwen3.8 megakernel adapted for RTX 4060 Ti 16 GB, with native CUDA DFlash2, mixed-IQ target weights, and an experimental Q4 KV cache.**

![Ada sm_89](https://img.shields.io/badge/NVIDIA-Ada%20Lovelace%20%7C%20sm__89-76B900?logo=nvidia)
![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B-76B900?logo=nvidia)
![DFlash2](https://img.shields.io/badge/speculative%20decoding-native%20DFlash2-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)

</div>

This repository snapshot contains the megakernel port source, build files, HTTP server, Ada launcher, CUDA validation tests, benchmark utilities, and the Open-Jet integration source files. It targets **Qwen3.8-27B**, including the GSQ-RCO mixed-quant GGUF, on NVIDIA Ada (`sm_89`) and DFlash2 from Qwen3.8-27B-DFlash2-Q4_K_M. It was developed and measured on an **RTX 4060 Ti 16 GB** at stock clocks.

> **This is an experimental, model-specific CUDA inference engine.** Q4 KV is lossy, memory headroom depends on other GPU processes, and the published performance sweep is one sample per point. Run the validation suite on your own machine before relying on numerical or quality behavior.

## What is included

For a plain-language explanation of autoregressive inference, the megakernel, DeltaNet, mixed GSQ-RCO weights, KV caches, and DFlash2, see [Concepts](CONCEPTS.md).

- The original persistent CUDA megakernel and prompt-prefill kernels in [`megakernel/qwen38_mk.cu`](megakernel/qwen38_mk.cu), [`megakernel/prefill.cuh`](megakernel/prefill.cuh), and [`megakernel/prefill_gemm.cuh`](megakernel/prefill_gemm.cuh).
- Native CUDA DFlash2 draft implementation and verifier integration in [`megakernel/dflash_native.cu`](megakernel/dflash_native.cu), [`megakernel/dflash_bridge.cpp`](megakernel/dflash_bridge.cpp), and [`megakernel/dflash_main.cpp`](megakernel/dflash_main.cpp). It runs the five-layer DFlash2 draft natively, then verifies proposals against the target model.
- Mixed IQ / K-quant weight decode, Q4/Q8 KV cache support, GGUF checks, server integration, tests, and benchmark scripts.
- [`port_ada.py`](port_ada.py), a top-level entry point to the included [`serve_ada.py`](megakernel/serve_ada.py) launcher. The conventional launcher name remains available as well. Open-Jet integration source snapshots and their tests are in [`openjet-integration/`](openjet-integration/).

## Hardware and model

| Component | Tested configuration |
|---|---|
| GPU | GeForce RTX 4060 Ti, 16 GB, Ada Lovelace, compute capability 8.9 (`sm_89`), 34 SMs |
| Clocks | Stock/default clocks |
| Target | [`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) · `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`; non-uniform mixed precision, 3.55 BPW, about 11.28 GiB |
| Draft | Separate Qwen3.8-27B DFlash2 `Q4_K_M` GGUF; about 1.05 GiB on disk, 5 draft blocks |
| Target KV | Q8_0 or custom experimental signed Q4; DFlash2's sliding cache remains FP16 |
| Tested context | Q4 target KV has been run with 72,000 context capacity; published speed sweep has populated prompts through 32,768 tokens |

**Quantization distinction:** the target is the ISTA-DASLab **GSQ-RCO mixed-precision model**, not the standard Qwen3.8-27B `Q4_K_M` target. The file comes from the [`models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF` model repository](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF). Its `IQ3_S` file-type label is an overall label; the 866 tensors use multiple formats, including IQ3_S, IQ3_XXS, IQ4_XS, IQ2 variants, Q2_K, Q4_K, Q6_K, F32, and BF16. GSQ-RCO uses non-uniform precision allocation designed to retain more quality at a low average bit rate. The megakernel loads these target tensors in their original formats and does not requantize target weights.

`Q4_K_M` describes the **separate DFlash2 draft model**. `--kv q4` selects a third, independent format: the experimental lossy Q4 **target KV cache**. Neither changes the GSQ-RCO target weights. The target and draft weights are not included. The launcher checks target/draft architecture, dimensions, tokenizer, and draft taps when native DFlash2 is selected.

## Context scaling on the 4060 Ti

The chart compares native DFlash2 with Q8 and Q4 target KV. The prompt is a repeated tokenized fixture; acceptance can differ on real conversations. The CSV has one measured repetition and 64 generated tokens per point. The small throughput differences are not statistically conclusive; Q4's clearest demonstrated effect is lower VRAM use.

![Decode and prefill throughput by populated prompt length](megakernel/bench/context-scaling-showcase/assets/throughput.png)

![Sampled memory footprint and DFlash2 acceptance](megakernel/bench/context-scaling-showcase/assets/memory-and-acceptance.png)

At 32,768 populated tokens, Q4 sampled 14,803 MiB versus Q8's 15,340 MiB, saving 537 MiB on the whole device. Decode measured 85.68 versus 85.21 tokens/s in this single sample. These values include display/background GPU memory and do not establish quality equivalence. See the detailed [benchmark README](megakernel/bench/context-scaling-showcase/README.md), [CSV](megakernel/bench/context-scaling-showcase/results.csv), and implementation notes [`ADA_GSQ.md`](megakernel/ADA_GSQ.md), [`DFLASH_NATIVE.md`](megakernel/DFLASH_NATIVE.md), and [`IMPLEMENTATION_RESULTS.md`](megakernel/IMPLEMENTATION_RESULTS.md).

## Build

Requirements: Linux, NVIDIA CUDA toolkit and driver, an Ada GPU, Python 3, and a local `llama.cpp` checkout/build. llama.cpp is used for GGUF/tokenizer support and shared libraries; the megakernel kernels and native DFlash2 implementation are in this repository. Build llama.cpp with CUDA enabled for Ada, then build the port:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j --target llama-server

make -C megakernel LLAMA=../llama.cpp ARCH=89 all qwen38_dflash
```

`LLAMA` must point to the llama.cpp source/build tree. If it is elsewhere, pass its absolute path, for example `LLAMA=/opt/llama.cpp`. CUDA toolkit/compiler and llama.cpp versions can affect build compatibility; this source snapshot was built against llama.cpp revision [`aa94f2086147cd4bfbcff818ce998dffce5fcb39`](https://github.com/ggml-org/llama.cpp/commit/aa94f2086147cd4bfbcff818ce998dffce5fcb39).

## Run the Ada port

Use the launcher with explicit model paths. `--engine megakernel-dflash2` selects the full megakernel target plus native DFlash2; `--engine megakernel` uses the megakernel's built-in MTP path; `--engine dflash2` and `--engine mtp` launch llama.cpp baselines.

```bash
python3 port_ada.py \
  --engine megakernel-dflash2 \
  --model /path/to/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --draft-model /path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --ctx 40000 --draft 7 --kv q4 --prefill-chunk 128 \
  --llama-root ./llama.cpp \
  --host 127.0.0.1 --port 8080
```

To use the in-tree launcher directly, substitute `python3 megakernel/serve_ada.py`. Add `--check` to validate paths and print the command without starting inference. The HTTP endpoint is `http://127.0.0.1:8080/v1/chat/completions`.

The repository includes a benchmark harness and graph generator under `megakernel/bench/context-scaling-showcase/`. Its commands and CSV schema are documented in that directory's README.

## Correctness and limits

The code includes CUDA unit tests and external-verification tools under [`megakernel/tests/`](megakernel/tests/). The implementation notes describe validation against llama.cpp and known numerical limits. Q4 KV quantization is intentionally lossy; it may change logits and generated text. Q8 remains the more conservative cache option. DFlash2 changes throughput according to acceptance rate and prompt distribution. This implementation is text-only and specialized to the validated Qwen3.8-27B tensor shapes; it is not a general GGUF runtime.

## Provenance and license


## What this work adds beyond Open-Jet

Open-Jet provides the original Qwen3.8 CUDA megakernel and the foundation for this port. The following are the Ada adaptation and additional work in this repository snapshot; they should not be attributed to the upstream Open-Jet project unless separately merged there:

- **RTX 4060 Ti / Ada support:** `sm_89` build and runtime path, resource checks for the 34-SM GPU, and memory-aware prefill chunk selection for a 16 GB card.
- **GSQ-RCO mixed-quant target support:** GGUF metadata, tensor-name, shape, and per-tensor-type validation, plus CUDA weight readers/dispatch for the model's actual mixture of IQ, K-quant, F32, and BF16 tensors. The target weights are used in their supplied formats rather than converted to a uniform Q4 quantization.
- **Native CUDA DFlash2 draft path:** a five-layer draft decoder with CUDA kernels captured in CUDA Graphs, GPU-side feature handling, selector/proposal generation, and integration with the megakernel target verifier. This is a native CUDA draft implementation, but the draft is **not fused into the target's persistent megakernel**. The llama.cpp/GGML implementation remains available as a reference path in the CLI.
- **Speculative verification and state handling:** T=8 verification for up to seven proposals, rejection/replay coverage, shared target embedding/output storage, and server-side checkpoint/reset/rewind/truncation support.
- **Ada-oriented kernel work:** packed mixed-IQ readers, DP4A paths, paired-row FFN processing, typed Q4_K/Q6_K draft MMQ, and the associated CPU/GPU oracles. Slower IMMA and pipeline candidates are retained as experiments or documented as rejected rather than presented as wins.
- **Memory and serving adaptations:** optional packed symmetric Q4 target KV to reduce cache capacity cost, Q8 as the default, VRAM preflight/runtime reporting, an Ada launcher, and Open-Jet/HTTP/C API integration. Q4 KV is lossy and is independent of target weight quantization.
- **Validation and performance records:** RTX 4060 Ti benchmarks at stock clocks, context-scaling tools, output and verifier checks, API compatibility changes for Unsloth sampling/reasoning fields, and implementation journals. The recorded measurements show a small short-prompt advantage over the local llama.cpp/DFlash2 baseline but no demonstrated general throughput win; see the limitations and exact conditions in the benchmark notes.

The main entry points for reviewing these additions are [the GSQ-RCO/Ada implementation note](megakernel/ADA_GSQ.md), [the native DFlash2 design and limits](megakernel/DFLASH_NATIVE.md), [the implementation history](megakernel/IMPLEMENTATION_LOG.md), and [the benchmark summary](megakernel/IMPLEMENTATION_RESULTS.md). The concepts guide is [CONCEPTS.md](CONCEPTS.md).

## How to reference this work

When citing the port or using its benchmark results, cite the GitHub repository containing this snapshot and identify the revision you used. Replace `<owner>` below with the GitHub account or organization that publishes this directory; do not cite the upstream Open-Jet project as the author of the Ada/DFlash2 additions.

**Suggested prose citation:**

> Qwen3.8-27B CUDA Megakernel: Ada Port. (2026). RTX 4060 Ti adaptation of the Open-Jet Qwen3.8 megakernel with native CUDA DFlash2 and GSQ-RCO mixed-precision target support. GitHub repository, `<repository URL>` (commit `<commit SHA>`).

**BibTeX template:**

```bibtex
@software{qwen38_megakernel_ada,
  title        = {Qwen3.8-27B CUDA Megakernel: Ada Port},
  author       = {{<owner or contributor name>}},
  year         = {2026},
  url          = {<repository URL>},
  version      = {<tag or commit SHA>},
  note         = {RTX 4060 Ti sm_89 port; target: ISTA-DASLab Qwen3.8-27B GSQ-RCO; native CUDA DFlash2 draft}
}
```

For reproducible performance claims, also report the GPU (RTX 4060 Ti 16 GB), stock clock settings, CUDA and driver versions, target and draft GGUF revisions, context and populated prompt length, KV format, proposal depth, sampling configuration, and benchmark commit. The target weights are from [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF); the separate draft is from [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF). Cite these model repositories separately when discussing the weights or draft model.

Please also retain the upstream references and licenses: [L-Forster/open-jet at the source revision](https://github.com/L-Forster/open-jet/commit/06223125d8320ca7c41d28e62cbc92dc0eefb197), [llama.cpp at the comparison revision](https://github.com/ggml-org/llama.cpp/commit/aa94f2086147cd4bfbcff818ce998dffce5fcb39), and the applicable notices listed above. This attribution distinguishes the upstream megakernel and llama.cpp reference code from the adaptations and measurements in this port.
