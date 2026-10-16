# 28 — Swift 1.5 GSQ-RCO target support

2026-10-12. RTX 4060 Ti 16 GB, `sm_89`, stock clocks.

## Verified model identity

The local `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` matches the
[UkisAI release](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF)
at revision `d74895bbe5db4bec1e0024e7cc87d59c02d7631a`:

- SHA-256: `1333c6ea70ef348d4ac6d62732772e8ad6571ac5b3754c14ed54f1a0d904a786`.
- File size: 11,771,546,912 bytes. Vocabulary: 248,320 tokens.
- Architecture: `qwen35`, 64 trunk layers (48 recurrent DeltaNet, 16 full attention).
- No MTP block or MTP weights. No stripping is needed for this standard file.
- All 851 tensor names, shapes and formats match the original GSQ-RCO trunk.
  This is structural compatibility; Swift's post-trained weight values differ.
- Formats: 353 F32, 96 BF16, 144 IQ3_S, 96 IQ4_XS, 78 IQ3_XXS, 39 Q4_K,
  17 IQ2_S, 13 Q2_K, 9 IQ2_XS, 5 IQ2_XXS, 1 IQ1_M.
- Tokenizer family, every token/type, ordered BPE merges, BOS/EOS/PAD IDs,
  add-BOS setting and embedded chat template exactly match the original target
  and the existing Qwen DFlash2 Q4_K_M draft.
- The full host schema test estimates 11,807,737,856 device weight bytes,
  including the existing alpha/beta expansion. Target weights are loaded in
  their supplied formats without requantization.

The release identifies Swift's contribution under **Swift Open License v1.0**;
the base model remains attributed to Qwen. The runtime code's license does not
replace the weights' license. Consult the release LICENSE and notices when
redistributing weights. Weights are not bundled with this source repository.

## Implementation

The CUDA kernels already support this exact trunk schema. Reuse the existing
mixed-IQ readers, cooperative decode, chunked prefill, target Q8/Q4 KV, DeltaNet
state/replay, native five-layer DFlash2 CUDA Graphs and verifier. No CUDA kernel
rewrite, weight conversion, token remapping or draft retraining is required.

- `serve_ada.py`: inspect target MTP metadata before choosing defaults. Standard
  targets use `--draft 0` for ordinary MK. External DFlash2 keeps seven proposals.
  Explicit MTP requests against a standard file fail before CUDA allocation.
  Report target name/vocabulary/MTP count/context; reject capacities beyond GGUF.
  Draft compatibility now includes token types, tokenizer preprocessor, PAD and
  add-BOS metadata. Preserve the original MTP-present default of four proposals.
- `capi.cpp`: `mk_model_mtp_layers()` uses the full host-only GGUF/schema validator
  to expose MTP availability without loading the tokenizer or GPU weights.
- `server/mk_server.py`: automatic MTP detection for direct serving and Open-Jet
  integration when no draft depth is specified. Explicit draft settings are
  preserved. An outdated library gives a rebuild instruction for this feature.
- `main.cu`: the ordinary MK CLI also selects zero drafts for no-MTP targets
  when `-d` is omitted, while preserving four for the original MTP-present file.
- `Makefile`: track schema/parser headers as dependencies of the shared C API.
- `tests/test_variant_launcher.py`: unavailable-MTP rejection and defaults.
- `tests/model_variant_gpu.py`: reproducible scalar/native/llama comparison with
  prompt/output IDs, per-case throughput, draft acceptance and three repetitions
  on performance fixtures. One engine occupies the GPU at a time.

The same changes are installed in `open-jet/megakernel/`. Existing HTTP behavior
uses the actual GGUF chat template for reasoning and tools. Vision remains
unsupported by this text-only MK server. The original and ASCII-condensed
targets remain separate supported configurations. This step does not prune
Swift's vocabulary or imply that the base condensed GGUF contains Swift weights.

## Validation results

Evidence is under [`../swift-validation/`](../swift-validation/).

- Checksum/size against the pinned release, full host tensor schema, exact
  tokenizer/template/tensor-layout comparison and isolated Open-Jet eligibility.
- Four launcher tests passed in both source trees. Four condensed export tests
  remain passing. Python compile and original/condensed launch regressions pass.
  Original MTP-present targets still default to four internal drafts; unavailable
  MTP requests against standard Swift fail with a useful configuration error.
- The new host C API detects one MTP block in the original target and none in
  Swift or the stripped condensed target. A missing file is rejected. The direct
  Open-Jet HTTP server, with no `--draft` argument, selected zero drafts and
  correctly answered `45` to the arithmetic request.
- At 4K/Q8, six functional outputs (four known arithmetic answers, a square
  function, and exact multilingual copying) matched across scalar MK, native
  DFlash2 and non-speculative llama.cpp. `café α → 中文 😀` was copied exactly.
- Seven of nine greedy fixtures matched between native and scalar MK, and
  native output text matched llama.cpp on seven of nine. Both 256-token coding
  fixtures diverged between scalar/native at token 122: their valid docstring
  examples differed (`13/15` versus `17/18`), while their function implementations
  were identical. Short-code llama.cpp also chose a different implementation;
  its output was truncated at 256 tokens. This is not bitwise numerical parity
  or proof of unchanged broad accuracy. All three MK benchmark repetitions were
  internally repeatable in each mode.
- Sixteen real HTTP/CUDA checks passed with native DFlash2, Q4 target KV,
  prefill chunk 128, and **72,000 context capacity**. These include streaming,
  exact requested token budgets, seeded sampling, multi-turn/prefix reuse,
  top-level low reasoning effort, structured tools and a tool-result round-trip.
- The longest prompt contained **35,330 actual tokens**. It retrieved
  `ORION-4060` from the beginning, after **132.94 seconds** of prefill
  (about 266 tokens/s); eight generated tokens took 0.184 seconds. This short
  answer does not provide a stable long-context generation speed benchmark.
  Full-72K population and quality remain untested.
- Runtime CUDA-visible use at 72K/Q4 was **15.066 GiB**, with **0.589 GiB free**.
  End-of-test NVML device use was **15,472/16,380 MiB**, including GPU/background
  overhead. VRAM preflight estimated 14.97 GiB. Q4 KV cost is 18,432 bytes per
  position across 16 target attention layers (1.236 GiB at 72K); DFlash2's FP16
  sliding KV and its 1.129 GiB allocations remain independent of target capacity.

The shared library was rebuilt by linking the existing unchanged CUDA objects
with the new host C API. Greedy comparisons used the prior C API with those same
CUDA objects; the 72K HTTP test and automatic-MTP test used the newly linked
library. Both library hashes are recorded. No CUDA kernel recompile or clock
adjustment was necessary. Prior libraries are preserved in the workspace's
`runtime-backups/swift-before/`. Test servers are stopped after each run.

### Performance fixtures and fresh baseline

4K allocated context, Q8 target KV, temperature zero, seven proposals, 256-token
limit, prefill chunk 128; three repetitions per MK mode after warmup. The long
coding fixture has 240 preceding records, rather than a fully populated 4K
context. Decode timing excludes loading, prefill and the first token. Output
divergences above must be considered when interpreting these comparisons.

| Fixture | Scalar MK median | Native DFlash2 median | Native / scalar | Draft acceptance |
|---|---:|---:|---:|---:|
| Short coding | 18.25 tok/s | 88.35 tok/s | 4.84× | 89.0% |
| Coding after records | 18.11 tok/s | 85.53 tok/s | 4.72× | 89.0% |
| Numbered text | 18.24 tok/s | 38.27 tok/s | 2.10× | 29.7% |

These are workload-specific gains against **ordinary MK**, not proof of an
advantage over llama.cpp/DFlash2. Acceptance differs substantially across tasks;
the draft is reused from base Qwen and has not been trained on Swift features.

The existing llama.cpp/DFlash2 baseline also ran successfully with Swift. Its
same three prompts, Q8 target/draft KV, seven-proposal setting, temperature zero,
256-token limit and three repetitions were measured after warmup, with full GPU
offload and fit disabled. Its prompt-processing implementation and draft-cache
format differ from native MK. Servers ran sequentially; no clocks were changed.

| Fixture | Native MK + DFlash2 median | llama.cpp + DFlash2 median | Native difference |
|---|---:|---:|---:|
| Short coding | 88.35 tok/s | 87.86 tok/s | +0.55% |
| Coding after records | 85.53 tok/s | 81.82 tok/s | +4.54% |
| Numbered text | 38.27 tok/s | 40.33 tok/s | −5.10% |

The small sequential fixture comparison shows competitive performance, **not a
general throughput win**. Different continuations and run-to-run GPU variation
limit interpretation. These figures measure generation throughput, whereas
Swift's upstream thinking-token reduction addresses the amount of generation
required by a task. Neither establishes a 9.18× CUDA throughput improvement.

## Run

From the workspace, after rebuilding the port:

```bash
python3 open-jet/megakernel/serve_ada.py --engine megakernel-dflash2 \
  --model Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf \
  --draft-model models-dflash2-qwen38/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --ctx 72000 --draft 7 --kv q4 --prefill-chunk 128
```

For the standalone repository use `python3 port_ada.py` with explicit target,
draft and `--llama-root` paths. Ordinary decoding uses `--engine megakernel`
and automatically selects zero drafts for this file. `--engine dflash2` runs the
llama.cpp/DFlash2 baseline. Internal MTP requires a separate matching `-mtp`
Swift file, which is not part of this validation.

## Interpretation

The draft was trained for the base Qwen model, not specifically Swift. A matching
tokenizer and feature layout allow execution, but do not predict acceptance or
quality. Target verification decides which proposals may be emitted. Greedy
agreement must be measured, and existing MK floating-point differences from
llama.cpp still apply. Q4 KV remains lossy, separately from the target weights.

The model-card figures about thinking tokens, task scores and task speed belong
to upstream model evaluation. They are not measured CUDA tokens/s gains for this
port, and must not be multiplied into speculative-decoding speedups.
