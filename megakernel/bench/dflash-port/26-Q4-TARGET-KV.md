# Q4 target KV implementation — 2026-10-11

Requested: retain native DFlash2 while increasing target context on the RTX
4060 Ti 16 GB. Stock clocks; preserve Q8 default and all GGUF weight bytes.

## Implementation in progress

- `kv_quant.cuh`: packed symmetric signed INT4, low nibble first, FP16 scale
  per 32 values; separate payload/scales. This is a custom cache format,
  not the GGML Q4_0 weight/block layout.
- `qwen38_mk.cu`: explicit mode 2, packed cache allocation, exact cache bytes
  in conservative preflight, async staging/dequantization into the existing
  FP16 tensor-core attention tiles. Decode and T=8 verification use it.
- `prefill.cuh`: Q4 cache writes after normalization/RoPE and packed reads in
  causal flash attention. Reuse existing FP32 accumulations and softmax.
- Server, launcher and both CLIs expose `--kv q4`. Auto stays FP16/Q8.
- Native DFlash sliding cache and draft quantization remain unchanged.

The 17 cache slots currently include the unused MTP slot when DFlash is
selected. K+V use 36,992 bytes/position in Q8 and 19,584 in Q4, including
scales: 47.06% cache saving, 1.0625 GiB at 65,536 positions. This is not a
47% reduction in total model memory. The conservative external draft and
CUDA/workspace reserves are retained.

Q4 changes target logits. A verifier with Q4 verifies against the Q4 target;
it does not restore Q8 accuracy. No quality or throughput claim until tested.

Pre-change binaries and CUDA sources are preserved in `q4-target/baseline/`.
Planned validation: independent packing/dequantization oracle; attention
oracle; external verifier rejection/replay; HTTP state/repeat/tail checks;
Q8 regression; paired Q4/Q8 quality and throughput; 65,536-capacity startup
and actual extended prefill. Results and remaining limits will be added.

## Intermediate checks

- Production packing/dequantization: 9,472 values each in FP16/Q8/Q4;
  independent CPU bytes, scales and FP16 reconstruction match exactly. Zero
  blocks, signed extrema, small values and guarded rows exercised.
- Compute Sanitizer memcheck on packing: **0 errors**.
- Existing host suite in `.venv-validation`: **44 tests + 12 subtests passed**.
- Launcher `--check` emits target `--kv q4`, depth 7, context 65,536, chunk128.
- Preserved Q8 native binary, fresh 20-generation task suite: **10/10 tasks**.
  The GPU runs outside the sandbox, with no clock changes.
- CUDA 13.4 compiler is optimizing the additional attention specializations.
  Numerical attention and integration results are still pending.

## Integrated results so far

The CUDA 13.4 `sm_89` production build completed. Native DFlash with Q4
target KV at 65,536 positions/chunk128 starts and generates successfully:
preflight **15.27 GiB**, cache **1.28 GB / 1.1953 GiB**, measured device use
**15.050 GiB**, free **0.597 GiB** after decode. A 32-token smoke decode is
not a throughput benchmark and does not fill the context.

Both Q4 and Q8 passed all 36 external rejection/replay cases, continued
scalar-ID checks, invalid-proposal checks, shared-weight bytes and long
chunked-prefill features. Fresh task suite: **10/10 scored tasks** in both
formats. Rebuilt Q8 matched preserved Q8 IDs on **20/20** generations; Q4
matched on **16/20**. This confirms output changes from Q4 and is not a
broad quality/perplexity guarantee. HTTP, extended retrieval, attention
oracle and paired throughput checks continue.

A preserved Q8 40K benchmark attempt was rejected at preflight (15.45 GiB
budget vs 15.45 GiB free, rounded display). Free memory was fractionally
below the exact budget. It is retained in `q4-target/bench-baseline-q8-40k/`.
Do not remove the reserve or bypass the guard. The paired Q8 benchmark will
use 38K to fit reliably in this session.

## Attention oracle and HTTP checks

Nine synthetic attention cases passed the independent double-precision
CPU softmax/matmul oracle using actual O3 production prefill/split-K CUDA
source: FP16/Q8/Q4 prefill, Q4 scalar/T=8 decode, partial query/KV tiles,
4093-offset queries and the final position **65,535**, with slot16 and
65,536-position head strides. Maximum absolute error **0.000213314**
relative to attention on the reconstructed FP16 cache. This measures
kernel correctness, not Q4-vs-Q8 model quality. The synthetic input uses
identity RoPE; the model's unchanged real RoPE is exercised by integration.

Thirteen HTTP checks at 65K capacity passed, including greedy/seed repeats,
streaming, six output tails, sampler validation, a second turn and actual
**2,791-token prefill**. Filtered O3 production T=8 verification memcheck
(two launches): **0 errors**. An extended actual-model retrieval prompt and
paired throughput are being measured next.

## Extended actual-model context

A second HTTP run with explicit chunk128 passed all 13 checks and a
**14,530-token prompt**, retrieving `ORION-4060` from the beginning after
1,200 records. Prefill **52.35 s**; answer **8 tokens**; nvidia-smi memory
**15,456 MiB / 16,380 MiB** after requests. Raw request/response and server
log: `q4-target/http-q4-extended/`. This is one retrieval fixture, not a
fully populated 65K context evaluation. Reserving a larger cache does not
make attention scan its unused positions.

Fresh Q8 task-suite decode timings for >=64-token cases were about
0.994–0.997 times the preserved binary's throughput. No substantial Q8
regression appeared there; paired three-repetition benchmarks continue.

## Final paired throughput

Three repetitions, 128 output tokens, warmup, identical short/long prompt
IDs and sampling seed, chunk128. Decode excludes load/prefill/first token.
The short fixture has 24 prompt tokens; the long fixture has 1,365.

| Configuration | Context capacity | Short tok/s | 1,365-token fixture tok/s | Sampling tok/s |
|---|---:|---:|---:|---:|
| Preserved Q8 native | 38,000 | 93.36 | 62.34 | 93.47 |
| Rebuilt Q8 native | 38,000 | 93.04 | 62.17 | 93.18 |
| Q4 native | 65,536 | 93.23 | 57.72 | 93.38 |

Rebuilt Q8 retains the same IDs across all fixtures, with 0.27–0.34% lower
throughput here. Q4 short/sampling IDs also match, and speeds differ by
less than 0.15% from the preserved Q8. All repetitions are deterministic.

Q4's 1,365-token continuation diverges at output index41: acceptance falls
from **102/168 (60.7%)** to **100/181 (55.2%)**, needing **27 cycles vs25**
for the same output count. Verification plus injection per cycle stays
about **68 ms**; draft per cycle about **13.4 ms**. The measured **7.41%**
throughput drop coincides with more verification cycles and a changed
continuation. It is not an isolated equal-output kernel speed comparison.
Q4 can therefore reduce speculative throughput on some workloads.

Raw timings/IDs/NVML samples, commands and binary hashes: directories
`q4-target/baseline-q8-38k`, `new-q8-38k`, `new-q4-65k`; aggregate
`q4-target/summary.json`. Preserved 38K Q8 steady CUDA use was **15.113 GiB**,
rebuilt Q8 **15.152 GiB**, Q4 65K **15.050 GiB** in the short runs. These
measurements include the device's other users and library overhead.

## Code and reproducibility

Modified production paths: `qwen38_mk.cu::{Impl,attn_proj,attn_core_t,attn_core,
pf_init,pf_run}`, `prefill.cuh::{pk_attn_prep,pk_attn}`, the new
`kv_quant.cuh` helpers, `qwen38_mk.h`, `main.cu`, `dflash_main.cpp`,
`server/mk_server.py::{Server,main}`, and `serve_ada.py::main`. API modes
remain 0=FP16, 1=Q8, -1=auto; mode2 adds Q4. Native DFlash and the C ABI
reuse the existing integer parameter. Makefile dependencies, build-cache
source tracking, benchmark controls and validation tests were updated.

From the workspace root:

```sh
make -C open-jet/megakernel ARCH=89 all qwen38_dflash
python3 open-jet/megakernel/serve_ada.py --engine megakernel-dflash2 --kv q4 --ctx 65536 --prefill-chunk 128
```

Context covers prompt + generated output (HTTP reserves16 tokens). Try
`--ctx 60000` if another application consumes the margin. Keep `--kv q8`
for the original cache precision; it is still the launcher default.
`auto` never chooses Q4. `--kv` other than Q8 requires a megakernel engine
in this launcher. The llama.cpp baseline stays unchanged.

Within `open-jet/megakernel`, numerical tests are `make ARCH=89
kv_quant_test kv_attention_test external_verify_test` followed by
`./kv_quant_test` and `./kv_attention_test`. The standalone attention test
uses ptxas O1 to build its large included translation unit; production
uses O3. Both an O3 attention executable and the final O1 test passed the
same nine independent CPU-oracle cases. `bench/q4_target_checks.py`
accepts `--model TARGET --draft-model DRAFT --output DIRECTORY` and records
serial verifier, quality, extended HTTP and bounded memcheck commands.
Benchmarks use `bench/dflash_port_bench.py --kv q4 --ctx 65536
--prefill-chunk 128` with model, draft and output arguments.

## Final limits

Q4 is an explicit experimental precision/memory trade-off. Model weights
retain their original GGUF quantizations; target KV logits can change.
The small task suite scored10/10 for each format, but Q4 matched16/20
generation IDs. No broad perplexity or fully populated65K model quality
evaluation was performed. Synthetic attention at position65,535 validates
addressing/math; actual-model retrieval tested14,530 prompt tokens.
Existing MK differences from llama.cpp are not resolved by this cache mode.

The memory guard/reserves remain enabled; no clock changes, model-file
rewrites, llama.cpp edits or persistent test servers. The unused MTP cache
slot is still allocated for layout compatibility, though DFlash loads no
MTP weights (`n_draft 0` in target initialization). There is no CPU offload
or artificial truncation/sliding window in the target cache.

## Final validation status

Completed: Ada production build; 44 CPU tests/12 subtests; packing oracle
(3 formats, 9,472 values each); nine O3 attention-oracle cases and the same
nine cases on the final test build; 36 rejection/replay cases per Q4/Q8;
20-generation paired task suite; 13 standard +13 extended HTTP checks;
regular Q4/MTP smoke; paired throughput (3 repetitions per fixture/config).
Memcheck: packing, two production wide verification launches, two synthetic
Q4 scalar/T=8 decode kernels and two Q4 prefill kernels each reported
**0 errors**. Logs, commands and hashes are retained in `q4-target/`.

Success: Q4 target plus native DFlash now runs at65,536 capacity with
conservative preflight, verified attention at the final position, useful
VRAM margin, and essentially unchanged short-fixture throughput. Q8 stays
the default. The long-fixture acceptance/throughput decline and limited
quality coverage remain explicit trade-offs.
