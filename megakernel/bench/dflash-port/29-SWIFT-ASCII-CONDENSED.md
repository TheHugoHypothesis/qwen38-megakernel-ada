# 29 — Swift 1.5 GSQ-RCO ASCII-condensed export

2026-10-12. RTX 4060 Ti 16 GB, `sm_89`; stock clocks.

## Scope and provenance

Apply the same verified 130,020-token subset used by the earlier P1G derivative
to the **Swift 1.5 IQ3_S weight file**. The reference contributes only its retained
tokenizer. Every trunk weight comes from Swift; the original Qwen/Pliny trunk is
not copied into this export. Swift's standard file already lacks MTP weights.

Inputs pinned by SHA-256:

| Input | SHA-256 |
|---|---|
| Standard Swift 1.5 GSQ-RCO IQ3_S | `1333c6ea70ef348d4ac6d62732772e8ad6571ac5b3754c14ed54f1a0d904a786` |
| Earlier prepared condensed tokenizer reference | `6a1cce42ad0abc0b4ca00610ba761a746299af18ab5b4effc7df822c9b755a99` |
| Earlier prepared condensed DFlash2 draft | `3b611e28dd72af6ced025ed2410927557304245ed244f6ddc14151ab061ccd96` |

Swift weights and post-training are credited to
[UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF).
The retained vocabulary is taken from the previously verified
[PlinyTheMedium P1G derivative](https://huggingface.co/PlinyTheMedium/Qwen3.8-27B-GSQ-RCO-ASCII-Condensed),
which credits the bsaleh03 pruning tools at commit
`376a426d5c6b30f28e03c6efaa9edcf85048773f`. This step reuses that exact subset;
it does not independently reimplement or claim to invent the P1G policy.
Swift Open License v1.0 and the source's other attribution metadata are preserved.
Weights are generated locally and are not included in the source repository.

## Implementation

`tools/condense_target.py` adds a reproducible export for a standard no-MTP target:

1. Verify input hashes, source architecture, stable token subset, retained token
   types, non-normal/special tokens, special ID remapping and filtered BPE order.
2. Require unchanged source chat-template and other tokenizer settings, and a
   pre-adapted DFlash2 draft with matching tokenizer and vocabulary-indexed
   selector dimensions.
3. Gather rows only in `token_embd.weight` and `output.weight` using their
   existing quantized bytes. Copy every other Swift tensor unchanged.
4. Verify all exported tensor payloads before exposing the target file, then
   exhaustively compare every retained vocabulary row and every non-vocabulary
   tensor against the unpruned Swift source. Preserve all other source metadata.
5. Validate both source and output with the existing host-only full MK schema
   validator when `--schema-validator` is supplied.
6. Copy the existing compatible condensed draft, verify its checksum, optionally
   copy the source weight license, and write the token mapping, `preparation.json`,
   and `SHA256SUMS`.

`tools/prepare_condensed.py` now permits explicit vocabulary-weight row gathering
in its existing exporter and supports mapping from a source without a draft mask
field. Existing selector export behavior remains unchanged. Seven CPU tests pass,
including mixed IQ2_S/Q4_K preservation and rejection of non-vocabulary gathering.

The CUDA kernels already support the retained head size and scoring tail. No
CUDA kernels, model quantization, or GPU clock settings are changed. The draft
is the earlier base-Qwen condensed DFlash2, not a newly trained Swift draft.

Output creation refuses an existing directory. The target is published after
the writer's payload checks; failure in a later draft/metadata check can leave
that verified target without a complete manifest. Such a directory must be
inspected before rerunning to a different directory.

## Export and validation

Output directory in the workspace: `models-swift-qwen38/ascii-condensed/`.

| Artifact | Size in bytes | SHA-256 |
|---|---:|---|
| `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-ASCII-Condensed-noMTP.gguf` | 11,230,506,048 | `288a9a35ac8de49f3dc506dbd820c6213129248d519a3b51c4402c412f16f7a3` |
| `Qwen3.8-27B-DFlash2-ASCII-Condensed-Q4_K_M.gguf` | 1,102,611,520 | `3b611e28dd72af6ced025ed2410927557304245ed244f6ddc14151ab061ccd96` |

Target size is about **10.46 GiB**, compared with 10.96 GiB for the standard
Swift file. The file saves 541,040,864 bytes, about 515.98 MiB, including reduced
tokenizer metadata. Vocabulary weight storage alone saves **534,716,000 bytes
(509.95 MiB)** on disk and GPU. The target retains 851 tensors and zero MTP
layers. The 849 non-vocabulary tensors and all 130,020 retained rows in each
vocabulary tensor were exhaustively verified, not sampled. Types, Swift license
and other source metadata are preserved; the host full-schema validator passed
before and after export. Draft bytes match the previous prepared draft exactly.

Completed 4K/Q8 tests in [`../swift-condensed-validation/`](../swift-condensed-validation/):

- Four known arithmetic tasks passed in ordinary MK and native DFlash2; those
  outputs also match non-speculative llama.cpp.
- The six functional outputs match across ordinary MK, native DFlash2 and
  llama.cpp on this same derivative. This includes a shared Unicode-copy failure:
  `café α → 中文 😀` becomes `cafÿ α → 中文 😀` in all three modes.
- Seven of nine native/scalar greedy fixtures match exactly; two coding fixtures
  diverge at token 122, just as with full-vocabulary Swift. No exact numerical
  parity or broad accuracy guarantee is claimed.
- For each of the three engine modes, **eight of nine output texts match its
  full-vocabulary Swift result**. The changed fixture is Unicode copying.
  These small fixtures do not establish multilingual or coding benchmark quality.
- One measurement per performance fixture, 256-token generation limit:
  native DFlash2 measured 92.69 tok/s short coding, 89.62 coding after records,
  and 40.17 numbered text; ordinary MK measured 18.69/18.53/18.67 tok/s.
  These are single samples, not a statistically demonstrated throughput gain.
  They use the same fixture text as stage 28, whose standard-file native medians
  were 88.35/85.53/38.27 tok/s over three samples. Loading, prefill and the first
  token are excluded from MK decode timing. Draft acceptance is 89.0% on the
  coding fixtures and 29.7% on numbered text.

The native HTTP/Q4 suite passed **all 16 checks at 96,000 allocated capacity**,
including requested token limits, seeded sampling, streaming, prefix/multi-turn
reuse, top-level low reasoning effort and a structured tool-result round-trip.
The longest populated prompt was **35,330 tokens**, and it retrieved
`ORION-4060` from the beginning after **132.93 seconds** of prefill. The eight-token
answer took 0.177 seconds, which is too short to establish stable long-context
throughput. Full-96K population and quality remain untested.

Runtime CUDA-visible use was **14.945 GiB**, with **0.711 GiB free**. Preflight
estimated 14.89 GiB; end-of-test NVML device use was **15,348/16,380 MiB**.
Target engine allocations were 12.763 GiB; the native draft logged 1.094 GiB.
The earlier standard Swift test used 15.066 GiB at 72K/Q4, so this derivative
served 24K more allocated capacity with slightly more free memory in these runs.
Those runs differ in time and allocator state; exact weight-byte savings are
established by the exhaustive export checks, rather than subtracting NVML values.

No quality equivalence to the unpruned Swift model is assumed. Test servers were
stopped after validation, and GPU clock settings were not changed.

## Reproduce

From the workspace, with the original Swift file, the earlier prepared P1G
reference/draft, and a built `model_schema_test`:

```bash
python3 qwen38-megakernel-ada/megakernel/tools/condense_target.py \
  --source-target Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf \
  --tokenizer-reference models-ascii-condensed-qwen38/prepared/Qwen3.8-27B-GSQ-RCO-ASCII-Condensed-noMTP.gguf \
  --draft models-ascii-condensed-qwen38/prepared/Qwen3.8-27B-DFlash2-ASCII-Condensed-Q4_K_M.gguf \
  --output-dir models-swift-qwen38/ascii-condensed \
  --llama-root llama.cpp \
  --schema-validator qwen38-megakernel-ada/megakernel/model_schema_test \
  --source-license models-swift-qwen38/release/LICENSE \
  --expected-source-sha256 1333c6ea70ef348d4ac6d62732772e8ad6571ac5b3754c14ed54f1a0d904a786 \
  --expected-reference-sha256 6a1cce42ad0abc0b4ca00610ba761a746299af18ab5b4effc7df822c9b755a99 \
  --expected-draft-sha256 3b611e28dd72af6ced025ed2410927557304245ed244f6ddc14151ab061ccd96
```

The output directory must not exist; the already-created verified bundle can be
used directly. The source license flag is optional and copies the original
license supplied by the user, without changing the GGUF's license metadata.

```bash
python3 open-jet/megakernel/serve_ada.py --engine megakernel-dflash2 \
  --model models-swift-qwen38/ascii-condensed/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-ASCII-Condensed-noMTP.gguf \
  --draft-model models-swift-qwen38/ascii-condensed/Qwen3.8-27B-DFlash2-ASCII-Condensed-Q4_K_M.gguf \
  --ctx 96000 --draft 7 --kv q4 --prefill-chunk 128
```

The standalone repository uses `port_ada.py` and explicit model/draft/llama-root
paths. Ordinary MK automatically uses zero drafts; the full-vocabulary DFlash2
file cannot be paired directly with the condensed target.

## Interpretation

ASCII condensation changes tokenization and the output distribution. It retains
specials, byte fallback and the reference's partial UTF-8 tokens; it does not
prohibit non-ASCII output. Non-ASCII text may need more tokens and may lose quality.
The earlier base condensed target made an accent-copy error in both MK and
llama.cpp; this Swift derivative must be evaluated on its own. MTP removal cannot
save extra memory here because the chosen Swift source already has no MTP head.

The expected target vocabulary-weight reduction is 534,716,000 bytes, about
509.95 MiB. Q4 KV remains a separate lossy runtime choice. Context capacity,
populated prompt length, memory use, draft acceptance, and numerical agreement
will be reported separately; none establishes broad task accuracy on its own.
