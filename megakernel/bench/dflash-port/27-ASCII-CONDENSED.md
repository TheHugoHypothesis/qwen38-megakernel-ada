# 27 — ASCII-condensed target, MTP-free export, and native DFlash2

2026-10-12. RTX 4060 Ti 16 GB, `sm_89`; stock clocks. No change to llama.cpp or model quantization.

## Verified input and exports

The downloaded [PlinyTheMedium release](https://huggingface.co/PlinyTheMedium/Qwen3.8-27B-GSQ-RCO-ASCII-Condensed) was pinned to `fb4b696c2de0b029bec89c9740903fdcb9a89559`. Its GGUF SHA-256 is `b18f585ee39446c0f8635dbc1c278d72a1530f133ae3b576e18e8156082b42f0`, matching the published checksum. The tokenizer retains 130,020 of 248,320 tokens, and the original MTP block is present.

`tools/prepare_condensed.py` verifies retained tokens, token types, remapped specials, and filtered BPE merge priority against the original DFlash2 tokenizer. With `--source-target`, it also checks every retained embedding/output row and every non-vocabulary tensor payload against the original GSQ-RCO GGUF. This run checked all rows, rather than a sample. The original source SHA-256 matched `58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f`.

| Export | Size in bytes | Tensors | SHA-256 |
|---|---:|---:|---|
| ASCII-condensed target without MTP | 11,230,505,920 | 851 | `6a1cce42ad0abc0b4ca00610ba761a746299af18ab5b4effc7df822c9b755a99` |
| Condensed DFlash2 Q4_K_M draft | 1,102,611,520 | 81 | `3b611e28dd72af6ced025ed2410927557304245ed244f6ddc14151ab061ccd96` |

The target export removes 15 `blk.64.*` tensors (348,469,248 payload bytes), sets block count to 64 and next-token-prediction layer count to zero, and preserves all trunk/vocabulary tensor bytes. The draft export gathers rows only in `selector_predecessor.weight` and `selector_successor.weight`, replaces the tokenizer with the target tokenizer, and remaps its mask token to 129,770. Other draft tensor payloads are unchanged. Neither export dequantizes or requantizes weights; all output payloads are exhaustively checked before the file is published locally.

The files, mapping array, and preparation manifest are in the workspace's `models-ascii-condensed-qwen38/prepared/` directory. Weights are not included in the source repository or uploaded anywhere.

## Runtime changes

- `model_spec.h` accepts either the 65-block/one-MTP schema or the 64-block/no-MTP schema and validates vocabulary dimensions against the tokenizer's token count. Vocabulary size must be even and between 32 and 248,320 because the target head processes row pairs.
- `qwen38_mk.cu/.h` derives vocabulary size from the actual embedding, uses it for score buffers/token bounds/shared tensors, and rejects internal MTP drafting when the model has no MTP block.
- `prefill.cuh` uses the runtime head width for scoring. The packed prefill GEMM requires complete 128-row tiles, so a bounded Q4_K tail kernel handles the condensed head's remaining rows. Decode already processes row pairs and needs no padded vocabulary.
- `dflash_native.cu` uses the runtime vocabulary width for logits, top-16 reduction, selector shapes, and token checks. A non-four-divisible even head uses the existing two-row MMQ path. Both full and condensed drafts retain the same five-layer CUDA Graph architecture.
- Native DFlash2 already skipped MTP weight uploads. Preflight now excludes those skipped weights, and `n_draft=0` allocates only 16 target attention slots, removing the previously unused seventeenth MTP slot. A no-MTP disk export alone does not save additional MTP weight VRAM on this already-skipping path.
- The launcher explains why an original-vocabulary draft cannot be paired directly with a condensed target. Open-Jet eligibility checks accept the validated condensed/no-MTP shape.

## Validation progress

The optimized `sm_89` build completed for the CLI, native DFlash2 CLI, C API library, and CUDA tests. Both the standalone package and the working `open-jet/` runtime contain the validated sources and binaries. Previous Open-Jet runtime binaries are preserved in `runtime-backups/ascii-condensed-before/` in the workspace.

Completed validation:

- Four synthetic export/remapping tests, malformed-GGUF/scalar parser checks, full CPU schema validation for the original and both condensed schemas, exhaustive input/output tensor-byte verification, and launcher/isolated integration-metadata checks.
- Four known-answer tasks per mode passed. Six generations matched exactly between the condensed MTP-present and MTP-free files when internal MTP was disabled. Native DFlash2 and scalar MK matched on all six cases; their output text also matched non-speculative llama.cpp on all six.
- All 36 external-verifier rejection/replay cases passed, including shared bytes, invalid IDs, 614 feature callbacks, and 1,120-token chunked prefill.
- The condensed output-head oracle compared 5,460,840 scores with the legacy native kernel exactly. Sampled CPU GGML/double comparisons had maximum NRMSE `3.73712109e-7` and absolute error `1.64463336e-6`, including the final vocabulary rows.
- Native-versus-GGML draft oracle: 110/112 candidate IDs matched, selector-score NRMSE was 0.0060662536, and proposals matched on the tested short fixture. This meets the existing oracle threshold and is not exact equality of every draft score.
- Scoring produced 130,020 finite logits, including the final 100-row partial tile. Filtered memcheck for the scoring-tail and argmax kernels reported zero errors. It does not sanitize every kernel or every context length.
- Thirteen HTTP/CUDA checks passed with native DFlash2, Q4 target KV, prefill chunk 128, and **96,000 allocated context capacity**. Runtime CUDA-visible use was **14.934 GiB**, with **0.721 GiB free**; sampled NVML use was 15,337/16,380 MiB. The longest populated prompt was **2,791 tokens**. This is a capacity/serving check, not full-96K input quality validation.
- The original 248,320-token GSQ-RCO target with its original DFlash2 draft passed the same 13 HTTP checks in the updated Open-Jet runtime at 4K/Q8.

The full Open-Jet Python pytest suite was not run: this environment lacks `pytest` and the integration import requires `tiktoken`. The standalone metadata checks and the real CUDA/HTTP tests above passed. Raw results and source hashes are in [the validation summary](../ascii-condensed-validation/summary.json) and adjacent artifacts.

## Memory and quality interpretation

Condensation removes 509.9 MiB of target embedding/head storage and 32.5 MiB from the draft selector matrices. Removing the unused MTP slot saves another 79.1 MiB of Q4 KV at 72,000 capacity (149.4 MiB for Q8). Physically removing the MTP tensors saves 332.3 MiB of payload on disk; those tensors were already skipped for native DFlash2, so it is not an additional MTP weight VRAM saving. The corrected preflight also stops budgeting skipped MTP weights.

The literal-copy prompt `Repeat exactly: café α → 中文 😀` produced `cafÿ α → 中文 😀` in all three MK modes **and in the llama.cpp reference**. This is an observed output-quality limitation with the condensed target, not a difference caused by physically stripping MTP or selecting native DFlash2 in these checks. The six-case suite is too small to establish broad quality equivalence or a general performance improvement. The existing context-scaling charts use the original unpruned target, not this new derivative.

Vocabulary pruning changes tokenization and the target's normalized output distribution. It can require more tokens for non-ASCII text. The adapted selector retains the original draft's trained rows but does not constitute retraining or establish equal draft acceptance. Compare it with the same condensed target's non-speculative output, and measure acceptance and quality on the actual workload.
