# Historical performance and correctness chart

This chart predates the IQ16/prefill optimizations and the DFlash2 test. DFlash2 was later downloaded and validated in this workspace through llama.cpp. The current performance summary is in [IMPLEMENTATION_RESULTS.md](../../IMPLEMENTATION_RESULTS.md). Statements below about missing drafts/tests describe the earlier investigation and are retained as historical context. Chart image exports are not included in this source snapshot.

Updated on 2026-10-10 with MTP benchmarks, profiles, and chunk-size sweeps. The plotting script reads existing records; GPU clocks remained at stock settings.

- `ganhos_e_correcao.png`: raster chart.
- `ganhos_e_correcao.svg` and `.pdf`: export formats.
- `chart_data.json`: values derived from raw records.
- `../plot_ada_results.py`: regeneration script (Python, matplotlib, NumPy).

The updated chart compares four proposals in both engines. Megakernel versus llama.cpp measured -23.49% greedy short (48.17 vs. 62.96 tok/s), -23.73% sampling (47.24 vs. 61.93), and -12.27% greedy long (39.93 vs. 45.51). Long continuations differed, which affects speculative acceptance. Earlier +4.63% and +2.67% figures used MTP2 in llama.cpp and did not demonstrate an advantage over the tuned baseline. Non-MTP decode was also slower in the megakernel.

Most configurations had two repetitions; the short greedy llama.cpp MTP case had three. There is no confidence interval or long-duration stability test. Generation timing excludes model load and prefill.

Token agreement is not task accuracy. In the initial reference, 1,280/1,280 tokens matched across ten short-prompt generations and 512/512 across four longer-prompt generations. These are repeated outputs from two prompts, not 1,792 independent examples; each continuation was 128 tokens. Initial teacher forcing matched 128/128 on both prompts. The long prompt was also checked with chunks 512 and 128.

The newer MTP4 reference matched non-speculative llama.cpp on the long prompt but diverged from the earlier MTP2 reference and the megakernel at zero-based index 41. The chart reports teacher forcing against the newer reference: 128/128 short and **127/128 = 99.22% long**. The cause has not been isolated; general numerical equivalence is not established. Teacher forcing uses `--score`, whose default is the FP16 prefill path; this percentage is not agreement between free-running generations.

Perplexity, task benchmarks, and statistical equivalence of sampling distributions were not measured. The 64K context allocation test used a short prompt and does not measure quality with a 64K-token input.

## Historical DFlash2 investigation

At the time of the original Open-Jet inspection, no DFlash/DFlash2 path was found in that code. Its `--draft 2` selected two proposals from the GGUF's built-in MTP block; it did not mean DFlash2. The local llama.cpp checkout already contained DFlash2 support in `common/speculative.cpp`, `src/models/dflash.cpp`, and `common/arg.cpp`.

The model cache initially contained only the IQ3_S-MTP target and BF16 mmproj, with no DFlash2 draft GGUF. Therefore the original investigation could not claim that DFlash2 ran with this target or quantify its VRAM/performance. It later found the [Qwen3.8-27B-DFlash2-GGUF draft](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF), including a Q4_K_M file around 1.14 GB. This provided a candidate, not a guarantee of compatibility, acceptance, output quality, or speed on the RTX 4060 Ti.

Integration into MK requires target hidden-state taps, draft convolutions and selector, plus block verification. The existing llama.cpp path was the appropriate reference before writing new kernels. For the subsequent implementation, see [DFLASH_PORT_LOG.md](../../DFLASH_PORT_LOG.md) and the full [optimization analysis](../../OPTIMIZATION_ANALYSIS.md).
