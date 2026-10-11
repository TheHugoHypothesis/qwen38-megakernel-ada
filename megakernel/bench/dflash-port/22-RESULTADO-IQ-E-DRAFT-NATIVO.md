# 22 — Target IQ optimization and native-draft MMQ results

2026-10-11. RTX 4060 Ti 16 GB, `sm_89`, 34 SMs. Clocks, power limit, driver, and model weights were unchanged. The selected build is preserved under `optimization-v2/selected-v2/` with sources, objects, executables, library, and SHA-256. llama.cpp remains at `aa94f2086147cd4bfbcff818ce998dffce5fcb39`.

## Outcome

The selected MK + native DFlash2 build improved from about 81 to 93 tok/s on the short prompt in the controlled comparison. It nominally beat llama.cpp on that case. On the long case it was close to the llama.cpp mean, with observed ranges overlapping. **This does not demonstrate general superiority over llama.cpp.** The target is a cooperative megakernel; the complete draft decoder still runs in CUDA Graphs. Fusing the draft into one persistent kernel remains future work.

Across six samples, selected throughput was 93.21/62.28/93.36 tok/s for short/long/sampling. Relative to the prior MK snapshot, gains were 15.18%/15.40%/15.88%. Against a fresh llama.cpp run, results were +1.49% short and -0.34% long. Do not treat the small short-prompt difference as a conclusive overall win.

## Changes and evidence

Stages 13–21 profiled target FFN, fused gate/up paths for six IQ formats, tested format pairs, specialized/inlined kernels, and paired-row layouts. IMMA and pipeline prototypes that did not improve the measured end-to-end path were rejected. Typed Q4_K/Q6_K MMQ uses four rows per warp with explicit output-tail handling. The selected candidate also limits final verification to the remaining output budget while retaining the trained draft block size.

Validation covered three preserved 128-token fixtures, 10/10 known-answer tasks, 17/20 generations matching the prior native version, 36 external-verifier cases, 13 HTTP checks, 34 CPU tests, and bounded CUDA memchecks. A two-turn HTTP test exposed an incorrect checkpoint; the issue was fixed and the test rerun. The passing checks do not prove broad quality equivalence or long-run numerical equivalence.

The target remained the ISTA-DASLab GSQ-RCO mixed-precision GGUF; no requantization occurred. DFlash2 remains Q4_K_M and Q8 target KV is the conservative default. Full raw results, source/build references, limits, and follow-up priorities are in `optimization-v2/selected-v2/` and this journal's adjacent artifacts.
