# DFlash2-to-MK port feasibility

2026-10-10. This was an investigation only: no kernels were changed and no new benchmarks were run. The recommended next step was a hybrid proof of feasibility. The relevant baseline was llama.cpp plus DFlash2, not MK plus built-in MTP.

The existing target verifier was limited to MAXT=5 while the DFlash2 model proposes blocks of eight. Raising the CLI draft count alone would not update attention shapes, shared-memory use, rejection/replay state, or validation. The real draft GGUF also does not provide an independent target embedding/head, so a hybrid needed feature taps and safe borrowing/sharing of target weights.

The proposed prototype would capture features from target layers `[6,20,34,48,62]`, feed them to the draft, verify proposals in MK, and test rollback at every rejection position. Compare its memory and end-to-end throughput against llama.cpp/DFlash2 with identical tokens, sampling, and context. This work later became the hybrid and native port; see [DFLASH_PORT_LOG.md](../../DFLASH_PORT_LOG.md).
