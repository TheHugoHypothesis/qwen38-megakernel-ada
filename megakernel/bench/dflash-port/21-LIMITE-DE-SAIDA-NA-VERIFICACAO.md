# 21 — Limit verification work at the output boundary

2026-10-11. Typed MMQ R4 was selected; the stage 19 target was preserved.

`dflash_main.cpp` and `capi.cpp::mk_step` now limit the number of proposals verified to the remaining output-token budget. The decoder still produces the trained eight-token block and keeps its original proposals; only verification is shortened. When one output token remains, it verifies the anchor with no proposals and skips a full draft pass. `verify_external` already supports T=1, so no CUDA kernel change was required.
