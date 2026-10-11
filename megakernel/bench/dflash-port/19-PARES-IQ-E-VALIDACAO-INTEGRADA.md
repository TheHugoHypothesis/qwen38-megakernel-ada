# 19 — Paired six-IQ path and integrated target validation

2026-10-11. Snapshot `optimization-v2/paired-six-v1/` includes sources, objects, binaries, and hashes. Clocks remained at stock settings.

For `rows_native_impl`, when T>5/R2, the `MK_IQ_PAIRED` option (default 1; `MK_IQ2_PAIRED` alias) processes two rows together for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, and IQ4_XS. IQ2 retains two dot8 updates; IQ3/IQ4 retain dot16. Guards preserve the T≤5 path. Integrated tests compare outputs and correctness with the prior snapshot.
