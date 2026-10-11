# 09 — API integration, memory, and proposal depth

The CLI, C API `mk_new_dflash`, and HTTP server use the native CUDA draft; the GGML reference remains selectable only in the CLI. The default uses DP4A/Q8 math, CUDA Graphs per depth, and FP16 sliding-window KV with 2056 rows. F32 remains available with `MK_DFLASH_MATH=f32`.

Removing MTP target weights saves about 0.343 GiB. The implementation reserves 1792 MiB explicitly for draft, encoder, and features before automatic target prefill-chunk selection, preventing target workspace from consuming the full VRAM budget.

API and output checks passed for the tested flows. The server is serial; concurrent sequences and broad quality equivalence were not established here.
