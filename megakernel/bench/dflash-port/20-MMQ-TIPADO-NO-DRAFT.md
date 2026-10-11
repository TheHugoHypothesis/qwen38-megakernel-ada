# 20 — Typed MMQ and R4 tiles in the native draft

2026-10-11. The target from stage 19 was unchanged.

`dflash_native.cu::fragment_typed<TYPE>` specializes Q4_K or Q6_K before the K loop; the generic `fragment` remains for embedding and reference use. `mmq_typed<T,TYPE,ROWS>` uses two or four rows per warp while sharing input/scale reads. It preserves dot8, explicit `fmaf`, offsets, and XOR reduction. `small_mm` selects the kernel during CUDA Graph capture. No buffers or weight conversions were added.
