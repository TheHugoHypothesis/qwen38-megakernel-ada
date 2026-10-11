# 02 — Real draft model and memory budget

The `qwen38_dflash` executable uses MK for the target and llama.cpp's existing DFlash2 GGML/CUDA graph for the draft. It reuses the draft quantizations, convolutions, Flash Attention, selector, and CUDA Graph path. Anchor-plus-mask feature injection avoids retaining the entire feature history.

Three-run throughput was 67.05 tok/s short, 55.65 long, and 64.41 with sampling. Sampled NVML peak use was 14,945 MiB. These measurements belong to the initial hybrid build; later native CUDA results are in stages 11 and 22.

Memory includes target weights, target KV, DeltaNet state, feature staging, draft weights, draft KV, workspaces, graph/library overhead, and display/background use. Do not treat `cudaMalloc` totals as total device use.
