# Completion: API, sampling, and allocated 32K context

DFlash2 was tested with 32K allocated context, `--fit off`, full GPU offload, Q8 KV, one sequence, and seven proposals. The run loaded and generated successfully. This confirms allocation and a generation smoke test at that setting; it is not a benchmark with 32K populated prompt tokens.

The HTTP/API tests covered the launchers and sampling path. Context allocation, populated context, and output length are distinct; do not infer long-context throughput or quality from the allocation test alone.
