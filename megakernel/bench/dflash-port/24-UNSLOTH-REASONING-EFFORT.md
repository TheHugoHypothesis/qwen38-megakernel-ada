# 24 — Unsloth Studio: reasoning effort without client changes

2026-10-11. This change updates the MK server's Python/HTTP compatibility while leaving GPU clocks, CUDA kernels, GGUF files, and the Unsloth installation untouched.

The diagnosis found that Studio sends `reasoning_effort` at the top level of the request, while the server previously read it only from nested template options. The server now reads the top-level field and supports aliases `high`→`xhigh` and `none`→reasoning disabled. Other request kwargs and the GGUF default are preserved. Restart the MK server to apply the change.

Validation passed 44 CPU tests/12 subtests and ten checks of GGUF-template rendering with the local serializer. No GPU inference was run for this HTTP-only change. See the server source and test artifacts for exact request cases.
