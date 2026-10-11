# 03 — Initial port: output quality and limits

The existing 20-prompt suite used greedy decoding, up to 128 output tokens, and reset both target and draft between prompts. The hybrid produced correct answers on 10/10 numeric tasks. Full generated text matched the previous MK version on 19/20 prompts and the non-speculative llama.cpp baseline on 15/20.

These are small-suite comparisons, not a broad accuracy or perplexity guarantee. Speculative continuations can differ while remaining valid for the target sampling distribution; this early prototype also had unresolved numerical divergence. Re-run quality checks for any kernel, cache, draft-depth, or sampling change.
