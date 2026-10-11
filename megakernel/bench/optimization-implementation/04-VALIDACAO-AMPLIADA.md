# Expanded validation: 20 prompts

The suite contains ten numeric problems with known answers and ten code/prose prompts in Portuguese and English. Non-speculative llama.cpp generated the reference. All engines used identical input token IDs and reset state between prompts.

All three engines answered 10/10 numeric tasks correctly. Full text/token equality differed across some open-ended prompts, so these checks should not be described as broad accuracy or perplexity equivalence. Preserve the prompt set and compare both task correctness and exact IDs when evaluating future changes.
