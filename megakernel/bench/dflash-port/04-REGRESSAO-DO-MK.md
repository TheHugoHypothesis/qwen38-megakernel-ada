# 04 — A/B regression check for the existing MK path

The preserved IQ16 optimized binary was compared with the current binary using MTP4, three interleaved repetitions, 128 output tokens, 4096 context, Q8 KV, identical prompt IDs, and automatic chunk selection.

No measurable regression was observed: throughput variation remained below 0.1%, and generated IDs matched the preserved baseline. This check only covers the recorded prompts and settings.
