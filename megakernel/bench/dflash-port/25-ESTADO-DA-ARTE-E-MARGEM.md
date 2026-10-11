# 25 — Competitive results, state of the art, and optimization headroom

2026-10-11. This review examined recorded results and primary sources. It did not start a new benchmark/inference, modify kernels, or restart the active server. GPU clocks remained at stock settings.

The current result is competitive, but state-of-the-art performance is not demonstrated. The selected native MK result was +1.49% against llama.cpp/DFlash2 on the short fixture and -0.34% on the long fixture. These small differences and overlapping observed ranges do not support a broad claim. Validation on representative prompts with more repetitions is needed.

The review prioritizes target FFN/quantized matrix-vector work and realistic end-to-end measurement. Further draft-only fusion has limited upside when draft time is a small part of the complete cycle. Amdahl-law scenarios in the report are estimates, not new performance results. The local RTX 4060 Ti data should be compared against local llama.cpp under identical settings; results from an RTX 3090 cannot establish performance on Ada.
