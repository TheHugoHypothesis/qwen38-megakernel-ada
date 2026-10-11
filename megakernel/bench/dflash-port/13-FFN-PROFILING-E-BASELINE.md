# 13 — MK port continuation: FFN profile and baseline

2026-10-10. Goal: improve eight-position target verification on the RTX 4060 Ti while preserving weight formats, validation, and stock clocks. The source/object/executable/library baseline is under `optimization-v2/baseline/` with `sha256.json`. llama.cpp was not modified.

Profiling identified target FFN as a primary optimization target. The comparison protocol requires rerunning previous MK, candidate MK, and llama.cpp/DFlash2 under the same settings; results from an RTX 3090 do not predict the Ada result.
