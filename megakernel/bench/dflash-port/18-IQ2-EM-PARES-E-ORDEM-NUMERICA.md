# 18 — Paired IQ2 processing while preserving numerical order

2026-10-10/11. This candidate covered 57 layers and was still undergoing compilation/validation when this snapshot was recorded.

The first candidate reduced target time to about 67–68 ms per launch and reached 89.985/59.921 tok/s, but three open-ended responses differed from the previous native version. All 10 known-answer tasks remained correct. The candidate was preserved under `optimization-v2/fused-six-v1/` with hashes before further source changes. This illustrates why task accuracy and token-identical output checks are both retained.
