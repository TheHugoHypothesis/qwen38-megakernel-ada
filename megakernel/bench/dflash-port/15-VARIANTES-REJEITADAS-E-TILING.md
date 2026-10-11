# 15 — IQ variants, negative results, and tiling follow-up

All runs used stock clocks, with measured sources and binaries preserved as snapshots with SHA-256 hashes. This stage integrated candidate IQ/FFN paths and tested a small factorial set of tiling choices.

Variants that increased registers, wasted work, or failed to improve end-to-end throughput were rejected. The retained direction was specialized packed IQ processing with independent validation, rather than assuming tensor-core conversion or a larger tile would help on `sm_89`. Further candidates were to be compared by complete target-cycle timing and output IDs.
