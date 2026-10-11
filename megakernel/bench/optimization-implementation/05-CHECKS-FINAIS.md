# Final A/B, memory, and numerical diagnosis

The final A/B interleaved three repetitions, using automatic chunking for the candidate and chunk 512 for the original. It measured about +25.5% short, +24.8% long, and +25.1% sampling MK throughput. The candidate preserved generated IDs against the original in these fixtures.

The long teacher-forcing comparison still diverged at token 41 (127/128 matched). This remained unresolved and limits claims of numerical equivalence. Memory and full commands are retained in the per-run JSON and implementation summary.
