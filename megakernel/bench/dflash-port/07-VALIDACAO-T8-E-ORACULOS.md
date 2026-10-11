# 07 — T=8 verification and intermediate oracles

The DeltaNet convolution ring was enlarged from 8 to 16 positions because T=8 requires retaining T+3 values. The verifier passed depths 1/4/7, every rejection position, scalar replay to 64 tokens, 243 feature callbacks, long prefill, and shared-weight checks.

The CPU oracle matched 1,271,808 Q4_K/Q6_K values exactly; 2,548 F32 dot products had maximum error 8.08e-6. Dynamic convolution passed for both branches. Independent sliding attention at position 2050 with eight tokens and a 2048-token window had maximum error 1.26e-9.
