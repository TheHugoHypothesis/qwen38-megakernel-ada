# 01 — External verifier and feature capture

2026-10-10, RTX 4060 Ti `sm_89`, stock clocks.

Implemented `MKEngine::verify_external`, feature capture via `enable_features`, and borrowed target tensors. Tests covered accept-all and rejection at every proposal position, scalar continuation IDs, state after rejection, shared tensor bytes, layout, and chunked prefill. The feature path captures the required target-layer inputs without retaining the full context.

The complete test command and raw artifacts are retained alongside this journal. This validates the tested verifier cases; it does not establish broad model quality or long-context numerical equivalence.
