# Context scaling benchmark

`context_scaling.py` measures native MK DFlash2 while increasing the number of
tokens actually prefetched. It launches a fresh process per context/KV pair,
uses a warmup plus repeated decode samples, and saves PNG/SVG/PDF, CSV, JSON,
per-run stdout, and logs. The graph shows decode throughput, prefill
throughput, speculative acceptance, and sampled VRAM. Failed preflights remain
recorded so the graph marks where Q8 stops fitting while Q4 continues.

The default sweep is a quick comparison through 16K:

```sh
python3 open-jet/megakernel/bench/context_scaling.py \
  --lengths 256 1024 4096 8192 16384 \
  --kv-formats q8 q4 --tokens 128 --reps 3 \
  --output open-jet/megakernel/bench/context-scaling-results
```

For a long sweep that reaches a 72K allocation, run:

```sh
python3 open-jet/megakernel/bench/context_scaling.py \
  --lengths 8192 16384 32768 49152 65536 71808 \
  --kv-formats q8 q4 --tokens 128 --reps 1 --prefill-chunk 128 \
  --timeout 3600 \
  --output open-jet/megakernel/bench/context-scaling-72k
```

The final point is 71,808 prompt tokens plus 128 generated tokens and 64
headroom tokens, for a 72,000-token allocation. Q8 points that exceed the
available VRAM are recorded as failed; Q4 is still measured. Long prefills are
expensive: the DFlash harness repeats prefill for warmup and each repetition.
Use `--reps 1` for the 72K sweep, then repeat only the sizes that matter.

The bundled prompt is the 24-token short fixture. The tool repeats its user
body while retaining the Qwen chat header and assistant suffix to create exact
prompt lengths. This is useful for measuring context-dependent kernel cost,
but repeated text can shift DFlash acceptance. For a representative workload,
pass a tokenized prompt with `--prompt-ids`; provide a full Qwen chat-formatted
prompt ending in the assistant generation prefix. The CSV/JSON include the
accepted draft fraction and tokens per cycle so throughput changes can be
interpreted alongside speculative efficiency.

Each point records the sampled whole-device VRAM peak from `nvidia-smi`; this
is an approximate peak and includes desktop processes. Keep clocks and other
GPU workloads unchanged across comparisons. Do not compare Q4 and Q8 points
with different populated lengths as an isolated KV-format speed comparison.

The checked-in-workspace graph/data from the initial expanded run lives in
`context-scaling-results/`. It used one measured repetition and 64 output
tokens per point, so it confirms the measurement path and shows the memory
trend, but it is not a final statistically stable throughput claim. Repeat
the sizes of interest with `--reps 3` or more before drawing small speed
conclusions.
