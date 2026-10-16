#!/usr/bin/env python3
"""Measure megakernel DFlash2 speed as the *populated* prompt grows.

Prompt IDs are made by cycling a real tokenized prompt fixture. This gives a
deterministic, exact-length compute workload, but repeated text can affect
speculative acceptance. Use a representative tokenized corpus with --prompt-ids
when comparing user workloads. Each (length, KV format) is a fresh process.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent
DEFAULT_MODEL = ENGINE.parent.parent / "models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF/blobs/58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f"
DEFAULT_DRAFT = ENGINE.parent.parent / "models-dflash2-qwen38/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
DEFAULT_PROMPT = HERE / "context-scaling-showcase/prompt.ids"


def read_ids(path):
    ids = [int(x) for x in path.read_text().split()]
    if not ids or any(x < 0 for x in ids):
        raise ValueError(f"invalid/empty token IDs: {path}")
    return ids


def make_prompt(source, length, path, end_token):
    # The bundled fixtures use Qwen's normal chat layout. Repeat the user-body
    # tokens while retaining its header and final assistant-generation suffix;
    # cycling the complete file would splice an im_end into the middle.
    try:
        end = source.index(end_token)
    except ValueError:
        end = -1
    if end >= 3:
        prefix, body, suffix = source[:3], source[3:end], source[end:]
    else:
        prefix, body, suffix = [], source, []
    if not body:
        raise ValueError("source prompt has no repeatable body tokens")
    if length < len(prefix) + len(suffix):
        raise ValueError("requested length cannot fit the chat header and generation suffix")
    body_size = length - len(prefix) - len(suffix)
    repeated = (body * ((body_size + len(body) - 1) // len(body)))[:body_size]
    ids = prefix + repeated + suffix
    if path is not None:
        path.write_text(" ".join(map(str, ids)) + "\n")
    return ids


def gpu_memory_mib():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=3)
        return int(result.stdout.splitlines()[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


def run_configuration(args, prompt_file, prompt_tokens, kv, ctx, output_dir):
    cmd = [str(args.binary.resolve()), "-m", str(args.model.resolve()),
           "--draft-model", str(args.draft_model.resolve()), "--prompt-ids", str(prompt_file),
           "-d", str(args.depth), "-c", str(ctx), "--backend", "native",
           "--prefill-chunk", str(args.prefill_chunk), "-n", str(args.tokens),
           "--reps", str(args.reps), "--temp", str(args.temperature), "--seed", str(args.seed),
           "--kv", kv]
    env = os.environ.copy()
    # Keep the measurement about the selected production path, not profiling hooks.
    for key in ("MK_PROFILE", "MK_PF_PROFILE", "MK_CHECK_FINITE", "MK_SCORE_PROBE_POSITION", "MK_SCORE_PROBE_FILE"):
        env.pop(key, None)
    libdir = args.llama_root.resolve() / "build/bin"
    env["LD_LIBRARY_PATH"] = str(libdir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    stem = f"ctx{prompt_tokens}-{kv}"
    out_path, log_path = output_dir / f"{stem}.out", output_dir / f"{stem}.log"
    samples, rows = [], []
    start = time.monotonic()
    with out_path.open("w") as stdout, log_path.open("w") as stderr:
        proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
        deadline = start + args.timeout
        try:
            while proc.poll() is None:
                used = gpu_memory_mib()
                if used is not None:
                    samples.append(used)
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{stem} exceeded {args.timeout}s")
                time.sleep(args.sample_interval)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    elapsed = time.monotonic() - start
    if proc.returncode:
        return {"prompt_tokens": prompt_tokens, "context_capacity": ctx, "kv": kv,
                "status": "failed", "returncode": proc.returncode, "elapsed_seconds": elapsed,
                "sampled_peak_gpu_mib": max(samples) if samples else None,
                "stdout": str(out_path), "log": str(log_path), "command": cmd}
    for line in out_path.read_text().splitlines():
        if line.startswith("RESULT "):
            row = json.loads(line[7:])
            if row.get("rep", -1) >= 0:
                rows.append(row)
    if len(rows) != args.reps or any(row.get("tokens", 0) < args.tokens * .8 for row in rows):
        return {"prompt_tokens": prompt_tokens, "context_capacity": ctx, "kv": kv,
                "status": "incomplete", "results_found": len(rows),
                "generated_tokens": [row.get("tokens") for row in rows], "elapsed_seconds": elapsed,
                "sampled_peak_gpu_mib": max(samples) if samples else None,
                "stdout": str(out_path), "log": str(log_path), "command": cmd}
    tps = [r["tps"] for r in rows]
    prefill_tps = [prompt_tokens / r["prefill_seconds"] for r in rows if r["prefill_seconds"] > 0]
    return {"prompt_tokens": prompt_tokens, "context_capacity": ctx, "kv": kv, "status": "ok",
            "repetitions": len(rows), "decode_tps_mean": statistics.mean(tps),
            "decode_tps_stdev": statistics.stdev(tps) if len(tps) > 1 else 0.0,
            "prefill_seconds_mean": statistics.mean(r["prefill_seconds"] for r in rows),
            "prefill_tps_mean": statistics.mean(prefill_tps) if prefill_tps else None,
            "accepted_fraction": (sum(r["accepted"] for r in rows) / sum(r["drafted"] for r in rows)
                                  if sum(r["drafted"] for r in rows) else None),
            "mean_tokens_per_cycle": (statistics.mean(r["tokens"] / r["cycles"] for r in rows if r["cycles"])
                                      if any(r["cycles"] for r in rows) else None),
            "sampled_peak_gpu_mib": max(samples) if samples else None,
            "samples": rows, "elapsed_seconds": elapsed,
            "stdout": str(out_path), "log": str(log_path), "command": cmd}


def write_graph(report, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen38-mpl-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    formats = sorted({r["kv"] for r in report["results"]})
    tick_values = sorted({r["prompt_tokens"] for r in report["results"]})
    def tick_label(n):
        return f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n)
    colors = {"q8": "#2764a5", "q4": "#df7b26", "f16": "#4c956c"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    ax, bx, cx, dx = axes.flat
    for kv in formats:
        points = sorted((r for r in report["results"] if r["kv"] == kv and r["status"] == "ok"),
                        key=lambda r: r["prompt_tokens"])
        if not points:
            continue
        x = [r["prompt_tokens"] for r in points]
        y = [r["decode_tps_mean"] for r in points]
        err = [r["decode_tps_stdev"] for r in points]
        color = colors.get(kv, None)
        if report["config"]["reps"] > 1:
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3, linewidth=2, label=kv.upper(), color=color)
        else:
            ax.plot(x, y, marker="o", linewidth=2, label=kv.upper(), color=color)
        bx.plot(x, [r["prefill_tps_mean"] for r in points], marker="o", linewidth=2,
                label=kv.upper(), color=color)
        cx.plot(x, [100 * r["accepted_fraction"] if r["accepted_fraction"] is not None else float("nan")
                    for r in points], marker="o", linewidth=2, label=kv.upper(), color=color)
        dx.plot(x, [r["sampled_peak_gpu_mib"] for r in points], marker="o", linewidth=2,
                label=kv.upper(), color=color)
    ymin, ymax = ax.get_ylim()
    for r in report["results"]:
        if r["status"] != "ok":
            ax.scatter([r["prompt_tokens"]], [ymin + .02 * (ymax-ymin)], marker="x",
                       color=colors.get(r["kv"], "#666"), clip_on=False)
    for axis in (ax, bx, cx, dx):
        axis.set_xscale("log", base=2)
        axis.set_xticks(tick_values, [tick_label(n) for n in tick_values])
        axis.set_xlabel("Populated prompt length (tokens)")
        axis.grid(True, which="both", alpha=.2)
        axis.legend(title="Target KV")
    ax.set_ylabel("Decode throughput (generated tokens/s)")
    ax.set_title("Generation speed after prefill")
    bx.set_ylabel("Prefill throughput (prompt tokens/s)")
    bx.set_title("Prompt ingestion speed")
    cx.set_ylabel("Draft tokens accepted (%)")
    cx.set_ylim(0, 105)
    cx.set_title("Speculative acceptance")
    dx.set_ylabel("Sampled peak device memory (MiB)")
    dx.set_title("VRAM footprint (whole device)")
    fig.suptitle(f"Qwen3.8-27B native DFlash2 · {report['config']['depth']} draft tokens · "
                 f"{report['config']['tokens']} output tokens · chunk {report['config']['prefill_chunk']}")
    variability = ("Decode error bars show run-to-run standard deviation. " if report["config"]["reps"] > 1
                   else "One measured repetition per point; variability is not estimated. ")
    fig.text(.02, .015, "Fresh process per point. Prompts repeat the selected fixture body; this can affect draft acceptance. "
             + variability + "VRAM is sampled whole-device nvidia-smi usage.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"context-scaling.{ext}", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT)
    ap.add_argument("--binary", type=Path, default=ENGINE / "qwen38_dflash")
    ap.add_argument("--llama-root", type=Path, default=ENGINE.parents[1] / "llama.cpp",
                    help="llama.cpp checkout providing gguf-py for tokenizer validation")
    ap.add_argument("--prompt-ids", type=Path,
                    help="tokenized source prompt; the bundled default is for the original target only")
    ap.add_argument("--lengths", nargs="+", type=int, default=[256, 1024, 4096, 8192, 16384],
                    help="actual populated prompt lengths, in tokens")
    ap.add_argument("--kv-formats", nargs="+", choices=("q4", "q8"), default=["q8", "q4"])
    ap.add_argument("--output", type=Path, default=HERE / "context-scaling-results")
    ap.add_argument("--tokens", type=int, default=128, help="generated tokens per repetition")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--prefill-chunk", default="128")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--context-overhead", type=int, default=64,
                    help="capacity tokens reserved beyond prompt+generation (default 64)")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds allowed per configuration")
    ap.add_argument("--sample-interval", type=float, default=.5)
    args = ap.parse_args()
    if args.tokens < 2 or args.reps < 1 or args.depth < 1 or args.context_overhead < 0:
        ap.error("tokens >= 2, reps >= 1, depth >= 1 and context-overhead >= 0 are required")
    if any(n < 1 for n in args.lengths):
        ap.error("prompt lengths must be positive")
    if args.prompt_ids is None:
        if args.model.resolve() != DEFAULT_MODEL.resolve():
            ap.error("a different target requires explicit --prompt-ids tokenized with that target")
        args.prompt_ids = DEFAULT_PROMPT
    for p in (args.model, args.draft_model, args.binary, args.prompt_ids):
        if not p.is_file():
            ap.error(f"file not found: {p}")
    sys.path.insert(0, str(args.llama_root.resolve() / 'gguf-py'))
    from gguf import GGUFReader
    target, draft = GGUFReader(args.model), GGUFReader(args.draft_model)
    def metadata(reader, key):
        return reader.fields[key].contents() if key in reader.fields else None
    for key in ('tokenizer.ggml.tokens', 'tokenizer.ggml.merges', 'tokenizer.ggml.eos_token_id'):
        if metadata(target, key) != metadata(draft, key):
            ap.error(f'target and draft tokenizer differ: {key}')
    vocab_size = len(metadata(target, 'tokenizer.ggml.tokens'))
    end_token = int(metadata(target, 'tokenizer.ggml.eos_token_id'))
    args.output.mkdir(parents=True, exist_ok=True)
    prompt_source = read_ids(args.prompt_ids)
    if any(token >= vocab_size for token in prompt_source):
        ap.error('source prompt contains token IDs outside the target vocabulary')
    for length in args.lengths:
        try:
            make_prompt(prompt_source, length, None, end_token)
        except ValueError as error:
            ap.error(str(error))
    report = {"description": "Context scaling for populated prompts; prompt IDs are cycled from a source fixture.",
              "config": {"model": str(args.model), "draft_model": str(args.draft_model),
                         "binary": str(args.binary), "source_prompt_ids": str(args.prompt_ids),
                         "source_prompt_length": len(prompt_source), "kv_formats": args.kv_formats,
                         "vocabulary_size": vocab_size, "eos_token_id": end_token,
                         "tokens": args.tokens, "reps": args.reps, "depth": args.depth,
                         "prefill_chunk": args.prefill_chunk, "temperature": args.temperature,
                         "seed": args.seed, "context_overhead": args.context_overhead},
              "results": []}
    scratch = args.output / "prompts"
    scratch.mkdir(exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    total = len(set(args.lengths)) * len(args.kv_formats)
    for i, length in enumerate(sorted(set(args.lengths)), 1):
        prompt_file = scratch / f"prompt-{length}.ids"
        make_prompt(prompt_source, length, prompt_file, end_token)
        # The engine keeps a minimum-sized base/workspace allocation even for
        # tiny prompts; longer points reserve enough for prompt and output.
        capacity = max(4096, length + args.tokens + args.context_overhead)
        for kv in args.kv_formats:
            print(f"[{i}/{len(set(args.lengths))}, {kv}] prompt={length} ctx={capacity}", flush=True)
            row = run_configuration(args, prompt_file, length, kv, capacity, args.output)
            report["results"].append(row)
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: row.get(k) for k in
                              ("prompt_tokens", "context_capacity", "kv", "status", "decode_tps_mean",
                               "prefill_tps_mean", "sampled_peak_gpu_mib", "log")}), flush=True)
    with (args.output / "results.csv").open("w", newline="") as f:
        fields = ["prompt_tokens", "context_capacity", "kv", "status", "repetitions",
                  "decode_tps_mean", "decode_tps_stdev", "prefill_seconds_mean", "prefill_tps_mean",
                  "accepted_fraction", "mean_tokens_per_cycle", "sampled_peak_gpu_mib", "elapsed_seconds", "log"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["results"])
    write_graph(report, args.output)
    print(f"\nGraph and measurements saved under: {args.output}")


if __name__ == "__main__":
    main()
