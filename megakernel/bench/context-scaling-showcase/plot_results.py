#!/usr/bin/env python3
"""Create the GitHub-ready context scaling plots from results.csv."""
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen38-mpl-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("results.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("assets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("prompt_tokens", "decode_tps_mean", "prefill_tps_mean", "accepted_fraction",
                    "sampled_peak_gpu_mib"):
            row[key] = float(row[key]) if row[key] else float("nan")
    formats = sorted({r["kv"] for r in rows})
    points = {kv: sorted((r for r in rows if r["kv"] == kv), key=lambda r: r["prompt_tokens"])
              for kv in formats}
    colors = {"q4": "#df7b26", "q8": "#2764a5"}
    sizes = sorted({int(r["prompt_tokens"]) for r in rows})
    labels = [f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n) for n in sizes]
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "semibold", "figure.facecolor": "white"})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for kv in formats:
        p = points[kv]
        x = [r["prompt_tokens"] for r in p]
        axes[0].plot(x, [r["decode_tps_mean"] for r in p], "o-", lw=2.2, ms=6,
                     color=colors.get(kv), label=kv.upper())
        axes[1].plot(x, [r["prefill_tps_mean"] for r in p], "o-", lw=2.2, ms=6,
                     color=colors.get(kv), label=kv.upper())
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, labels)
        ax.set_xlabel("Populated prompt length")
        ax.grid(True, which="major", alpha=.2)
        ax.legend(frameon=False, title="Target KV")
    axes[0].set_title("Decode throughput")
    axes[0].set_ylabel("Generated tokens / second")
    axes[1].set_title("Prefill throughput")
    axes[1].set_ylabel("Prompt tokens / second")
    fig.suptitle("Qwen3.8-27B · native DFlash2 on RTX 4060 Ti", fontsize=14, weight="semibold")
    fig.tight_layout(rect=(0, 0, 1, .92))
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"throughput.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for kv in formats:
        p = points[kv]
        x = [r["prompt_tokens"] for r in p]
        axes[0].plot(x, [r["sampled_peak_gpu_mib"] for r in p], "o-", lw=2.2, ms=6,
                     color=colors.get(kv), label=kv.upper())
        axes[1].plot(x, [100 * r["accepted_fraction"] for r in p], "o-", lw=2.2, ms=6,
                     color=colors.get(kv), label=kv.upper())
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, labels)
        ax.set_xlabel("Populated prompt length")
        ax.grid(True, which="major", alpha=.2)
        ax.legend(frameon=False, title="Target KV")
    axes[0].set_title("Sampled peak device memory")
    axes[0].set_ylabel("MiB · whole device")
    axes[1].set_title("DFlash draft acceptance")
    axes[1].set_ylabel("Accepted draft tokens (%)")
    axes[1].set_ylim(0, 105)
    fig.suptitle("Memory footprint and speculative efficiency", fontsize=14, weight="semibold")
    fig.tight_layout(rect=(0, 0, 1, .92))
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"memory-and-acceptance.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots written to {args.output}")


if __name__ == "__main__":
    main()
