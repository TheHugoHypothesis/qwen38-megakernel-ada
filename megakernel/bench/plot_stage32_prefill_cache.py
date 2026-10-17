#!/usr/bin/env python3
"""Plot Stage 32 long-context prefill, DFlash feature-window injection, and multi-turn prompt caching results."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen38-mpl-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "bench/ada-deep/comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    s31_agent = json.loads((root / "bench/ada-deep/agent/candidate-production/results.json").read_text())["runs"]
    s32_agent = json.loads((root / "bench/ada-deep/agent/stage32-chunk-auto/results.json").read_text())["runs"]
    s31_cache = json.loads((root / "bench/ada-deep/stage31-agent-cache-baseline/results.json").read_text())["turns"]
    s32_cache_128 = json.loads((root / "bench/ada-deep/stage32-agent-cache-chunk128/results.json").read_text())["turns"]
    s32_cache_auto = json.loads((root / "bench/ada-deep/stage32-agent-cache-auto/results.json").read_text())["turns"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    ax1, ax2, ax3 = axes

    # Panel 1: Long-context prefill latency (8.2K, 16.8K, 33.7K)
    patch_31 = sorted([r for r in s31_agent if r["case"]["workload"] == "patch"], key=lambda r: r["case"]["prompt_tokens"])
    patch_32 = sorted([r for r in s32_agent if r["case"]["workload"] == "patch"], key=lambda r: r["case"]["prompt_tokens"])
    labels_p1 = [f"{r['case']['prompt_tokens']:,}\ntok" for r in patch_31]
    x1 = np.arange(len(labels_p1))
    w = 0.35
    v31_pf = [r["prefill_seconds"] for r in patch_31]
    v32_pf = [r["prefill_seconds"] for r in patch_32]
    b1 = ax1.bar(x1 - w / 2, v31_pf, w, label="Stage 31 (chunk=128)", color="#6c757d")
    b2 = ax1.bar(x1 + w / 2, v32_pf, w, label="Stage 32 (chunk=auto)", color="#198754")
    for bar, s31, s32 in zip(b2, v31_pf, v32_pf):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0,
                 f"{s32:.1f}s\n({s31/s32:.2f}×)", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_xticks(x1, labels_p1)
    ax1.set_ylabel("Prefill time (seconds, lower is better)")
    ax1.set_ylim(0, max(v31_pf) * 1.16)
    ax1.set_title("Populated Agent Context Prefill")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(fontsize=8)

    # Panel 2: DFlash prefill GPU feature injection time (ms)
    v31_inj = [r["native_injection_prefill_gpu_seconds"] * 1e3 for r in patch_31]
    v32_inj = [r["native_injection_prefill_gpu_seconds"] * 1e3 for r in patch_32]
    b3 = ax2.bar(x1 - w / 2, v31_inj, w, label="Stage 31 (full prompt injection)", color="#6c757d")
    b4 = ax2.bar(x1 + w / 2, v32_inj, w, label="Stage 32 (2,056-row SWA window)", color="#0d6efd")
    for bar, s31, s32 in zip(b4, v31_inj, v32_inj):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 22.0,
                 f"{s32:.1f} ms\n({s31/s32:.1f}×)", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_xticks(x1, labels_p1)
    ax2.set_ylabel("DFlash GPU injection (ms, lower is better)")
    ax2.set_ylim(0, max(v31_inj) * 1.18)
    ax2.set_title("DFlash Prefill Feature Injection")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(fontsize=8)

    # Panel 3: 4-Turn 8.2K Agent Tool-Loop Prefill Latency (log scale)
    turn_labels = ["Turn 1\nCold 8.1K", "Turn 2\n8.1K Retry", "Turn 3\nTool #1", "Turn 4\nTool #2"]
    x3 = np.arange(len(turn_labels))
    w3 = 0.27
    t31 = [t["prompt_ms"] for t in s31_cache]
    t32_128 = [t["prompt_ms"] for t in s32_cache_128]
    t32_auto = [t["prompt_ms"] for t in s32_cache_auto]
    ax3.bar(x3 - w3, t31, w3, label="Stage 31 (115.2s total)", color="#6c757d")
    ax3.bar(x3, t32_128, w3, label="Stage 32 chunk=128 (29.2s total)", color="#fd7e14")
    b_auto = ax3.bar(x3 + w3, t32_auto, w3, label="Stage 32 chunk=auto (11.9s total)", color="#198754")
    ax3.set_yscale("log")
    for bar, s31, s32 in zip(b_auto, t31, t32_auto):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.18,
                 f"{s31/s32:.1f}×", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax3.set_xticks(x3, turn_labels)
    ax3.set_ylabel("Turn prefill latency (ms, log scale)")
    ax3.set_ylim(20, 90000)
    ax3.set_title("4-Turn 8.2K Agent Tool Loop & Cache Reuse")
    ax3.grid(axis="y", which="both", alpha=0.25)
    ax3.legend(fontsize=8, loc="upper right")

    fig.suptitle("Stage 32 — Long-Context Prefill, SWA Window Injection & Multi-Turn Agent Prompt Caching (RTX 4060 Ti 16 GB)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"stage32-prefill-and-caching.{ext}", dpi=180)
    plt.close(fig)
    print("Saved plots to", out_dir / "stage32-prefill-and-caching.{png,svg,pdf}")


if __name__ == "__main__":
    main()
