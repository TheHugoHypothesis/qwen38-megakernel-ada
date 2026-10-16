#!/usr/bin/env python3
"""Plot ada_verify_bench.py results, including regressions and ID agreement."""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('report', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    report = json.loads(args.report.read_text())
    rows = report['summary']
    cases = report['cases']
    variants = list(dict.fromkeys(row['variant'] for row in rows))
    output = args.output or args.report.parent
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/qwen38-mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cases), figsize=(max(7, 4.3 * len(cases)), 5.4), squeeze=False)
    colors = {'preserved7': '#87929c', 'full7': '#335c81', 'depth3': '#d38b35',
              'fused3': '#508a68', 'verify3': '#9b6f9e', 'adaptive7': '#4b899a'}
    labels = {'preserved7': 'Previous / 7', 'full7': 'Updated / 7', 'depth3': 'Draft 3 / legacy',
              'fused3': 'Draft 3 / fused', 'verify3': 'Draft 7 / verify 3', 'adaptive7': 'Draft 7 / auto'}
    for axis, case in zip(axes[0], cases):
        selected = [next(row for row in rows if row['case'] == case and row['variant'] == v) for v in variants]
        x = list(range(len(selected)))
        bars = axis.bar(x, [r['mean_tps'] for r in selected],
                        color=[colors[v] for v in variants],
                        yerr=[r.get('stddev_tps') or 0 for r in selected], capsize=3)
        axis.bar_label(bars, fmt='%.1f', padding=4, fontsize=9)
        axis.set_xticks(x, [labels[v] for v in variants], rotation=40, ha='right', fontsize=9)
        axis.set_title(case.replace('_', ' ').title())
        axis.set_ylabel('Generated tokens / second')
        axis.set_ylim(0, max(r['mean_tps'] for r in selected) * 1.22)
        axis.grid(axis='y', alpha=.18)
        axis.set_axisbelow(True)
    repetitions = report['repetitions']
    fig.suptitle(f'Swift 1.5 GSQ-RCO ASCII-condensed · RTX 4060 Ti · {report["kv"].upper()} KV\n'
                 f'{report["ctx"]:,} context capacity · {report["tokens_limit"]} output-token limit · stock clocks')
    agreement = sum(r['full7_ids_equal'] is True for r in rows)
    compared = sum(r['full7_ids_equal'] is not None for r in rows)
    fig.text(.02, .025, f'{repetitions} measured repetition(s) per configuration; bars show mean, whiskers sample SD. '
             f'Exact full7 output IDs: {agreement}/{compared} case/configuration groups.\n'
             'Native decode timing excludes model loading, prefill and the first token. ID agreement is not task accuracy.', fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .89))
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'ada-verification.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
