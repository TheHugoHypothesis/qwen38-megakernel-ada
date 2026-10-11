#!/usr/bin/env python3
"""Export measured before/after throughput and bounded quality evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/qwen38-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent / 'dflash-port' / 'optimization-v2'
CASES = ('short', 'long')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, nargs='+',
                        default=[BASE/'before-b/results.json', BASE/'before-c/results.json'])
    parser.add_argument('--after', type=Path, nargs='+',
                        default=[BASE/'after-a/results.json', BASE/'after-b/results.json'])
    parser.add_argument('--llama', type=Path, nargs='+',
                        default=[BASE/'llama-fresh-b/results.json', BASE/'llama-fresh-c/results.json'])
    parser.add_argument('--quality', type=Path, default=BASE/'quality-final/results.json')
    parser.add_argument('--output', type=Path, default=BASE/'report')
    args = parser.parse_args()
    paths = args.before + args.after + args.llama + [args.quality]
    reports = {str(p): json.loads(p.read_text()) for p in paths}
    series = []
    for label, inputs, variant, color in (
        ('MK antes desta etapa', args.before, 'original', '#95a2ad'),
        ('MK otimizado nesta etapa', args.after, 'inlined', '#2466a8'),
        ('llama.cpp DFlash2', args.llama, None, '#338b65'),
    ):
        values = {}
        for case in CASES:
            rows = [r for p in inputs for r in reports[str(p)]['runs']
                    if r['case'] == case and (r.get('variant') == variant if variant else r['draft'] == 7)]
            if not rows:
                raise ValueError(f'No samples for {label}/{case}')
            if variant and not all(r['canonical_ids_equal'] for r in rows):
                raise ValueError(f'Changed native fixture IDs in {label}/{case}')
            samples = [r['tps'] if variant else r['response']['timings']['predicted_per_second'] for r in rows]
            values[case] = {'mean': statistics.mean(samples), 'min': min(samples),
                            'max': max(samples), 'samples': samples}
        series.append({'name': label, 'values': values, 'color': color})
    quality = reports[str(args.quality)]['summary']
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 5.3),
                                 gridspec_kw={'width_ratios': [1.5, 1]})
    x = np.arange(len(CASES)); width = .24
    for index, item in enumerate(series):
        means = [item['values'][c]['mean'] for c in CASES]
        ranges = [[item['values'][c]['mean'] - item['values'][c]['min'] for c in CASES],
                  [item['values'][c]['max'] - item['values'][c]['mean'] for c in CASES]]
        bars = ax.bar(x + (index - 1) * width, means, width * .93,
                      label=item['name'], color=item['color'], yerr=ranges, capsize=3)
        ax.bar_label(bars, fmt='%.2f', padding=5, fontsize=9)
    ax.set_xticks(x, ['Prompt curto', 'Prompt longo¹'])
    ax.set_ylabel('Tokens/s de decode'); ax.set_ylim(0, 108)
    ax.grid(axis='y', alpha=.15); ax.legend(fontsize=9, loc='upper right')
    ax.set_title('128 tokens · contexto 4K Q8 · sete propostas')
    totals = [quality['tasks'], quality['generations'], quality['generations']]
    counts = [quality['correct_tasks'], quality['equal_prior_native_ids'], quality['equal_llama_contents']]
    bars = bx.barh(['Tarefas objetivas corretas', 'IDs iguais ao MK anterior',
                   'Texto igual ao llama sem spec.'],
                  [100 * n / d for n, d in zip(counts, totals)],
                  color=['#338b65', '#2466a8', '#95a2ad'])
    bx.bar_label(bars, labels=[f'{n}/{d}' for n, d in zip(counts, totals)], padding=4)
    bx.invert_yaxis(); bx.set_xlim(0, 119)
    bx.set_xlabel('% nesta suíte limitada'); bx.set_title('Qualidade: evidências específicas')
    fig.suptitle('Otimização do MK + DFlash2 · RTX 4060 Ti 16 GB · clocks padrão', fontsize=13)
    fig.text(.025, .025,
             '¹ A continuação longa diverge entre MK e llama; o caso curto coincide. Barras de erro: mínimo–máximo observado.\n'
             'Decode exclui load, prefill e primeiro token. Suite pequena: acurácia ampla e perplexidade não estabelecidas.\n'
             'Target cooperativo MK; draft CUDA Graphs. O draft ainda não é um único kernel persistente.', fontsize=8.7)
    fig.tight_layout(rect=(0, .15, 1, .94))
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(args.output/f'iq-optimization.{extension}', dpi=180)
    data = {'series': series, 'quality': quality,
            'gains_pct': {c: 100 * (series[1]['values'][c]['mean'] / series[0]['values'][c]['mean'] - 1) for c in CASES},
            'versus_llama_pct': {c: 100 * (series[1]['values'][c]['mean'] / series[2]['values'][c]['mean'] - 1) for c in CASES},
            'limits': 'Six observed samples per bar; two processes/server sessions. Min-max is not a confidence interval. Long continuations differ between engines. Quality ID/text equality is not an accuracy score.',
            'sources': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output/'chart-data.json').write_text(json.dumps(data, indent=2)+'\n')
    print(args.output/'iq-optimization.png')


if __name__ == '__main__':
    main()
