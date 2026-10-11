#!/usr/bin/env python3
"""Regenerate the implementation report from recorded GPU measurements."""
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'optimization-implementation'
OUTPUT = RESULTS / 'report'


def main():
    OUTPUT.mkdir(exist_ok=True)
    sources = {}

    def read(rel):
        path = ROOT / rel
        raw = path.read_bytes()
        sources[rel] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    ab = read('optimization-implementation/final-ab/results.json')
    sampling = read('optimization-implementation/final-sampling/results.json')
    df = read('optimization-implementation/dflash2-q4/results.json')
    df_sampling = read('optimization-implementation/dflash2-sampling/results.json')
    decode = {}
    for case in ('short', 'long', 'sampling'):
        spec = sampling if case == 'sampling' else ab
        name = 'short' if case == 'sampling' else case
        decode[case] = {
            'Megakernel original': [r['decode_tps'] for r in spec['runs']
                                   if r['case'] == name and r['engine'] == 'baseline'],
            'Megakernel IQ16': [r['decode_tps'] for r in spec['runs']
                               if r['case'] == name and r['engine'] == 'candidate']}
        llama = read(f'optimization-analysis/llama-mtp4-{case}/results.json')
        decode[case]['llama.cpp MTP 4'] = [r['timings']['predicted_per_second'] for r in llama['llama']]
        draft = df_sampling if case == 'sampling' else df
        decode[case]['llama.cpp DFlash2 7'] = [r['response']['timings']['predicted_per_second']
                                             for r in draft['runs'] if r['case'] == name and r['draft'] == 7]

    prefill = {engine: [r['prefill_seconds'] for r in ab['runs']
                        if r['case'] == 'long' and r['engine'] == engine]
               for engine in ('baseline', 'candidate')}
    quality = {}
    for name, folder in [('Referência sem spec.', 'quality-nospec'),
                         ('Megakernel IQ16', 'quality-megakernel'),
                         ('DFlash2 7', 'quality-dflash2')]:
        rows = read(f'optimization-implementation/{folder}/results.json')['runs']
        tasks = [r for r in rows if 'task_correct' in r]
        quality[name] = {'correct_tasks': sum(r['task_correct'] for r in tasks),
                         'tasks': len(tasks), 'generations': len(rows),
                         'identical_generations': len(rows) if folder == 'quality-nospec'
                         else sum(r['canonical_ids_equal'] for r in rows)}

    df32 = read('optimization-implementation/dflash2-32k/results.json')
    memory = {'megakernel': {}, 'dflash2': {}}
    for ctx in (4096, 32768, 65536):
        rel = f'optimization-implementation/final-checks/context-{ctx}.log'
        path = ROOT / rel
        sources[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        values = re.search(r'([\d.]+) GiB engine allocations, ([\d.]+) GiB device used, ([\d.]+) GiB free', path.read_text())
        memory['megakernel'][ctx] = dict(zip(('engine_gib', 'device_used_gib', 'free_gib'), map(float, values.groups())))
    for ctx, draft in [(4096, read('optimization-implementation/quality-dflash2/results.json')), (32768, df32)]:
        snapshots = [r['gpu_after_request'] for r in draft['runs']]
        peak = max(int(re.search(r'(\d+) MiB,', s)[1]) for s in snapshots)
        memory['dflash2'][ctx] = {'max_observed_nvml_mib': peak, 'max_observed_nvml_gib': peak / 1024}

    gains = {case: 100 * (mean(v['Megakernel IQ16']) / mean(v['Megakernel original']) - 1)
             for case, v in decode.items()}
    prefill_gain = 100 * (mean(prefill['baseline']) / mean(prefill['candidate']) - 1)
    data = {'decode_tps_samples': decode, 'megakernel_gain_percent': gains,
            'prefill_long_seconds': prefill, 'prefill_long_gain_percent': prefill_gain,
            'quality': quality, 'memory': memory, 'source_sha256': sources,
            'notes': ['Clocks padrão; sm_89; Qwen3.8-27B GSQ-RCO IQ3_S-mtp; contexto 4096/Q8; 128 tokens.',
                      'Decode exclui carga e prefill; 3 repetições, exceto llama.cpp longo/sampling com 2.',
                      'Barras de erro são mínimo/máximo observados, não intervalos de confiança.',
                      'A/B megakernel intercalado; llama.cpp/DFlash2 medidos em execuções separadas.',
                      'No caso longo, a continuação megakernel difere da referência llama.cpp.',
                      'Sampling com seed 123 e temp=1/top_k=20/top_p=.95; não compara identidade entre engines.',
                      'Dez problemas numéricos e vinte prompts não demonstram qualidade universal; PPL não medida.',
                      'Memória: snapshots observados, não pico instrumentado; CUDA e NVML diferem.']}
    (OUTPUT / 'chart_data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#f7f9fc', 'axes.facecolor': '#f7f9fc',
                         'savefig.facecolor': '#f7f9fc', 'font.family': 'DejaVu Sans'})
    fig = plt.figure(figsize=(14, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=.39, wspace=.28)
    ax = fig.add_subplot(grid[0, :])
    colors = ['#8793a4', '#2866c8', '#d99226', '#25866e']
    engines = list(decode['short'])
    xs = np.arange(3)
    width = .19
    for i, engine in enumerate(engines):
        samples = [decode[c][engine] for c in decode]
        avgs = np.array([mean(s) for s in samples])
        err = np.array([[m - min(s) for m, s in zip(avgs, samples)],
                        [max(s) - m for m, s in zip(avgs, samples)]])
        bars = ax.bar(xs + (i - 1.5) * width, avgs, width, color=colors[i], label=engine,
                      yerr=err, error_kw={'capsize': 3, 'linewidth': 1})
        ax.bar_label(bars, labels=[f'{v:.1f}'.replace('.', ',') for v in avgs], padding=5, fontsize=11)
    ax.set_xticks(xs, ['Greedy · 24 tokens de entrada', 'Greedy · 1365 tokens de entrada¹', 'Sampling · 24 tokens de entrada'])
    ax.set_ylabel('Geração (tokens/s)')
    ax.set_ylim(0, 112)
    ax.set_title('Throughput medido na RTX 4060 Ti 16 GB', loc='left', fontsize=17, weight='bold', pad=16)
    ax.legend(loc='upper left', ncol=4, frameon=False, fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18)

    gain_ax = fig.add_subplot(grid[1, 0])
    labels = ['Decode curto', 'Decode longo', 'Decode sampling', 'Prefill longo²']
    values = list(gains.values()) + [prefill_gain]
    bars = gain_ax.barh(labels, values, color=[colors[1]] * 3 + ['#57a1da'])
    gain_ax.bar_label(bars, labels=[f'+{v:.1f}%'.replace('.', ',') for v in values], padding=5)
    gain_ax.set_xlim(0, 38)
    gain_ax.invert_yaxis()
    gain_ax.set_xlabel('Ganho de throughput sobre o megakernel original (%)')
    gain_ax.set_title('Efeito das mudanças no megakernel', loc='left', fontsize=14, weight='bold', pad=12)
    gain_ax.grid(axis='x', alpha=.18)
    gain_ax.set_axisbelow(True)

    qax = fig.add_subplot(grid[1, 1])
    x = np.arange(3)
    task_values = [100 * r['correct_tasks'] / r['tasks'] for r in quality.values()]
    id_values = [100 * r['identical_generations'] / r['generations'] for r in quality.values()]
    b1 = qax.bar(x - .17, task_values, .32, color='#25866e', label='Problemas numéricos corretos')
    b2 = qax.bar(x + .17, id_values, .32, color='#8793a4', label='Gerações idênticas à referência')
    qax.bar_label(b1, labels=[f"{r['correct_tasks']}/{r['tasks']}" for r in quality.values()], padding=4)
    qax.bar_label(b2, labels=[f"{r['identical_generations']}/{r['generations']}" for r in quality.values()], padding=4)
    qax.set_xticks(x, ['Referência\nsem especulação', 'Megakernel\nIQ16', 'DFlash2\n7 propostas'])
    qax.set_ylim(0, 130)
    qax.set_yticks([0, 25, 50, 75, 100])
    qax.set_ylabel('Percentual (%)')
    qax.set_title('Corretude no conjunto local', loc='left', fontsize=14, weight='bold', pad=12)
    qax.legend(loc='upper left', frameon=False, fontsize=9)
    qax.grid(axis='y', alpha=.18)
    qax.set_axisbelow(True)

    fig.suptitle('Otimizações implementadas · pesos GSQ-RCO preservados · clocks padrão', x=.06, y=.975,
                 ha='left', fontsize=19, weight='bold', color='#17283c')
    fig.subplots_adjust(left=.08, right=.97, top=.9, bottom=.18)
    fig.text(.08, .11, '¹ Continuações diferentes no longo afetam a aceitação. Decode exclui carga e prefill.\n'
             '² Prefill de 1365 tokens: 2,575 → 1,988 s (−22,8% de latência).\n'
             'Qualidade: 10 contas e 20 prompts; identidade não é acurácia. PPL e qualidade ampla não medidas.\n'
             'Teacher forcing MK longo (chunk 512): 127/128. VRAM DFlash2: ~14,06 GiB em 4K; ~15,13 GiB em 32K.',
             ha='left', va='top', fontsize=10, color='#435366', linespacing=1.6)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUTPUT / f'ganhos_implementados.{ext}', dpi=180, bbox_inches='tight', pad_inches=.2)
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    main()
