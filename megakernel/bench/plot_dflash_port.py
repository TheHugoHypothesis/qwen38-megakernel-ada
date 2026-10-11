#!/usr/bin/env python3
"""Export prototype performance and limited task-quality evidence from raw JSON."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/qwen38-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent/'dflash-port'


def main():
    paths = [ROOT/'hybrid/results.json', ROOT/'llama-dflash4/results.json',
             ROOT/'regular-mk-regression/results.json', ROOT/'quality/results.json']
    hybrid, llama, regression, quality = [json.loads(p.read_text()) for p in paths]
    cases = ['short', 'long']
    series = [
        ('MK MTP4 anterior', [next(r['baseline_tps'] for r in regression['summary'] if r['case']==c) for c in cases], '#8a99a8'),
        ('MK + DFlash2 4', [next(r['tps'] for r in hybrid['summary'] if r['case']==c) for c in cases], '#2672bb'),
        ('llama.cpp + DFlash2 4', [next(r['tps'] for r in llama['summary'] if r['case']==c) for c in cases], '#329064'),
    ]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 5.2), gridspec_kw={'width_ratios':[1.45, 1]})
    x = np.arange(2)
    for i, (name, values, color) in enumerate(series):
        bars = ax.bar(x+(i-1)*.24, values, .23, label=name, color=color)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)
    ax.set_xticks(x, ['Prompt curto', 'Prompt longo'])
    ax.set_ylabel('Tokens/s de decode')
    ax.set_ylim(0, 91)
    ax.set_title('128 tokens · 3 repetições · contexto Q8 4096')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=.15)
    q = quality['summary']
    values = [100*q['correct_tasks']/q['tasks'], 100*q['equal_prior_mk_contents']/q['generations'],
              100*q['equal_llama_contents']/q['generations']]
    bars = bx.barh(['Problemas numéricos\ncorretos', 'Texto igual ao\nMK anterior', 'Texto igual ao\nllama sem especulação'], values,
                   color=['#329064', '#2672bb', '#8a99a8'])
    bx.bar_label(bars, labels=[f"{q['correct_tasks']}/{q['tasks']}", f"{q['equal_prior_mk_contents']}/{q['generations']}",
                            f"{q['equal_llama_contents']}/{q['generations']}"], padding=4)
    bx.set_xlim(0, 118)
    bx.set_xlabel('% nesta suíte limitada')
    bx.set_title('Qualidade: métricas distintas')
    bx.invert_yaxis()
    fig.suptitle('Primeiro porte DFlash2 para MK · RTX 4060 Ti 16 GB · clocks padrão', fontsize=13)
    fig.text(.02, .025, 'Load/prefill excluídos. Continuação longa difere entre engines. Identidade de texto não mede acurácia ampla.\n'
             'Draft GGML/CUDA reutilizado; verificador MK limitado a 4 propostas. Equivalência numérica completa continua pendente.', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .93))
    out = ROOT/'report'
    out.mkdir(exist_ok=True)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out/f'dflash-port.{ext}', dpi=180)
    (out/'sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in paths}, indent=2)+'\n')
    print(out/'dflash-port.png')


if __name__ == '__main__':
    main()
