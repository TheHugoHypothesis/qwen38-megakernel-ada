#!/usr/bin/env python3
"""Export the measured native port results and bounded quality evidence."""
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

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'dflash-port'

def main():
    paths = [ROOT/'native-final/results.json', ROOT/'quality-final/results.json',
             ROOT/'native-f32-depth7/results.json', ROOT/'hybrid/results.json',
             ROOT/'regular-mk-regression/results.json',
             BASE/'optimization-implementation/dflash2-q4/results.json']
    final, quality, f32, hybrid, mtp, llama = [json.loads(p.read_text()) for p in paths]
    cases = ['short', 'long']
    def native_values(report):
        return [next(r['tps'] for r in report['summary'] if r['case']==c) for c in cases]
    values = [
        ('MK MTP4¹', [next(r['baseline_tps'] for r in mtp['summary'] if r['case']==c) for c in cases], '#92a1ad'),
        ('MK + draft GGML4¹', native_values(hybrid), '#87b6d6'),
        ('MK + draft CUDA F32/7', native_values(f32), '#b7a8d6'),
        ('MK + draft CUDA Q8/7', native_values(final), '#2466a8'),
        ('llama.cpp DFlash2/7¹', [next(r['tps'] for r in llama['summary'] if r['case']==c and r['draft']==7) for c in cases], '#338b65'),
    ]
    q=quality['summary']
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12.6,5.3),gridspec_kw={'width_ratios':[1.7,1]})
    x=np.arange(2);width=.16
    for i,(name,ys,color) in enumerate(values):
        bars=ax.bar(x+(i-2)*width,ys,width*.94,label=name,color=color)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    ax.set_xticks(x,['Prompt curto','Prompt longo'])
    ax.set_ylim(0,108);ax.set_ylabel('Tokens/s de decode');ax.grid(axis='y',alpha=.15)
    ax.legend(fontsize=8,loc='upper right');ax.set_title('128 tokens · 3 repetições · contexto 4K Q8')
    labels=['Tarefas numéricas\ncorretas','Texto igual ao\nMK MTP anterior','Texto igual ao\nllama sem especulação']
    qs=[100*q['correct_tasks']/q['tasks'],100*q['equal_prior_mk_contents']/q['generations'],100*q['equal_llama_contents']/q['generations']]
    bars=bx.barh(labels,qs,color=['#338b65','#2466a8','#92a1ad'])
    bx.bar_label(bars,labels=[f"{q['correct_tasks']}/{q['tasks']}",f"{q['equal_prior_mk_contents']}/{q['generations']}",f"{q['equal_llama_contents']}/{q['generations']}"],padding=4)
    bx.set_xlim(0,118);bx.invert_yaxis();bx.set_xlabel('% nesta suíte limitada');bx.set_title('Qualidade: evidências distintas')
    fig.suptitle('DFlash2 nativo no backend MK · RTX 4060 Ti 16 GB · clocks padrão',fontsize=13)
    fig.text(.025,.025,'¹ Baselines preservados de medições anteriores na mesma GPU. Load/prefill excluídos; continuação longa difere.\n'
             'Draft CUDA Graphs; target cooperativo MK. Não é um único kernel persistente. Acurácia ampla/perplexidade não estabelecidas.',fontsize=8.8)
    fig.tight_layout(rect=(0,.12,1,.94))
    out=ROOT/'native-report';out.mkdir(exist_ok=True)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'dflash-native.{ext}',dpi=180)
    result={'series':[{ 'name':name,'tps':dict(zip(cases,ys))} for name,ys,_ in values],
            'quality':q,'sources':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'samples':{c:[r['tps'] for r in final['runs'] if r['case']==c] for c in ['short','long','sampling']}}
    (out/'chart-data.json').write_text(json.dumps(result,indent=2)+'\n')
    print(out/'dflash-native.png')

if __name__=='__main__':main()
