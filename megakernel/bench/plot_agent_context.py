#!/usr/bin/env python3
"""Plot populated agent-context production results and separately profiled phases."""
import argparse,json,os
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--production',nargs='+',type=Path,required=True)
    ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/qwen38-mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    profiles=json.loads(a.profile.read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8));ax,bx,cx,dx=axes.flat
    for path in a.production:
        r=json.loads(path.read_text());label=path.parent.name
        for workload in ('patch','tool'):
            rows=sorted((x for x in r['runs'] if x['case']['workload']==workload),key=lambda x:x['case']['prompt_tokens'])
            if not rows:continue
            x=[q['case']['prompt_tokens']/1024 for q in rows];name=label+' / '+workload
            ax.plot(x,[q['tps'] for q in rows],'o-',label=name)
            bx.plot(x,[q['prefill_seconds'] for q in rows],'o-',label=name)
            cx.plot(x,[100*q['acceptance'] for q in rows],'o-',label=name)
            dx.plot(x,[q['tokens_per_cycle'] for q in rows],'o-',label=name)
    for axis in axes.flat:
        axis.set_xlabel('Populated prompt (Ki tokens)');axis.grid(alpha=.2);axis.legend(fontsize=8)
    ax.set_ylabel('Generated tokens/s');ax.set_title('Production decode (no profiling)')
    bx.set_ylabel('Seconds');bx.set_title('Prefill latency, including first output token')
    cx.set_ylabel('Draft proposals accepted (%)');cx.set_ylim(0,105);cx.set_title('Acceptance depends on the request')
    dx.set_ylabel('New tokens / target cycle');dx.set_title('Effective verification yield')
    fig.suptitle('Swift 1.5 ASCII-condensed · native DFlash2 · RTX 4060 Ti · stock clocks')
    fig.text(.02,.015,'Synthetic agent/tool conversations rendered with the target template. Early EOS is retained.\nSingle-run points are exploratory; no task-accuracy or general performance superiority claim.',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.95))
    for ext in ('png','svg','pdf'):fig.savefig(a.output/f'agent-context.{ext}',dpi=180)
    plt.close(fig)
    rows=profiles['runs'];fig,axis=plt.subplots(figsize=(11,5.5));bottom=[0.]*len(rows)
    groups={'FFN gate/up':['ffn_gu'],'FFN down':['ffn_down'],'Attention':['attn_core'],
            'GDN core':['dn_core'],'Other target work':[]}
    used=set(sum((v for v in groups.values()),[]))
    for group,keys in groups.items():
        values=[sum(v for k,v in r['phases_ms'].items() if k in keys or (not keys and k not in used)) for r in rows]
        axis.bar(range(len(rows)),values,bottom=bottom,label=group)
        bottom=[p+q for p,q in zip(bottom,values)]
    axis.set_xticks(range(len(rows)),[f"{r['case']['workload']}\n{r['case']['prompt_tokens']:,}" for r in rows])
    axis.set_ylabel('Instrumented target milliseconds / launch');axis.legend(fontsize=8)
    axis.set_title('Populated agent-context target phases; draft/host costs excluded')
    fig.text(.02,.015,'Phase timings are instrumented and separate from the production throughput chart. Mixed verification/tail shapes contribute to each average.',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ('png','svg','pdf'):fig.savefig(a.output/f'agent-target-phases.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
