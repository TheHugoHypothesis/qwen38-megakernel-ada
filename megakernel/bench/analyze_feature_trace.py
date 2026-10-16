#!/usr/bin/env python3
"""Compare production DFlash layer-input traces at common token positions."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('legacy',type=Path);ap.add_argument('candidate',type=Path)
    ap.add_argument('--prompt-ids',type=Path,required=True)
    ap.add_argument('--reference-report',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); read=lambda p:list(map(int,p.read_text().split()))
    x,y=read(a.legacy.with_suffix('.ids')),read(a.candidate.with_suffix('.ids'))
    prompt=read(a.prompt_ids)
    first_id=next((i for i,(p,q) in enumerate(zip(x,y)) if p!=q),min(len(x),len(y)))
    def shape(p):
        size=p.with_suffix('.f32').stat().st_size
        if not size or size%(64*5120*4):raise ValueError('truncated or empty feature trace')
        return (size//(64*5120*4),64,5120)
    left=np.memmap(a.legacy.with_suffix('.f32'),mode='r',dtype=np.float32,shape=shape(a.legacy))
    right=np.memmap(a.candidate.with_suffix('.f32'),mode='r',dtype=np.float32,shape=shape(a.candidate))
    n=min(len(left),len(right),len(prompt)+first_id)
    changes=[]
    first_per_layer={}
    for position in range(n):
        for layer in range(64):
            u,v=left[position,layer],right[position,layer]
            different=u.view(np.uint32)!=v.view(np.uint32)
            if not np.any(different):continue
            delta=v.astype(np.float64)-u
            row={'position':position,'generated_offset':position-len(prompt),'layer_input':layer,
                 'different_elements':int(np.count_nonzero(different)),
                 'max_abs':float(np.max(np.abs(delta))),
                 'nrmse':float(np.linalg.norm(delta)/max(np.linalg.norm(u.astype(np.float64)),1e-30))}
            if len(changes)<16:changes.append(row)
            first_per_layer.setdefault(layer,row)
    report={'first_output_difference':first_id,'same_output_ids':x==y,'common_positions_compared':n,
            'first_changed_features':changes,'first_change_per_layer':list(first_per_layer.values()),
            'legacy_ids':x,'candidate_ids':y,
            'limits':'Layer inputs localize a first affected layer, not the operation within it. Traces include only committed feature positions.'}
    if a.reference_report:
        old=json.loads(a.reference_report.read_text())['runs']
        for name,ids,variant in [('legacy',x,'depth3'),('candidate',y,'fused3')]:
            reference=next(r for r in old if r['case']=='numbered_lines' and r['variant']==variant)
            report[name+'_matches_untraced_ids']=ids==reference['ids']
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('legacy_ids','candidate_ids','first_change_per_layer')},indent=2))


if __name__=='__main__':main()
