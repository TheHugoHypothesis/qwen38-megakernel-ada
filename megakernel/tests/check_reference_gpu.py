#!/usr/bin/env python3
"""Check functional token-ID preservation against a recorded same-model Q8 run.

Baseline preservation does not imply the recorded model answer was correct;
known failures, including Unicode pruning differences, are retained explicitly.
"""
import argparse,ctypes,hashlib,json,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'server'))
from mk_server import Engine


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--draft-model',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--lib',type=Path,default=ROOT/'libqwen38mk.so')
    a=ap.parse_args();reference=json.loads(a.reference.read_text())
    if Path(reference['model']).name!=a.model.name or reference['kv']!='q8':ap.error('reference must match the target and Q8 cache')
    names={x['name'] for x in reference['cases'] if not x['benchmark']}
    os.environ['MK_PREFILL_CHUNK']='128';os.environ['MK_DFLASH_VERIFY']='full';os.environ['MK_FFN_IQ_SMALL']='0'
    result={'library_sha256':hashlib.sha256(a.lib.read_bytes()).hexdigest(),'checks':[],
            'limits':__doc__}
    for mode,draft in [('mk_scalar',None),('mk_dflash2',a.draft_model)]:
        eng=Engine(str(a.lib.resolve()),str(a.model.resolve()),4096,7 if draft else 0,None,1,
                   draft_model=str(draft.resolve()) if draft else None)
        eng.L.mk_free.argtypes=[ctypes.c_void_p];eng.L.mk_free.restype=None
        try:
            for row in reference['runs']:
                if row['mode']!=mode or row['rep']!=0 or row['case'] not in names:continue
                eng.reset();eng.set_sampling(0,.95,20,0,123)
                ids=[eng.prefill(row['prompt_ids'])]
                while len(ids)<64 and not eng.is_eog(ids[-1]):
                    for token in eng.step(64-len(ids)):
                        ids.append(token)
                        if eng.is_eog(token):break
                valid=[t for t in ids if not eng.is_eog(t)]
                same=valid==row['output_ids']
                text=b''.join(eng.piece(t) for t in valid).decode(errors='replace')
                result['checks'].append({'mode':mode,'case':row['case'],'same_ids':same,'ids':valid,'text':text,
                                         'reference_task_correct':row.get('task_correct')})
                a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
                if not same:raise AssertionError(f"{mode} {row['case']} changed reference IDs")
        finally:eng.L.mk_free(eng.h)
    print('PASS',len(result['checks']),'functional baseline-preservation checks',flush=True)


if __name__=='__main__':main()
