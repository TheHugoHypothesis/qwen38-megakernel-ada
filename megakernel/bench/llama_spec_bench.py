#!/usr/bin/env python3
"""Benchmark local llama.cpp speculative methods with recorded prompt IDs."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request
from statistics import mean

ROOT = Path(__file__).resolve().parent


def request(base, endpoint, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base+endpoint, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,type=Path)
    ap.add_argument('--draft-model',type=Path)
    ap.add_argument('--server',required=True,type=Path)
    ap.add_argument('--type',choices=['draft-mtp','draft-dflash','none'],default='draft-dflash')
    ap.add_argument('--drafts',default='4,7')
    ap.add_argument('--cases',default='short,long')
    ap.add_argument('--ctx',type=int,default=4096)
    ap.add_argument('--tokens',type=int,default=128)
    ap.add_argument('--reps',type=int,default=3)
    ap.add_argument('--port',type=int,default=18097)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--suite',type=Path)
    ap.add_argument('--reference',type=Path)
    ap.add_argument('--temperature',type=float,default=0)
    ap.add_argument('--fit',choices=('on','off'),default='on')
    args=ap.parse_args()
    if args.type=='draft-dflash' and not args.draft_model:
        ap.error('--draft-model is required for draft-dflash')
    args.output.mkdir(parents=True,exist_ok=True)
    base=f'http://127.0.0.1:{args.port}'
    report={'type':args.type,'context':args.ctx,'temperature':args.temperature,'runs':[],'configurations':[]}
    specs=json.loads(args.suite.read_text()) if args.suite else [{'name':x} for x in args.cases.split(',')]
    references={}
    if args.reference:
        for row in json.loads(args.reference.read_text())['runs']:
            if row['rep']==0: references[row['case']]=row['response']['tokens']
    def save():
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    for depth in map(int,args.drafts.split(',')):
        cmd=[str(args.server.resolve()),'-m',str(args.model.resolve()),'-ngl','99','-fa','on',
             '-ctk','q8_0','-ctv','q8_0','-c',str(args.ctx),'--parallel','1',
             '--host','127.0.0.1','--port',str(args.port),'--fit',args.fit]
        if args.type!='none':
            cmd+=['--spec-type',args.type,'--spec-draft-n-max',str(depth)]
        if args.draft_model:
            cmd+=['--spec-draft-model',str(args.draft_model.resolve()),'--spec-draft-ngl','99',
                  '--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0']
        report['configurations'].append({'depth':depth,'command':cmd})
        save()
        with (args.output/f'llama-d{depth}.log').open('w') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+300
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'llama-server exited ({process.returncode}); see log')
                    try:
                        with urllib.request.urlopen(base+'/health',timeout=1) as response:
                            if response.status==200:break
                    except OSError:pass
                    if time.monotonic()>deadline:raise TimeoutError('server startup')
                    time.sleep(.25)
                report['configurations'][-1]['gpu_loaded']=subprocess.run(['nvidia-smi','--query-gpu=memory.used,clocks.current.graphics,clocks.current.memory','--format=csv'],capture_output=True,text=True).stdout
                save()
                for spec in specs:
                    case=spec['name']
                    if args.suite:
                        text=f"<|im_start|>user\n{spec['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                        prompt_ids=request(base,'/tokenize',{'content':text,'add_special':False,'parse_special':True})['tokens']
                        reference=references.get(case)
                    else:
                        canonical=json.loads((ROOT/f'optimization-analysis/llama-mtp4-{case}/results.json').read_text())
                        prompt_ids=canonical['prompt_ids']; reference=canonical['llama'][0]['tokens']
                    if args.temperature>0:reference=None
                    payload={'prompt':prompt_ids,'n_predict':args.tokens,
                             'temperature':args.temperature,'top_k':20,'top_p':.95,'min_p':0,'seed':123,
                             'repeat_penalty':1,'cache_prompt':False,'return_tokens':True}
                    warm=dict(payload,n_predict=16)
                    request(base,'/completion',warm)
                    for rep in range(args.reps):
                        start=time.monotonic()
                        response=request(base,'/completion',payload)
                        elapsed=time.monotonic()-start
                        ids=response.get('tokens',[])
                        prefix=next((i for i,(a,b) in enumerate(zip(ids,reference)) if a!=b),min(len(ids),len(reference))) if reference is not None else None
                        row={'case':case,'draft':depth,'rep':rep,'response':response,
                             'prompt_ids':prompt_ids,
                             'request_seconds':elapsed,'canonical_prefix':prefix,
                             'canonical_ids_equal':ids==reference if reference is not None else None}
                        if 'answer' in spec:
                            try: answer=json.loads(response['content'].strip())['answer']
                            except (ValueError,KeyError,TypeError): answer=None
                            row['expected_answer']=spec['answer']; row['answer']=answer
                            row['task_correct']=answer==spec['answer']
                        row['gpu_after_request']=subprocess.run(['nvidia-smi','--query-gpu=memory.used,clocks.current.graphics,clocks.current.memory','--format=csv'],capture_output=True,text=True).stdout
                        report['runs'].append(row);save()
                        print(case,depth,rep,response['timings']['predicted_per_second'],
                              'canonical prefix',prefix,flush=True)
            finally:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    report['summary']=[]
    for case in [x['name'] for x in specs]:
        for depth in map(int,args.drafts.split(',')):
            rows=[r for r in report['runs'] if r['case']==case and r['draft']==depth]
            report['summary'].append({'case':case,'draft':depth,'tps':mean(r['response']['timings']['predicted_per_second'] for r in rows),
                'all_canonical_ids_equal':all(r['canonical_ids_equal'] for r in rows) if rows[0]['canonical_ids_equal'] is not None else None})
    save();print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
