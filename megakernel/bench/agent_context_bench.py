#!/usr/bin/env python3
"""Build real-tokenized agent/tool fixtures and profile populated context growth.

Fixtures are synthetic conversations containing frozen repository source and a
tool result, rendered by the target GGUF template. They are not an OpenCode trace
or a scored coding benchmark. Profile timings and production throughput are run
separately. GPU clocks/power settings are never changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('server/mk_server.py','capi.cpp','qwen38_mk.cu','native_quant.cuh',
         'native_iq_mma.cuh','prefill.cuh','prefill_gemm.cuh','tests/external_verify.cpp')
TOOLS=[{'type':'function','function':{'name':'read_file','description':'Read a repository file.',
        'parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path']}}},
       {'type':'function','function':{'name':'run_tests','description':'Run named tests.',
        'parameters':{'type':'object','properties':{'command':{'type':'string'}},'required':['command']}}}]


def classify_output(text):
    if '<tool_call>' in text and '</tool_call>' in text:
        return {'output_kind':'tool_call','tool_names':re.findall(r'<function=([^>]+)>',text)}
    return {'output_kind':'text_or_incomplete_reasoning','tool_names':[]}


def prepare(args):
    sys.path.insert(0,str(args.llama_root.resolve()/'gguf-py'))
    sys.path.insert(0,str(ROOT/'server'))
    import gguf
    from mk_server import make_template,normalize_messages
    r=gguf.GGUFReader(args.model)
    template=make_template(r.fields['tokenizer.chat_template'].contents())
    pieces=[]; hashes={}
    for name in SOURCES:
        payload=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(payload).hexdigest()
        text=payload.decode()
        # Source literals must not become role/EOG control tokens in the fixture.
        text=re.sub(r'<\|([^|]+)\|>',r'[special:\1]',text)
        pieces.append(f'FILE: megakernel/{name}\n{text}\nEND FILE\n')
    corpus='\n'.join(pieces)
    prompts=args.output/'prompts';prompts.mkdir(parents=True,exist_ok=True)
    (prompts/'corpus.txt').write_text(corpus)
    cases=[]
    for workload in args.workloads:
        for requested in args.lengths:
            chars=min(len(corpus),requested*3)
            stem=prompts/f'{workload}-{requested}'
            for attempt in range(4):
                messages=[{'role':'system','content':'You are a coding agent. Inspect source and test evidence, make focused changes, and validate them. Use only the provided tools and visible files.'},
                    {'role':'user','content':'Inspect this CUDA inference server for request validation and state consistency. Preserve its OpenAI-compatible API.'},
                    {'role':'assistant','content':'I will inspect the server and backend before proposing a focused patch.',
                     'tool_calls':[{'id':'call_fixture_1','type':'function','function':{'name':'read_file','arguments':'{"path":"megakernel/server/mk_server.py"}'}}]},
                    {'role':'tool','tool_call_id':'call_fixture_1','content':corpus[:chars]},
                    {'role':'user','content':('Return a focused unified diff improving input validation and a regression test. Keep your explanation brief. Start with the patch.' if workload=='patch' else
                      'Before editing, call read_file for megakernel/capi.cpp to inspect how backend errors propagate. Do not write the patch yet.')}]
                rendered=template.render(messages=normalize_messages(messages),tools=TOOLS,
                    add_generation_prompt=True,enable_thinking=workload=='tool',reasoning_effort='medium')
                stem.with_suffix('.txt').write_text(rendered)
                cmd=[str(args.tokenizer.resolve()),str(args.model.resolve()),str(stem.with_suffix('.txt').resolve()),str(stem.with_suffix('.ids').resolve())]
                with stem.with_suffix('.tokenizer.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
                ids=list(map(int,stem.with_suffix('.ids').read_text().split()))
                if abs(len(ids)-requested)<=requested*.03:break
                newchars=min(len(corpus),max(1,int(chars*requested/len(ids))))
                if chars==newchars or attempt==3:break
                chars=newchars
            row={'workload':workload,'requested_tokens':requested,'prompt_tokens':len(ids),
                 'prompt_ids':str(stem.with_suffix('.ids').resolve()),'prompt_text_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
                 'thinking':workload=='tool','corpus_characters':chars}
            cases.append(row);print('prepared',workload,requested,'actual',len(ids),flush=True)
    manifest={'model':str(args.model.resolve()),'sources_sha256':hashes,'cases':cases,
              'limits':__doc__,'corpus_sha256':hashlib.sha256(corpus.encode()).hexdigest()}
    (args.output/'fixtures.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--draft-model',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--llama-root',type=Path,default=ROOT.parents[1]/'llama.cpp')
    ap.add_argument('--tokenizer',type=Path,default=ROOT/'tools/tokenize_ids')
    ap.add_argument('--binary',type=Path,default=ROOT/'qwen38_dflash')
    ap.add_argument('--lengths',nargs='+',type=int,default=[8192,16384,32768])
    ap.add_argument('--workloads',nargs='+',choices=['patch','tool'],default=['patch','tool'])
    ap.add_argument('--kv',choices=['q8','q4'],default='q4');ap.add_argument('--ctx',type=int,default=40000)
    ap.add_argument('--tokens',type=int,default=256);ap.add_argument('--reps',type=int,default=1)
    ap.add_argument('--verify-width',choices=['full','3','auto'],default='full')
    ap.add_argument('--t4-fusion',choices=['0','1'],default='0')
    ap.add_argument('--profile',action='store_true');ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--reuse-fixtures',action='store_true');ap.add_argument('--label',default='production')
    a=ap.parse_args()
    if a.reps<1 or a.tokens<2 or any(n<512 for n in a.lengths):ap.error('invalid sizes or repetitions')
    a.output.mkdir(parents=True,exist_ok=True)
    fixtures=json.loads((a.output/'fixtures.json').read_text()) if a.reuse_fixtures else prepare(a)
    if fixtures['model']!=str(a.model.resolve()):ap.error('fixture manifest belongs to another target')
    if a.prepare_only:return
    if not a.draft_model:ap.error('--draft-model is required to run CUDA')
    run_dir=a.output/a.label;run_dir.mkdir(parents=True,exist_ok=True)
    report={'model':str(a.model.resolve()),'draft_model':str(a.draft_model.resolve()),
            'profile':a.profile,'kv':a.kv,'capacity':a.ctx,'clocks_changed':False,'runs':[],
            'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'limits':__doc__}
    for case in fixtures['cases']:
        if case['workload'] not in a.workloads or case['requested_tokens'] not in a.lengths:continue
        if case['prompt_tokens']+a.tokens+16>=a.ctx:ap.error('capacity too small for a populated fixture plus generation')
        name=f"{case['workload']}-{case['requested_tokens']}";stem=run_dir/name
        cmd=[str(a.binary.resolve()),'-m',str(a.model.resolve()),'--draft-model',str(a.draft_model.resolve()),
             '--prompt-ids',case['prompt_ids'],'-d','7','-c',str(a.ctx),'--kv',a.kv,'--prefill-chunk','128',
             '-n',str(a.tokens),'--reps',str(a.reps),'--temp','0','--seed','123','--no-warmup']
        env=os.environ.copy()
        for k in ('MK_PROFILE','MK_PF_PROFILE','MK_CHECK_FINITE','MK_SCORE_PROBE_POSITION','MK_SCORE_PROBE_FILE','MK_DFLASH_TRACE'):env.pop(k,None)
        env['MK_FFN_IQ_SMALL']=a.t4_fusion;env['MK_DFLASH_VERIFY']=a.verify_width
        env['LD_LIBRARY_PATH']=str(a.llama_root.resolve()/'build/bin')+os.pathsep+env.get('LD_LIBRARY_PATH','')
        if a.profile:env['MK_PROFILE']='1'
        print('running',a.label,name,'populated',case['prompt_tokens'],flush=True)
        with stem.with_suffix('.out').open('w') as out,stem.with_suffix('.log').open('w') as log:subprocess.run(cmd,env=env,stdout=out,stderr=log,check=True,timeout=1800)
        phases={}
        for line in stem.with_suffix('.log').read_text().splitlines():
            m=re.match(r'\s+(\w+)\s+total/launch\s+([\d.]+) ms',line)
            if m:phases[m[1]]=float(m[2])
        ids=None; text=None
        for line in stem.with_suffix('.out').read_text().splitlines():
            if line.startswith('IDS:'):ids=list(map(int,line[4:].split()))
            if line.startswith('TEXT_HEX:'):text=bytes.fromhex(line[9:]).decode(errors='replace')
            if line.startswith('RESULT '):
                row=json.loads(line[7:]);row.update(case=case,ids=ids,text=text,phases_ms=phases,command=cmd,
                    environment={k:v for k,v in env.items() if k.startswith('MK_')},
                    acceptance=row['accepted']/max(1,row['drafted']),tokens_per_cycle=(row['tokens']-1)/max(1,row['cycles']))
                row.update(classify_output(text or ''))
                report['runs'].append(row)
            if line.startswith('VERIFY_POLICY '):report['runs'][-1]['verify_policy']=json.loads(line[14:])
        memory=re.search(r'\[mk\] runtime VRAM: ([\d.]+) GiB engine allocations, ([\d.]+) GiB device used, ([\d.]+) GiB free',stem.with_suffix('.log').read_text())
        if memory:
            for row in report['runs']:
                if row['case']==case:row['runtime_memory_gib_at_prefill']=dict(zip(('engine','device_used','free'),map(float,memory.groups())))
        (run_dir/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('saved',run_dir/'results.json',flush=True)


if __name__=='__main__':main()
