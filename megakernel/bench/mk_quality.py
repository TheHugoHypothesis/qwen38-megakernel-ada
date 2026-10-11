#!/usr/bin/env python3
"""Validate the CUDA C API using canonical token IDs and a reusable engine."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
ENGINE=ROOT.parent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--suite',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--chunk',default='auto')
    ap.add_argument('--draft',type=int,default=4)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ['MK_PREFILL_CHUNK']=args.chunk
    spec=importlib.util.spec_from_file_location('ada_mk_server',ENGINE/'server/mk_server.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    engine=module.Engine(str(ENGINE/'libqwen38mk.so'),str(args.model.resolve()),4096,args.draft,
                         str(ENGINE/'draft_vocab_32k.bin'),1)
    engine.L.mk_free.argtypes=[ctypes.c_void_p];engine.L.mk_free.restype=None
    questions={x['name']:x for x in json.loads(args.suite.read_text())}
    refs=[x for x in json.loads(args.reference.read_text())['runs'] if x['rep']==0]
    report={'draft':args.draft,'chunk':args.chunk,'runs':[]}
    try:
        engine.set_sampling(0,.95,20,0,123)
        for row in refs:
            case=row['case']; ids=row['prompt_ids']; expected=row['response']['tokens']
            expected=[x for x in expected if not engine.is_eog(x)]
            engine.reset(); start=time.monotonic()
            first=engine.prefill(ids); pf_end=time.monotonic()
            generated=[]; batch=[first]; stopped=False
            while len(generated)<128 and not stopped:
                for token in batch:
                    if engine.is_eog(token):stopped=True;break
                    generated.append(token)
                    if len(generated)==128:break
                if not stopped and len(generated)<128:batch=engine.step()
            end=time.monotonic()
            text=b''.join(engine.piece(x) for x in generated).decode('utf-8',errors='replace')
            prefix=next((i for i,(a,b) in enumerate(zip(generated,expected)) if a!=b),min(len(generated),len(expected)))
            result={'case':case,'ids':generated,'content':text,'canonical_prefix':prefix,
                    'canonical_ids_equal':generated==expected,'prefill_seconds':pf_end-start,
                    'decode_seconds':end-pf_end,'request_seconds':end-start}
            question=questions[case]
            if 'answer' in question:
                try:answer=json.loads(text.strip())['answer']
                except (ValueError,KeyError,TypeError):answer=None
                result.update(answer=answer,expected_answer=question['answer'],task_correct=answer==question['answer'])
            report['runs'].append(result)
            (args.output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
            print(case,'equal',result['canonical_ids_equal'],'prefix',prefix,'task',result.get('task_correct'),flush=True)
    finally:engine.L.mk_free(engine.h)
    report['summary']={'equal_generations':sum(x['canonical_ids_equal'] for x in report['runs']),
        'generations':len(report['runs']),'correct_tasks':sum(x.get('task_correct',False) for x in report['runs']),
        'tasks':sum('task_correct' in x for x in report['runs'])}
    (args.output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report['summary']))


if __name__=='__main__':main()
