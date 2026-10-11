#!/usr/bin/env python3
"""Run serial CUDA memory, scoring, and sanitizer checks on the local model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
ENGINE=ROOT.parent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=[str(ENGINE/'qwen38_mk'),'-m',str(args.model.resolve()),'--kv','q8','-d','4',
          '--draft-vocab',str(ENGINE/'draft_vocab_32k.bin'),'--ids','--quiet']
    short=ROOT/'optimization-analysis/validated-prompt.ids'
    long=ROOT/'optimization-analysis/llama-mtp4-long/prompt.ids'
    reference=ROOT/'optimization-analysis/llama-mtp4-long/reference.ids'
    rows=[]
    def run(name,cmd,wanted=0,timeout=600):
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
        (args.output/f'{name}.log').write_text(result.stderr)
        (args.output/f'{name}.out').write_text(result.stdout)
        row={'name':name,'command':cmd,'returncode':result.returncode,'expected_returncode':wanted}
        rows.append(row)
        (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(name,'exit',result.returncode,flush=True)
        if result.returncode!=wanted:raise RuntimeError(f'{name}: unexpected exit; see log')
        return result
    for ctx in (4096,32768,65536):
        run(f'context-{ctx}',base+['-c',str(ctx),'--prefill-chunk','auto','--prompt-ids',str(short),'-n','16'])
    rejected=run('context-rejected',base+['-c','262144','--prefill-chunk','auto','--prompt-ids',str(short),'-n','1'],1)
    if 'does not fit' not in rejected.stderr:raise RuntimeError('context was rejected for an unexpected reason')
    run('score-long-probe',base+['-c','4096','--prefill-chunk','512','--prompt-ids',str(long),
        '--score',str(reference),'--score-probe','41','--score-probe-file',str(args.output/'probe-logits.f32')])
    sanitizer=shutil.which('compute-sanitizer')
    if not sanitizer:
        path=Path('/usr/local/cuda/bin/compute-sanitizer')
        if path.is_file():sanitizer=str(path)
    if sanitizer:
        result=run('memcheck',[sanitizer,'--tool','memcheck','--error-exitcode','99',
            '--kernel-name','kns=mk_cycle','--launch-count','4']+base+
            ['-c','4096','--prefill-chunk','512','--prompt-ids',str(short),'-n','16'],timeout=1200)
        if 'ERROR SUMMARY: 0 errors' not in result.stdout+result.stderr:
            raise RuntimeError('sanitizer did not report a clean summary')
    else:
        rows.append({'name':'memcheck','skipped':'compute-sanitizer unavailable'})
    (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    hashes={p:hashlib.sha256((ENGINE/p).read_bytes()).hexdigest() for p in
        ('qwen38_mk.cu','native_quant.cuh','native_prefill.cuh','prefill.cuh','qwen38_mk','libqwen38mk.so')}
    (args.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')


if __name__=='__main__':main()
