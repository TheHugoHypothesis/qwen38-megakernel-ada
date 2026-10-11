#!/usr/bin/env python3
"""Interleave preserved and candidate CLI runs with identical prompt IDs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from statistics import mean

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--drafts', default='0,4')
    ap.add_argument('--cases', default='short,long')
    ap.add_argument('--chunk', default='512')
    ap.add_argument('--baseline-chunk', default='512')
    ap.add_argument('--temperature', type=float, default=0)
    ap.add_argument('--candidate', type=Path, default=ENGINE/'qwen38_mk')
    ap.add_argument('--baseline', type=Path, default=ROOT/'optimization-implementation/baseline/qwen38_mk')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(ENGINE.parents[1]/'llama.cpp/build/bin') + ':' + env.get('LD_LIBRARY_PATH', '')
    for key in ('MK_PROFILE', 'MK_PF_PROFILE', 'MK_PREFILL', 'MK_PREFILL_CHUNK',
                'MK_NATIVE_GEMM', 'MK_CHECK_FINITE', 'MK_SCORE_PROBE_POSITION', 'MK_SCORE_PROBE_FILE'):
        env.pop(key, None)
    report = {'runs': [], 'hashes': {}, 'temperature': args.temperature,
              'limits': 'CLI loads separately; measured decode excludes load/prefill.'}
    for name, path in (('baseline', baseline), ('candidate', args.candidate)):
        report['hashes'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for case in args.cases.split(','):
        fixture = 'validated' if case == 'short' else 'long'
        old = json.loads((ROOT/f'results-ada-{fixture}/results.json').read_text())
        canonical = json.loads((ROOT/f'optimization-analysis/llama-mtp4-{case}/results.json').read_text())
        prompt = args.output/f'{case}-prompt.ids'
        prompt.write_text(' '.join(map(str, old['prompt_ids'])))
        for draft in map(int, args.drafts.split(',')):
            baseline_ids = None
            for rep in range(args.reps):
                order = [('baseline', baseline), ('candidate', args.candidate)]
                if rep % 2:
                    order.reverse()
                for name, binary in order:
                    cmd = [str(binary.resolve()), '-m', str(args.model.resolve()), '-c', '4096',
                           '--kv', 'q8', '-d', str(draft), '-n', str(args.tokens),
                           '--prefill-chunk', args.baseline_chunk if name == 'baseline' else args.chunk,
                           '--draft-vocab', str(ENGINE/'draft_vocab_32k.bin'),
                           '--prompt-ids', str(prompt.resolve()), '--ids', '--quiet']
                    if args.temperature > 0:
                        cmd += ['--temp', str(args.temperature), '--top-k', '20', '--top-p', '.95',
                                '--min-p', '0', '--seed', '123']
                    done = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
                    prefix = f'{case}-{name}-d{draft}-r{rep}'
                    (args.output/f'{prefix}.log').write_text(done.stderr)
                    (args.output/f'{prefix}.out').write_text(done.stdout)
                    if done.returncode:
                        raise RuntimeError(f'{prefix}: exit {done.returncode}; see log')
                    ids = list(map(int, re.search(r'^IDS: (.*)$', done.stdout, re.M)[1].split()))
                    if name == 'baseline':
                        baseline_ids = ids
                    speed = float(re.search(r'\[mk\] decode .* = ([\d.]+) tok/s', done.stderr)[1])
                    pf = re.search(r'\[mk\] prompt .*? in ([\d.]+) s', done.stderr)
                    row = {'case': case, 'engine': name, 'draft': draft, 'rep': rep,
                           'decode_tps': speed, 'ids': ids, 'command': cmd,
                           'canonical_prefix': next((i for i, (a,b) in enumerate(zip(ids,canonical['llama'][0]['tokens'])) if a!=b), min(len(ids),len(canonical['llama'][0]['tokens']))) if args.temperature == 0 else None,
                           'prefill_seconds': float(pf[1]) if pf else None}
                    report['runs'].append(row)
                    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
                    print(prefix, speed, 'canonical prefix', row['canonical_prefix'], flush=True)
            rows = [x for x in report['runs'] if x['case']==case and x['draft']==draft]
            same = all(x['ids']==baseline_ids for x in rows)
            a = mean(x['decode_tps'] for x in rows if x['engine']=='baseline')
            b = mean(x['decode_tps'] for x in rows if x['engine']=='candidate')
            report.setdefault('summary', []).append({'case':case,'draft':draft,'baseline_tps':a,
                'candidate_tps':b,'gain_percent':100*(b/a-1),'all_generation_ids_equal':same})
            (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
