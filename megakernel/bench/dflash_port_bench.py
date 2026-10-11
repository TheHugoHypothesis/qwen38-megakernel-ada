#!/usr/bin/env python3
"""Measure native or reference DFlash serially, preserving IDs and GPU samples."""
import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--draft-model', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--backend', choices=('native','ggml'), default='native')
    ap.add_argument('--binary', type=Path, default=ENGINE/'qwen38_dflash')
    ap.add_argument('--cases', nargs='+', choices=('short','long','sampling'), default=['short','long','sampling'])
    ap.add_argument('--depth', type=int, default=7)
    ap.add_argument('--kv', choices=('q8','q4','f16','auto'), default='q8')
    ap.add_argument('--prefill-chunk', default='auto')
    ap.add_argument('--ctx', type=int, default=4096)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('ada_launcher', ENGINE/'serve_ada.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    compatibility = launcher.check_draft(args.model, args.draft_model)
    report = {'compatibility': compatibility, 'target_kv': args.kv,
              'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(), 'runs': [], 'configurations': [],
              'limits': 'CLI decode excludes load, prefill and first token. NVML peak is sampled, not exact.'}
    def save():
        (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    env = os.environ.copy()
    library_dir = ENGINE.parents[1]/'llama.cpp/build/bin'
    env['LD_LIBRARY_PATH'] = str(library_dir)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    for key in ('MK_PROFILE', 'MK_PF_PROFILE', 'MK_CHECK_FINITE', 'MK_SCORE_PROBE_POSITION', 'MK_SCORE_PROBE_FILE'):
        env.pop(key, None)
    report['environment'] = {key: env[key] for key in ('MK_FFN_IQ_FUSED','MK_IQ_INLINE','MK_FFN_IQ_MMA','MK_FFN_DOWN_GLOBAL','MK_IQ_PAIRED','MK_IQ2_PAIRED','MK_DFLASH_MATH','MK_DFLASH_GRAPH','MK_DFLASH_MMQ_TYPED','MK_DFLASH_MMQ_ROWS','MK_DFLASH_MMQ_HEAD_ROWS','MK_DFLASH_BUDGET') if key in env}
    for case, fixture, temp in [('short', 'short', 0), ('long', 'long', 0), ('sampling', 'short', .7)]:
        if case not in args.cases:
            continue
        ids = ROOT/f'optimization-analysis/llama-mtp4-{fixture}/prompt.ids'
        cmd = [str(args.binary.resolve()), '-m', str(args.model.resolve()), '--draft-model',
               str(args.draft_model.resolve()), '--prompt-ids', str(ids), '-d', str(args.depth), '-c', str(args.ctx),
               '--backend', args.backend, '--prefill-chunk', args.prefill_chunk,
               '-n', str(args.tokens), '--reps', str(args.reps), '--temp', str(temp), '--seed', '123']
        if args.kv != 'q8': cmd += ['--kv', args.kv]
        samples = []
        with (args.output/f'{case}.out').open('w') as stdout, (args.output/f'{case}.log').open('w') as stderr:
            proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
            deadline = time.monotonic()+600
            try:
                while proc.poll() is None:
                    sample = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,clocks.current.graphics,clocks.current.memory',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                    if sample.returncode == 0:
                        samples.append([int(x.strip()) for x in sample.stdout.splitlines()[0].split(',')])
                    if time.monotonic() > deadline:
                        raise TimeoutError(case)
                    time.sleep(.15)
            finally:
                if proc.poll() is None:
                    proc.kill(); proc.wait()
            if proc.returncode:
                raise RuntimeError(f'{case}: exit {proc.returncode}; see log')
        generation = None
        for line in (args.output/f'{case}.out').read_text().splitlines():
            if line.startswith('IDS:'):
                generation = list(map(int, line[4:].split()))
            elif line.startswith('RESULT '):
                row = json.loads(line[7:])
                row.update(case=case, ids=generation)
                report['runs'].append(row)
        report['configurations'].append({'case': case, 'command': cmd, 'gpu_samples': samples,
            'sampled_peak_mib': max(x[0] for x in samples) if samples else None})
        report.setdefault('summary', []).append({'case': case,
            'tps': statistics.mean(r['tps'] for r in report['runs'] if r['case']==case),
            'ids_repeatable': len({tuple(r['ids']) for r in report['runs'] if r['case']==case})==1})
        save()
        print(json.dumps(report['summary'][-1]), flush=True)


if __name__ == '__main__':
    main()
