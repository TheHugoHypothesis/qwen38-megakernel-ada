#!/usr/bin/env python3
"""Serial integration checks for explicit Q4 target KV; leaves other GPU processes alone."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--draft-model', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--ctx', type=int, default=65536)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, draft = str(args.model.resolve()), str(args.draft_model.resolve())
    fixtures = [str(ROOT/f'bench/optimization-analysis/llama-mtp4-{case}/prompt.ids') for case in ('short', 'long')]
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(ROOT.parents[1]/'llama.cpp/build/bin')+os.pathsep+env.get('LD_LIBRARY_PATH', '')
    env['MK_PREFILL_CHUNK'] = '128'
    for key in ('MK_PROFILE', 'MK_PF_PROFILE', 'MK_SCORE_PROBE_POSITION', 'MK_SCORE_PROBE_FILE', 'MK_DFLASH_TRACE'):
        env.pop(key, None)
    report = {'checks': [], 'ctx': args.ctx, 'target_kv': 'q4',
              'sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                         ('libqwen38mk.so', 'qwen38_dflash', 'qwen38_mk.cu', 'prefill.cuh', 'kv_quant.cuh')},
              'limits': 'Small task suite and bounded HTTP prompts; capacity is not a fully populated-context quality evaluation.'}

    def save():
        (args.output/'status.json').write_text(json.dumps(report, indent=2)+'\n')

    def run(name, cmd, timeout=600):
        report['active'] = name
        save()
        started = time.monotonic()
        with (args.output/f'{name}.out').open('w') as stdout, (args.output/f'{name}.log').open('w') as stderr:
            done = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=timeout)
        report.pop('active', None)
        report['checks'].append({'name': name, 'command': cmd, 'exit_code': done.returncode,
                                 'elapsed_s': time.monotonic()-started})
        save()
        print(f'{name}: exit {done.returncode}', flush=True)
        if done.returncode:
            raise RuntimeError(f'{name} failed; inspect {args.output/name}.log')

    for mode in (2, 1):
        run(f'external-kv{mode}', [str(ROOT/'external_verify_test'), model, *fixtures, str(mode)])
    quality = ROOT/'bench/dflash_port_quality.py'
    prior = args.output.parent/'quality-baseline-q8/results.json'
    for mode in ('q8', 'q4'):
        cmd = [sys.executable, str(quality), '--model', model, '--draft-model', draft,
               '--kv', mode, '--output', str(args.output.parent/f'quality-{mode}')]
        if prior.is_file():
            cmd += ['--prior-native', str(prior)]
        run(f'quality-{mode}', cmd)
    run('http-q4', [sys.executable, str(ROOT/'tests/gpu_server_smoke.py'), '--model', model,
                   '--draft-model', draft, '--draft', '7', '--kv', 'q4', '--ctx', str(args.ctx),
                   '--prefill-chunk', '128', '--long-records', '1200', '--port', '18097', '--output', str(args.output.parent/'http-q4')])
    run('memcheck-q4-wide', ['/usr/local/cuda/bin/compute-sanitizer', '--tool', 'memcheck',
                            '--error-exitcode', '9', '--kernel-name', 'kns=mk_cycle_wide', '--launch-count', '2',
                            str(ROOT/'qwen38_dflash'), '-m', model, '--draft-model', draft,
                            '--prompt-ids', fixtures[0], '--kv', 'q4', '-c', '4096', '--prefill-chunk', '128',
                            '-d', '7', '-n', '16', '--reps', '1'])
    out = (args.output/'memcheck-q4-wide.out').read_text()+(args.output/'memcheck-q4-wide.log').read_text()
    if 'ERROR SUMMARY: 0 errors' not in out:
        raise RuntimeError('missing clean memcheck summary')


if __name__ == '__main__':
    main()
