#!/usr/bin/env python3
"""Validate the selected IQ target on the local model, serially on one GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ENGINE = Path(__file__).resolve().parents[1]
BASE = ENGINE/'bench/dflash-port'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--draft-model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, draft = str(args.model.resolve()), str(args.draft_model.resolve())
    prompts = {case: ENGINE/f'bench/optimization-analysis/llama-mtp4-{case}/prompt.ids'
               for case in ('short', 'long')}
    flags = dict(MK_FFN_IQ_FUSED='1', MK_IQ_INLINE='1', MK_FFN_IQ_MMA='0',
                 MK_FFN_DOWN_GLOBAL='0', MK_IQ_PAIRED='1', MK_IQ2_PAIRED='1',
                 MK_DFLASH_MMQ_TYPED='1', MK_DFLASH_MMQ_ROWS='4',
                 MK_DFLASH_MMQ_HEAD_ROWS='4', MK_DFLASH_BUDGET='1')
    env = os.environ.copy(); env.update(flags)
    for key in ('MK_PROFILE', 'MK_PF_PROFILE', 'MK_CHECK_FINITE', 'MK_SCORE_PROBE_POSITION', 'MK_SCORE_PROBE_FILE', 'MK_DFLASH_TRACE'):
        env.pop(key, None)
    env['LD_LIBRARY_PATH'] = str(ENGINE.parents[1]/'llama.cpp/build/bin')+os.pathsep+env.get('LD_LIBRARY_PATH', '')
    names = ('qwen38_mk.cu', 'native_quant.cuh', 'native_iq_mma.cuh', 'qwen38_mk',
             'dflash_native.cu', 'dflash_native.h', 'dflash_main.cpp', 'capi.cpp',
             'server/mk_server.py', 'Makefile', 'qwen38_dflash', 'libqwen38mk.so',
             'external_verify_test', 'tests/external_verify.cpp', 'tests/gpu_server_smoke.py')
    report = {'flags': flags, 'sha256': {n: hashlib.sha256((ENGINE/n).read_bytes()).hexdigest() for n in names},
              'tests': [], 'limits': 'Small task/HTTP suite and bounded memcheck; not broad perplexity or fully populated 32K.'}

    def save():
        (args.output/'status.json').write_text(json.dumps(report, indent=2)+'\n')

    def run(name, command, timeout=600):
        report['active'] = name; save(); started = time.monotonic()
        with (args.output/f'{name}.out').open('w') as stdout, (args.output/f'{name}.log').open('w') as stderr:
            done = subprocess.run(command, env=env, cwd=ENGINE, stdout=stdout, stderr=stderr, timeout=timeout)
        report.pop('active', None)
        report['tests'].append({'name': name, 'command': command, 'exit_code': done.returncode,
                                'elapsed_s': time.monotonic()-started})
        save(); print(name, 'exit', done.returncode, flush=True)
        if done.returncode:
            raise RuntimeError(f'{name} failed; inspect {args.output/name}.log')
        return (args.output/f'{name}.out').read_text(), (args.output/f'{name}.log').read_text()

    run('external-verify', [str(ENGINE/'external_verify_test'), model,
                           str(prompts['short']), str(prompts['long'])])
    run('quality', [sys.executable, str(ENGINE/'bench/dflash_port_quality.py'), '--model', model,
        '--draft-model', draft, '--depth', '7', '--backend', 'native',
        '--prior-native', str(BASE/'quality-final/results.json'),
        '--output', str(args.output.parent/'quality-final')])
    reference = json.loads((BASE/'regular-mk-regression/results.json').read_text())
    for case in ('short', 'long'):
        stdout, _ = run(f'mtp-{case}', [str(ENGINE/'qwen38_mk'), '-m', model,
            '--prompt-ids', str(prompts[case]), '-n', '128', '-d', '4', '-c', '4096',
            '--kv', 'q8', '--prefill-chunk', 'auto', '--ids', '--quiet'])
        match = re.search(r'^IDS:\s*(.*)$', stdout, re.MULTILINE)
        ids = list(map(int, match.group(1).split())) if match else None
        expected = next(r['ids'] for r in reference['runs'] if r['case'] == case and r['engine'] == 'candidate')
        report['tests'][-1]['prior_mtp_ids_equal'] = ids == expected; save()
        if ids != expected:
            raise RuntimeError(f'MTP {case} changed its previous fixture IDs')
    run('http-32k', [sys.executable, str(ENGINE/'tests/gpu_server_smoke.py'), '--model', model,
        '--draft-model', draft, '--draft', '7', '--ctx', '32768', '--port', '18094',
        '--output', str(args.output.parent/'http-final-32k')])
    if args.sanitize:
        for name, kernel, launches in (('memcheck-wide', 'mk_cycle_wide', '2'),
                                      ('memcheck-draft-mmq', 'mmq_typed', '64')):
            stdout, stderr = run(name, ['/usr/local/cuda/bin/compute-sanitizer',
                '--tool', 'memcheck', '--error-exitcode', '99', '--kernel-name', 'kns='+kernel,
                '--launch-count', launches, str(ENGINE/'qwen38_dflash'), '-m', model, '--draft-model', draft,
                '--backend', 'native', '--prompt-ids', str(prompts['short']), '-c', '4096',
                '-d', '7', '-n', '16', '--reps', '1', '--temp', '0', '--seed', '123',
                '--prefill-chunk', 'auto'], timeout=1200)
            clean = 'ERROR SUMMARY: 0 errors' in stdout+stderr
            report['tests'][-1]['clean_summary'] = clean; save()
            if not clean:
                raise RuntimeError(f'{name} did not report a clean summary')
    print(json.dumps(report['tests'], indent=2), flush=True)


if __name__ == '__main__':
    main()
