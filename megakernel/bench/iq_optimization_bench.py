#!/usr/bin/env python3
"""Compare IQ target variants serially against recorded DFlash fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ENGINE = Path(__file__).resolve().parents[1]
FLAGS = {
    'original': (0, 0, 0, 0, 0),
    'fused': (1, 0, 0, 0, 1),
    'inlined': (1, 1, 0, 0, 1),
    'unpaired': (1, 1, 0, 0, 0),
    'mma': (1, 0, 1, 0, 1),
    'combined': (1, 1, 1, 0, 1),
    'global_only': (1, 0, 0, 1, 1),
    'global': (1, 1, 0, 1, 1),
    'classic_draft': (1, 1, 0, 0, 1),
    'typed2': (1, 1, 0, 0, 1),
    'head4': (1, 1, 0, 0, 1),
    'typed4': (1, 1, 0, 0, 1),
    'budget_off': (1, 1, 0, 0, 1),
    'budget_on': (1, 1, 0, 0, 1),
}
FLAG_NAMES = ('MK_FFN_IQ_FUSED', 'MK_IQ_INLINE', 'MK_FFN_IQ_MMA', 'MK_FFN_DOWN_GLOBAL', 'MK_IQ_PAIRED')
DRAFT_NAMES = ('MK_DFLASH_MMQ_TYPED', 'MK_DFLASH_MMQ_ROWS', 'MK_DFLASH_MMQ_HEAD_ROWS')
DRAFT_FLAGS = {'inlined': ('1', '4', '4'), 'classic_draft': ('0', '2', '2'), 'typed2': ('1', '2', '2'),
               'head4': ('1', '2', '4'), 'typed4': ('1', '4', '4'),
               'budget_off': ('1', '4', '4'), 'budget_on': ('1', '4', '4')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ENGINE/'bench/dflash-port/native-final/results.json')
    parser.add_argument('--binary', type=Path, default=ENGINE/'qwen38_dflash')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=('short', 'long', 'sampling'), default=['short', 'long', 'sampling'])
    parser.add_argument('--variants', nargs='+', choices=tuple(FLAGS), default=['inlined'])
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.reps < 1:
        parser.error('--reps must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    canonical = json.loads(args.reference.read_text())
    report = {'binary': str(args.binary.resolve()), 'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              'reference': str(args.reference.resolve()), 'profiled': args.profile, 'configurations': [], 'runs': [],
              'limits': 'CLI decode excludes load, prefill and first token. NVML peak is sampled. Clocks are not set.'}

    def save():
        (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')

    save()
    for case_index, case in enumerate(args.cases):
        config = next(c for c in canonical['configurations'] if c['case'] == case)
        reference = next(r['ids'] for r in canonical['runs'] if r['case'] == case)
        variants = args.variants if case_index % 2 == 0 else list(reversed(args.variants))
        for variant in variants:
            command = config['command'].copy()
            command[0] = str(args.binary.resolve())
            command[command.index('--reps')+1] = str(args.reps)
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = str(ENGINE.parents[1]/'llama.cpp/build/bin')+os.pathsep+env.get('LD_LIBRARY_PATH', '')
            for key in ('MK_PROFILE', 'MK_PF_PROFILE', 'MK_CHECK_FINITE', 'MK_SCORE_PROBE_POSITION', 'MK_SCORE_PROBE_FILE', 'MK_DFLASH_TRACE'):
                env.pop(key, None)
            flags = dict(zip(FLAG_NAMES, map(str, FLAGS[variant])))
            flags['MK_IQ2_PAIRED'] = flags['MK_IQ_PAIRED'] # reproduce earlier snapshots
            if variant in DRAFT_FLAGS:
                flags.update(zip(DRAFT_NAMES, DRAFT_FLAGS[variant]))
            if variant in ('inlined', 'budget_off', 'budget_on'):
                flags['MK_DFLASH_BUDGET'] = '0' if variant == 'budget_off' else '1'
            env.update(flags)
            if args.profile:
                env['MK_PROFILE'] = '1'
            samples = []
            stem = args.output/f'{case}-{variant}'
            started = time.monotonic()
            with stem.with_suffix('.out').open('w') as stdout, stem.with_suffix('.log').open('w') as stderr:
                process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
                try:
                    while process.poll() is None:
                        sample = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw',
                                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                        if sample.returncode == 0:
                            samples.append(sample.stdout.strip())
                        if time.monotonic()-started > 1200:
                            raise TimeoutError(f'{case}/{variant}')
                        time.sleep(.15)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            report['configurations'].append({'case': case, 'variant': variant, 'flags': flags, 'command': command,
                                             'native_mmq_environment': {k: env.get(k, 'C++ default') for k in (*DRAFT_NAMES, 'MK_DFLASH_BUDGET')},
                                             'gpu_samples': samples, 'exit_code': process.returncode})
            save()
            if process.returncode:
                raise RuntimeError(f'{case}/{variant}: exit {process.returncode}; see {stem}.log')
            ids = None
            rows = []
            for line in stem.with_suffix('.out').read_text().splitlines():
                if line.startswith('IDS:'):
                    ids = list(map(int, line[4:].split()))
                elif line.startswith('RESULT '):
                    row = json.loads(line[7:])
                    row.update(case=case, variant=variant, ids=ids, canonical_ids_equal=ids == reference)
                    rows.append(row)
            if len(rows) != args.reps:
                raise RuntimeError(f'{case}/{variant}: expected {args.reps} RESULT rows, got {len(rows)}')
            report['runs'].extend(rows)
            summary = {'case': case, 'variant': variant, 'tps': statistics.mean(r['tps'] for r in rows),
                       'min_tps': min(r['tps'] for r in rows), 'max_tps': max(r['tps'] for r in rows),
                       'canonical_ids_equal': all(r['canonical_ids_equal'] for r in rows),
                       'cycles': [r['cycles'] for r in rows], 'accepted': [r['accepted'] for r in rows]}
            report.setdefault('summary', []).append(summary)
            save()
            print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
