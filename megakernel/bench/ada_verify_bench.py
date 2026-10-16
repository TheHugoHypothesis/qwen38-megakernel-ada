#!/usr/bin/env python3
"""Compare Ada T4 fusion/verification policies using model-specific prompt IDs.

All variants keep the trained DFlash block unchanged unless explicitly labelled
depth3. Alternating process order limits ordering bias. Decode excludes prefill
and first token; different output IDs are reported, not treated as accuracy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {'full7': (7, '0', 'full'), 'depth3': (3, '0', 'full'),
            'fused3': (3, '1', 'full'), 'verify3': (7, '1', '3'),
            'adaptive7': (7, '1', 'auto'), 'preserved7': (7, '0', 'full')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--draft-model', type=Path, required=True)
    ap.add_argument('--fixture-report', type=Path, required=True, help='model_variant_gpu.py results for this same target')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--binary', type=Path, default=ROOT / 'qwen38_dflash')
    ap.add_argument('--baseline-binary', type=Path)
    ap.add_argument('--variants', nargs='+', choices=tuple(VARIANTS), default=['full7', 'verify3', 'adaptive7'])
    ap.add_argument('--cases', nargs='+', default=['short_code', 'long_code', 'numbered_lines'])
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--kv', choices=['q8', 'q4'], default='q8')
    args = ap.parse_args()
    if args.reps < 1 or args.tokens < 2:
        ap.error('positive repetitions and at least two output tokens are required')
    source = json.loads(args.fixture_report.read_text())
    if Path(source['model']).name != args.model.name:
        ap.error('fixture report belongs to another target/tokenizer; generate matching prompt IDs first')
    if 'preserved7' in args.variants and not args.baseline_binary:
        ap.error('--baseline-binary is needed for preserved7')
    args.output.mkdir(parents=True, exist_ok=True)
    prompts = []
    for case in args.cases:
        row = next(r for r in source['runs'] if r['mode'] == 'mk_scalar' and r['case'] == case)
        path = args.output / (case + '.ids')
        path.write_text(' '.join(map(str, row['prompt_ids'])) + '\n')
        prompts.append(path.resolve())
    report = {'model': str(args.model.resolve()), 'draft': str(args.draft_model.resolve()),
              'cases': args.cases, 'kv': args.kv, 'ctx': args.ctx, 'temperature': 0, 'tokens_limit': args.tokens,
              'repetitions': args.reps, 'clocks_changed': False, 'runs': [], 'configurations': [],
              'limits': 'Sequential process A/B. Native timing excludes load/prefill/first token. Compare IDs alongside throughput; this is not task accuracy.'}

    def save():
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

    for rep in range(args.reps):
        order = args.variants if rep % 2 == 0 else list(reversed(args.variants))
        for variant in order:
            depth, small, policy = VARIANTS[variant]
            binary = args.baseline_binary if variant == 'preserved7' else args.binary
            cmd = [str(binary.resolve()), '-m', str(args.model.resolve()), '--draft-model', str(args.draft_model.resolve()),
                   '-d', str(depth), '-c', str(args.ctx), '--kv', args.kv, '--prefill-chunk', '128',
                   '-n', str(args.tokens), '--reps', '1', '--temp', '0', '--seed', '123']
            for path in prompts:
                cmd += ['--prompt-ids', str(path)]
            env = os.environ.copy()
            for key in ('MK_PROFILE', 'MK_PF_PROFILE', 'MK_CHECK_FINITE', 'MK_SCORE_PROBE_POSITION', 'MK_SCORE_PROBE_FILE', 'MK_DFLASH_TRACE'):
                env.pop(key, None)
            flags = {'MK_FFN_IQ_SMALL': small, 'MK_DFLASH_VERIFY': policy, 'MK_FFN_IQ_FUSED': '1',
                     'MK_IQ_INLINE': '1', 'MK_IQ_PAIRED': '1', 'MK_FFN_IQ_MMA': '0', 'MK_FFN_DOWN_GLOBAL': '0',
                     'MK_DFLASH_GRAPH': '1', 'MK_DFLASH_MATH': 'q8', 'MK_DFLASH_MMQ_TYPED': '1',
                     'MK_DFLASH_MMQ_ROWS': '4', 'MK_DFLASH_MMQ_HEAD_ROWS': '4', 'MK_DFLASH_BUDGET': '1'}
            env.update(flags)
            # Also works for a preserved binary in an evidence subdirectory.
            libdir = ROOT.parents[1] / 'llama.cpp/build/bin'
            env['LD_LIBRARY_PATH'] = str(libdir.resolve()) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
            stem = args.output / f'r{rep}-{variant}'
            report['configurations'].append({'rep': rep, 'variant': variant, 'command': cmd, 'environment': flags,
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()})
            save()
            with stem.with_suffix('.out').open('w') as stdout, stem.with_suffix('.log').open('w') as stderr:
                subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, check=True, timeout=1200)
            ids = None
            for line in stem.with_suffix('.out').read_text().splitlines():
                if line.startswith('IDS:'):
                    ids = list(map(int, line[4:].split()))
                elif line.startswith('RESULT '):
                    row = json.loads(line[7:])
                    row.update(rep=rep, variant=variant, case=args.cases[row['case_index']], ids=ids)
                    report['runs'].append(row)
                elif line.startswith('VERIFY_POLICY '):
                    report['runs'][-1]['verify_policy'] = json.loads(line[14:])
            save()
            print('completed', rep, variant, flush=True)
    report['summary'] = []
    for case in args.cases:
        reference = next((r['ids'] for r in report['runs'] if r['case'] == case and r['variant'] in ('full7', 'preserved7')), None)
        for variant in args.variants:
            rows = [r for r in report['runs'] if r['case'] == case and r['variant'] == variant]
            report['summary'].append({'case': case, 'variant': variant,
                'mean_tps': statistics.mean(r['tps'] for r in rows), 'min_tps': min(r['tps'] for r in rows),
                'max_tps': max(r['tps'] for r in rows), 'stddev_tps': statistics.stdev(r['tps'] for r in rows) if len(rows)>1 else None,
                'repeatable_ids': len({tuple(r['ids']) for r in rows}) == 1 if len(rows)>1 else None,
                'full7_ids_equal': all(r['ids'] == reference for r in rows) if reference is not None else None,
                'cycles': [r['cycles'] for r in rows], 'acceptance': sum(r['accepted'] for r in rows)/max(1,sum(r['drafted'] for r in rows))})
    save()
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
