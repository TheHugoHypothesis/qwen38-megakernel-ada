#!/usr/bin/env python3
"""Run the existing 20-question suite through one reusable MK/DFlash engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--draft-model', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--backend', choices=('native','ggml'), default='native')
    ap.add_argument('--depth', type=int, default=7)
    ap.add_argument('--kv', choices=('q8','q4','f16','auto'), default='q8')
    ap.add_argument('--prior-native', type=Path,
                    help='Compare token IDs and text with a previous native report.')
    ap.add_argument('--binary', type=Path, default=ENGINE/'qwen38_dflash')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ROOT/'optimization-implementation/quality-nospec/results.json').read_text())
    prior_mk = {r['case']: r for r in json.loads((ROOT/'optimization-implementation/quality-megakernel/results.json').read_text())['runs']}
    prior_native = ({r['case']: r for r in json.loads(args.prior_native.read_text())['runs']}
                    if args.prior_native else {})
    questions = {r['name']: r for r in json.loads((ROOT/'optimization-implementation/quality-suite.json').read_text())}
    rows = [r for r in reference['runs'] if r['rep']==0]
    cmd = [str(args.binary.resolve()), '-m', str(args.model.resolve()), '--draft-model',
           str(args.draft_model.resolve()), '-n', '128', '-d', str(args.depth), '--reps', '1',
           '--backend', args.backend, '--prefill-chunk', 'auto']
    if args.kv != 'q8': cmd += ['--kv', args.kv]
    for i, row in enumerate(rows):
        path = args.output/f'prompt-{i}.ids'
        path.write_text(' '.join(map(str, row['prompt_ids'])))
        cmd += ['--prompt-ids', str(path.resolve())]
    report = {'command': cmd, 'target_kv': args.kv, 'runs': [],
              'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              'environment': {key: os.environ.get(key, 'C++ default') for key in
                  ('MK_FFN_IQ_FUSED', 'MK_IQ_INLINE', 'MK_FFN_IQ_MMA', 'MK_FFN_DOWN_GLOBAL', 'MK_IQ_PAIRED', 'MK_IQ2_PAIRED',
                   'MK_DFLASH_MATH', 'MK_DFLASH_GRAPH', 'MK_DFLASH_MMQ_TYPED',
                   'MK_DFLASH_MMQ_ROWS', 'MK_DFLASH_MMQ_HEAD_ROWS', 'MK_DFLASH_BUDGET')},
              'limits': 'Small task suite; does not establish broad perplexity or all-layer numerical equivalence.'}
    if args.prior_native:
        report['prior_native'] = {'path': str(args.prior_native.resolve()),
            'sha256': hashlib.sha256(args.prior_native.read_bytes()).hexdigest()}
    with (args.output/'quality.out').open('w') as stdout, (args.output/'quality.log').open('w') as stderr:
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(ENGINE.parents[1]/'llama.cpp/build/bin')+os.pathsep+env.get('LD_LIBRARY_PATH', '')
        done = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, timeout=600)
    if done.returncode:
        raise RuntimeError(f'hybrid exit {done.returncode}; see quality.log')
    ids = None
    text = None
    for line in (args.output/'quality.out').read_text().splitlines():
        if line.startswith('IDS:'):
            ids = list(map(int, line[4:].split()))
        elif line.startswith('TEXT_HEX:'):
            text = bytes.fromhex(line[9:]).decode('utf-8', errors='replace')
        elif line.startswith('RESULT '):
            result = json.loads(line[7:])
            row = rows[result['case_index']]
            case = row['case']
            result.update(case=case, ids=ids, content=text,
                llama_content_equal=text==row['response']['content'],
                prior_mk_content_equal=text==prior_mk[case]['content'])
            if prior_native:
                result.update(prior_native_ids_equal=ids==prior_native[case]['ids'],
                              prior_native_content_equal=text==prior_native[case]['content'])
            if 'answer' in questions[case]:
                try:
                    answer = json.loads(text.strip())['answer']
                except (ValueError, KeyError, TypeError):
                    answer = None
                result.update(answer=answer, expected_answer=questions[case]['answer'],
                              task_correct=answer==questions[case]['answer'])
            report['runs'].append(result)
    if len(report['runs']) != len(rows):
        raise RuntimeError('incomplete quality output')
    report['summary'] = {'generations': len(report['runs']),
        'equal_llama_contents': sum(r['llama_content_equal'] for r in report['runs']),
        'equal_prior_mk_contents': sum(r['prior_mk_content_equal'] for r in report['runs']),
        'correct_tasks': sum(r.get('task_correct', False) for r in report['runs']),
        'tasks': sum('task_correct' in r for r in report['runs'])}
    if prior_native:
        report['summary'].update(
            equal_prior_native_ids=sum(r['prior_native_ids_equal'] for r in report['runs']),
            equal_prior_native_contents=sum(r['prior_native_content_equal'] for r in report['runs']))
    (args.output/'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
