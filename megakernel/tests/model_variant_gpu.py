#!/usr/bin/env python3
"""Validate and measure a compatible refined target with an existing DFlash2 draft.

Runs one GPU engine at a time. Records prompt/output IDs, acceptance, timing and
llama.cpp comparisons; disagreement is reported rather than hidden. Model-level
quality and long-context capability cannot be inferred from these fixtures.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from mk_server import Engine


def suite():
    functional = [
        ('sum', 'Answer with just the integer: what is 17 + 28?', '45'),
        ('multiply', 'Answer with just the integer: what is 12 times 13?', '156'),
        ('divide', 'Answer with just the integer: what is 144 divided by 12?', '12'),
        ('portuguese', 'Responda somente o inteiro: quanto é 19 + 23?', '42'),
        ('python', 'Write a short Python function that returns the square of an integer.', None),
        ('unicode', 'Repeat exactly: café α → 中文 😀', 'café α → 中文 😀'),
    ]
    cases = [dict(name=n, prompt=p, expected=e, tokens=64, benchmark=False) for n, p, e in functional]
    code = 'Write a Python function to test primality. Include docstring, implementation and five tests.'
    cases += [dict(name='short_code', prompt=code, tokens=256, benchmark=True),
              dict(name='long_code', prompt='\n'.join(f'Record {i}: alpha beta gamma delta.' for i in range(240)) + '\n' + code,
                   tokens=256, benchmark=True),
              dict(name='numbered_lines', prompt='Generate exactly 100 numbered lines from 0001 to 0100. Each line: 0001 | Parallel matrix processing requires efficient memory management and synchronization. No introduction. Start immediately.',
                   tokens=256, benchmark=True)]
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--draft-model', required=True, type=Path)
    ap.add_argument('--lib', type=Path, default=ROOT / 'libqwen38mk.so')
    ap.add_argument('--llama-server', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--port', type=int, default=18106)
    args = ap.parse_args()
    if args.reps < 1 or args.ctx < 4096:
        ap.error('use at least one repetition and context >= 4096')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['MK_PREFILL_CHUNK'] = '128'
    cases = suite()
    report = {'model': str(args.model.resolve()), 'draft_model': str(args.draft_model.resolve()),
              'library_sha256': hashlib.sha256(args.lib.read_bytes()).hexdigest(),
              'ctx': args.ctx, 'kv': 'q8', 'temperature': 0, 'stock_clocks': True, 'cases': cases, 'runs': []}

    def save():
        (args.output / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    def stats(eng):
        values = [ctypes.c_long() for _ in range(3)]
        eng.L.mk_stats(eng.h, *[ctypes.byref(x) for x in values])
        return [x.value for x in values]

    def text(spec):
        return '<|im_start|>user\n' + spec['prompt'] + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    for mode, draft in [('mk_scalar', None), ('mk_dflash2', args.draft_model)]:
        eng = Engine(str(args.lib.resolve()), str(args.model.resolve()), args.ctx,
                     7 if draft else 0, None, 1, draft_model=str(draft.resolve()) if draft else None)
        eng.L.mk_free.argtypes, eng.L.mk_free.restype = [ctypes.c_void_p], None
        eng.L.mk_stats.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_long)] * 3
        eng.L.mk_stats.restype = None
        try:
            eng.set_sampling(0, .95, 20, 0, 123)

            def generate(spec):
                eng.reset()
                prompt_ids = eng.tokenize(text(spec))
                before = stats(eng)
                start = time.perf_counter()
                first = eng.prefill(prompt_ids)
                prefill_seconds = time.perf_counter() - start
                ids = [first]
                start = time.perf_counter()
                while len(ids) < spec['tokens'] and not eng.is_eog(ids[-1]):
                    for token in eng.step(spec['tokens'] - len(ids)):
                        ids.append(token)
                        if eng.is_eog(token):
                            break
                seconds = time.perf_counter() - start
                after = stats(eng)
                valid = [i for i in ids if not eng.is_eog(i)]
                return {'prompt_ids': prompt_ids, 'output_ids': valid,
                        'text': b''.join(eng.piece(i) for i in valid).decode('utf-8', errors='replace'),
                        'prefill_seconds': prefill_seconds, 'decode_seconds': seconds,
                        'decode_tokens': len(ids) - 1, 'tps': (len(ids) - 1) / seconds,
                        'cycles': after[0] - before[0], 'drafted': after[1] - before[1], 'accepted': after[2] - before[2]}

            generate(dict(cases[4], tokens=16))  # warmup, excluded
            for spec in cases:
                for rep in range(args.reps if spec['benchmark'] else 1):
                    row = {'mode': mode, 'case': spec['name'], 'rep': rep, **generate(spec)}
                    if spec.get('expected') is not None:
                        row['task_correct'] = row['text'].strip() == spec['expected']
                    report['runs'].append(row)
                    save()
                    print(mode, spec['name'], rep, f"{row['tps']:.1f} tok/s", f"accepted {row['accepted']}/{row['drafted']}", flush=True)
        finally:
            eng.L.mk_free(eng.h)

    cmd = [str(args.llama_server.resolve()), '-m', str(args.model.resolve()), '-ngl', '99', '-fa', 'on',
           '-ctk', 'q8_0', '-ctv', 'q8_0', '-c', str(args.ctx), '--parallel', '1', '--fit', 'off',
           '--host', '127.0.0.1', '--port', str(args.port)]
    report['llama_command'] = cmd
    base = f'http://127.0.0.1:{args.port}'
    def request(endpoint, payload):
        req = urllib.request.Request(base + endpoint, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response)
    with (args.output / 'llama.log').open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('llama-server exited; inspect llama.log')
                try:
                    with urllib.request.urlopen(base + '/health', timeout=1):
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('llama startup')
                time.sleep(.25)
            for spec in cases:
                ids = request('/tokenize', {'content': text(spec), 'add_special': False, 'parse_special': True})['tokens']
                scalar = next(r for r in report['runs'] if r['mode'] == 'mk_scalar' and r['case'] == spec['name'])
                if ids != scalar['prompt_ids']:
                    raise AssertionError('llama/MK tokenizer mismatch')
                data = request('/completion', {'prompt': ids, 'n_predict': spec['tokens'], 'temperature': 0,
                                              'repeat_penalty': 1, 'cache_prompt': False, 'return_tokens': True})
                report['runs'].append({'mode': 'llama_scalar', 'case': spec['name'], 'rep': 0,
                    'prompt_ids': ids, 'output_ids': data.get('tokens', []), 'text': data['content'], 'response': data,
                    'tps': data['timings']['predicted_per_second']})
                save()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    report['comparisons'] = []
    for spec in cases:
        rows = {m: next(r for r in report['runs'] if r['mode'] == m and r['case'] == spec['name'])
                for m in ('mk_scalar', 'mk_dflash2', 'llama_scalar')}
        a, b = rows['mk_scalar']['output_ids'], rows['mk_dflash2']['output_ids']
        prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        report['comparisons'].append({'case': spec['name'],
            'native_scalar_ids_equal': a == b, 'native_scalar_common_prefix': prefix,
            'native_llama_text_equal': rows['mk_dflash2']['text'] == rows['llama_scalar']['text'],
            'scalar_llama_text_equal': rows['mk_scalar']['text'] == rows['llama_scalar']['text']})
    report['benchmarks'] = []
    for spec in (s for s in cases if s['benchmark']):
        modes = {}
        for mode in ('mk_scalar', 'mk_dflash2'):
            rows = [r for r in report['runs'] if r['mode'] == mode and r['case'] == spec['name']]
            modes[mode] = {'median_tps': statistics.median(r['tps'] for r in rows),
                          'min_tps': min(r['tps'] for r in rows), 'max_tps': max(r['tps'] for r in rows),
                          'repeatable_ids': len({tuple(r['output_ids']) for r in rows}) == 1 if len(rows) > 1 else None,
                          'acceptance': sum(r['accepted'] for r in rows) / max(1, sum(r['drafted'] for r in rows))}
        report['benchmarks'].append({'case': spec['name'], **modes,
                                    'native_scalar_speedup': modes['mk_dflash2']['median_tps'] / modes['mk_scalar']['median_tps']})
    save()
    print(json.dumps({'comparisons': report['comparisons'], 'benchmarks': report['benchmarks']}, indent=2), flush=True)
    if not all(r.get('task_correct', True) for r in report['runs'] if r['case'] in ('sum', 'multiply', 'divide', 'portuguese')):
        raise AssertionError('known arithmetic task failed; see results.json')


if __name__ == '__main__':
    main()
