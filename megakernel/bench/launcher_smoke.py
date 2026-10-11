#!/usr/bin/env python3
"""Exercise the actual serve_ada launcher and chat API; clean up owned servers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engines', default='dflash2,megakernel')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--port', type=int, default=18098)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    launcher = Path(__file__).resolve().parents[1] / 'serve_ada.py'
    base = f'http://127.0.0.1:{args.port}'
    reports = []

    def save():
        (args.output / 'results.json').write_text(json.dumps(reports, indent=2) + '\n')

    def request(endpoint, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + endpoint, data=data,
                                     headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=300)

    def complete(body):
        with request('/v1/chat/completions', body) as response:
            return json.load(response)

    for engine in args.engines.split(','):
        cmd = [sys.executable, str(launcher), '--engine', engine, '--port', str(args.port)]
        row = {'engine': engine, 'command': cmd, 'checks': []}
        reports.append(row)
        save()
        with (args.output / f'{engine}.log').open('w') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 300
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'{engine} exited; see {engine}.log')
                    try:
                        with request('/health') as response:
                            if response.status == 200:
                                row['health'] = json.load(response)
                                break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('server startup')
                    time.sleep(.25)
                with request('/v1/models') as response:
                    row['models'] = json.load(response)
                body = {'messages': [{'role': 'user', 'content':
                         'Compute 2 + 2. Reply only with JSON: {"answer": NUMBER}.'}],
                        'temperature': 0, 'max_tokens': 64,
                        'chat_template_kwargs': {'enable_thinking': False}}
                first, second = complete(body), complete(body)
                content = first['choices'][0]['message']['content']
                assert json.loads(content)['answer'] == 4
                assert content == second['choices'][0]['message']['content']
                row['checks'].append({'check': 'answer_and_greedy_repeat', 'responses': [first, second]})
                chunks, done = [], False
                with request('/v1/chat/completions', {**body, 'stream': True}) as response:
                    for line in response:
                        if not line.startswith(b'data: '):
                            continue
                        data = line[6:].strip()
                        if data == b'[DONE]':
                            done = True
                            break
                        for choice in json.loads(data).get('choices', []):
                            chunks.append(choice['delta'].get('content') or '')
                assert done and ''.join(chunks) == content
                row['checks'].append({'check': 'stream_matches', 'content': ''.join(chunks)})
                sampled = {**body, 'messages': [{'role': 'user', 'content':
                           'Write a short Python function to test primality.'}],
                           'temperature': 1, 'top_k': 20, 'top_p': .95, 'seed': 123}
                a, b = complete(sampled), complete(sampled)
                assert a['choices'][0]['message'] == b['choices'][0]['message']
                row['checks'].append({'check': 'seed_repeat', 'responses': [a, b]})
                history = body['messages'] + [{'role': 'assistant', 'content': content},
                          {'role': 'user', 'content': 'Now compute 4 + 4 in the same JSON format.'}]
                turn = complete({**body, 'messages': history})
                assert json.loads(turn['choices'][0]['message']['content'])['answer'] == 8
                row['checks'].append({'check': 'second_turn', 'response': turn})
                row['gpu_after_requests'] = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,clocks.current.graphics,clocks.current.memory',
                     '--format=csv'], capture_output=True, text=True).stdout
                row['passed'] = True
                print(f'PASS {engine}: health, models, answer, repeat, stream, seed, second turn', flush=True)
            except Exception as exc:
                row['passed'] = False
                row['error'] = repr(exc)
                raise
            finally:
                save()
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    main()
