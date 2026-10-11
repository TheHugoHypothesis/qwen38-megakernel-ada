#!/usr/bin/env python3
"""Exercise the real HTTP/CUDA engine; run separately from CPU unit tests."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--lib", type=Path)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--draft-model", type=Path)
    ap.add_argument("--draft", type=int, default=4)
    ap.add_argument("--kv", choices=("q8", "q4", "f16", "auto"), default="q8")
    ap.add_argument("--prefill-chunk", default="auto")
    ap.add_argument("--long-records", type=int, help="extended prefill and secret-code retrieval check")
    ap.add_argument("--port", type=int, default=18094)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(root / "server/mk_server.py"), "-m", str(args.model.resolve()),
           "-c", str(args.ctx), "--kv", args.kv, "-d", str(args.draft),
           "--prefill-chunk", args.prefill_chunk, "--port", str(args.port)]
    if args.draft_model:
        cmd += ["--draft-model", str(args.draft_model.resolve())]
    if args.lib:
        cmd += ["--lib", str(args.lib.resolve())]
    base = f"http://127.0.0.1:{args.port}"
    reports = []

    def request(body):
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=600)

    def complete(body):
        with request(body) as response:
            return json.load(response)

    with (args.output / "server.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("server exited; inspect server.log")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1) as r:
                        if r.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup")
                time.sleep(.25)
            body = {"messages": [{"role": "user", "content": "Write a Python function to test primality."}],
                    "temperature": 0, "max_tokens": 64,
                    "chat_template_kwargs": {"enable_thinking": False}}
            first, second = complete(body), complete(body)
            content = first["choices"][0]["message"]["content"]
            assert content and content == second["choices"][0]["message"]["content"], "greedy repeat differs"
            reports += [{"check": "greedy_repeat", "responses": [first, second]}]
            chunks, done = [], False
            with request({**body, "stream": True}) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    obj = json.loads(data)
                    for choice in obj.get("choices", []):
                        chunks.append(choice["delta"].get("content") or "")
            assert done and "".join(chunks) == content, "stream differs from non-stream"
            reports += [{"check": "stream", "content": "".join(chunks)}]
            if args.draft_model:
                for limit in (1, 2, 3, 7, 8, 9):
                    bounded = complete({**body, "max_tokens": limit})
                    bounded_text = bounded["choices"][0]["message"]["content"]
                    assert bounded["usage"]["completion_tokens"] == limit, "token budget was not respected"
                    assert content.startswith(bounded_text), "bounded decode changed the greedy prefix"
                    reports += [{"check": "bounded_tail", "limit": limit, "response": bounded}]
            sampled = {**body, "temperature": .7, "top_k": 20, "top_p": .95, "seed": 123}
            a, b = complete(sampled), complete(sampled)
            assert a["choices"][0]["message"] == b["choices"][0]["message"], "seed repeat differs"
            reports += [{"check": "seed_repeat", "responses": [a, b]}]
            for invalid in [{"presence_penalty": 1}, {"temperature": .7, "top_k": 40}]:
                try:
                    complete({**body, **invalid})
                except urllib.error.HTTPError as error:
                    assert error.code == 400
                    reports += [{"check": "unsupported_sampler", "body": invalid,
                                 "error": json.loads(error.read())}]
                else:
                    raise AssertionError("unsupported sampler did not return 400")
            history = body["messages"] + [{"role": "assistant", "content": content},
                                         {"role": "user", "content": "Explain its time complexity."}]
            turn = complete({**body, "messages": history})
            assert turn["choices"][0]["message"]["content"]
            reports += [{"check": "second_turn", "response": turn}]
            records = args.long_records if args.long_records is not None else (240 if args.draft_model else 120)
            long_prompt = "\n".join(f"Record {i}: alpha beta gamma delta." for i in range(records))
            if args.long_records is not None:
                long_prompt = "The secret code is ORION-4060.\n" + long_prompt
                long_prompt += "\nWhat is the secret code from the beginning? Reply with the code only."
            else:
                long_prompt += "\nSummarize these records in one sentence."
            long = complete({**body, "messages": [{"role": "user", "content": long_prompt}]})
            assert long["usage"]["prompt_tokens"] > 512
            if args.draft_model:
                assert long["usage"]["prompt_tokens"] > 2048, "draft SWA boundary was not exercised"
            assert long["choices"][0]["message"]["content"]
            if args.long_records is not None:
                assert "ORION-4060" in long["choices"][0]["message"]["content"], "extended-context secret retrieval failed"
            reports += [{"check": "prefill_chunk_boundary", "response": long}]
            memory = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True)
            result = {"checks": reports, "command": cmd, "nvidia_memory_mib": memory.stdout.strip()}
            (args.output / "results.json").write_text(json.dumps(result, indent=2))
            print(f"PASS: {len(reports)} HTTP/CUDA checks; context {args.ctx}, KV {args.kv}, draft {args.draft}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
