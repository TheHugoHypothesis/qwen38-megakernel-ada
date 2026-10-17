#!/usr/bin/env python3
"""Benchmark multi-turn agent prompt caching and long-context prefill over the HTTP server."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--draft-model", required=True, type=Path)
    ap.add_argument("--lib", type=Path, default=Path(__file__).resolve().parents[1] / "libqwen38mk.so")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--ctx", type=int, default=40000)
    ap.add_argument("--kv", default="q4")
    ap.add_argument("--prefill-chunk", default="auto")
    ap.add_argument("--no-reasoning-cache", action="store_true")
    ap.add_argument("--port", type=int, default=18119)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = json.loads((root / "bench/ada-deep/agent/fixtures.json").read_text())
    patch_8k = next(c for c in fixtures["cases"] if c["workload"] == "patch" and c["requested_tokens"] == 8192)
    raw_prompt = Path(patch_8k["prompt_ids"]).with_suffix(".txt").read_text()
    t0 = raw_prompt.find("<tool_response>\n")
    t1 = raw_prompt.rfind("</tool_response>")
    first_user_content = raw_prompt[t0 + len("<tool_response>\n"):t1]

    tools = [
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read a repository file before proposing a patch.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }},
        {"type": "function", "function": {
            "name": "grep_code",
            "description": "Search for a symbol in the repository.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        }},
    ]

    cmd = [
        sys.executable, str(root / "server/mk_server.py"),
        "-m", str(args.model.resolve()),
        "--draft-model", str(args.draft_model.resolve()),
        "--lib", str(args.lib.resolve()),
        "-c", str(args.ctx),
        "--kv", args.kv,
        "-d", "7",
        "--prefill-chunk", str(args.prefill_chunk),
        "--port", str(args.port),
    ]
    base = f"http://127.0.0.1:{args.port}"

    def complete(body):
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.load(resp)

    import os
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str((root.parents[1] / "llama.cpp/build/bin").resolve()) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    if args.no_reasoning_cache:
        env["MK_DISABLE_REASONING_CACHE"] = "1"
    with (args.output / "server.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
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
                time.sleep(0.25)

            turns = []

            # Turn 1: Cold 8K populated repo prompt -> thinking + tool call
            msg_turn1 = [
                {"role": "system", "content": "You are a systems coding agent. Inspect files with tools before editing."},
                {"role": "user", "content": first_user_content + "\n\nFirst, call read_file on megakernel/capi.cpp."},
            ]
            body1 = {
                "messages": msg_turn1,
                "tools": tools,
                "reasoning_effort": "low",
                "temperature": 0,
                "seed": 123,
                "max_tokens": 192,
            }
            r1 = complete(body1)
            m1 = r1["choices"][0]["message"]
            turns.append({
                "turn": "1_cold_8k_tool_call",
                "prompt_tokens": r1["usage"]["prompt_tokens"],
                "cached_tokens": r1["usage"]["prompt_tokens_details"]["cached_tokens"],
                "prompt_ms": r1["timings"]["prompt_ms"],
                "completion_tokens": r1["usage"]["completion_tokens"],
                "predicted_ms": r1["timings"]["predicted_ms"],
                "predicted_per_second": r1["timings"]["predicted_per_second"],
                "has_reasoning": bool(m1.get("reasoning_content")),
                "tool_calls": m1.get("tool_calls") or [],
            })

            # Turn 2: Same-prompt retry / resample at seed=456 (tests pre-final-token checkpoint hit on 8K context)
            r2 = complete({**body1, "seed": 456})
            turns.append({
                "turn": "2_same_prompt_8k_retry",
                "prompt_tokens": r2["usage"]["prompt_tokens"],
                "cached_tokens": r2["usage"]["prompt_tokens_details"]["cached_tokens"],
                "prompt_ms": r2["timings"]["prompt_ms"],
                "completion_tokens": r2["usage"]["completion_tokens"],
                "predicted_ms": r2["timings"]["predicted_ms"],
                "predicted_per_second": r2["timings"]["predicted_per_second"],
            })

            # Turn 3: Tool response roundtrip where client strips reasoning_content (standard OpenAI SDK behavior)
            calls = m1.get("tool_calls") or []
            assert calls, f"expected tool call on turn 1, got {m1}"
            stripped_assistant = {
                "role": "assistant",
                "content": m1.get("content"),
                "tool_calls": calls,
            }  # Note: reasoning_content intentionally omitted!
            msg_turn3 = msg_turn1 + [
                stripped_assistant,
                {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "content": "int mk_rewind_to_checkpoint(void * hp) { return h->eng->rewind_to_checkpoint() ? 0 : -1; }",
                },
                {"role": "user", "content": "Now call grep_code with pattern checkpoint_pos."},
            ]
            r3 = complete({**body1, "messages": msg_turn3})
            m3 = r3["choices"][0]["message"]
            turns.append({
                "turn": "3_tool_roundtrip_stripped_reasoning",
                "prompt_tokens": r3["usage"]["prompt_tokens"],
                "cached_tokens": r3["usage"]["prompt_tokens_details"]["cached_tokens"],
                "incremental_tokens": r3["usage"]["prompt_tokens"] - r3["usage"]["prompt_tokens_details"]["cached_tokens"],
                "prompt_ms": r3["timings"]["prompt_ms"],
                "completion_tokens": r3["usage"]["completion_tokens"],
                "predicted_ms": r3["timings"]["predicted_ms"],
                "predicted_per_second": r3["timings"]["predicted_per_second"],
                "tool_calls": m3.get("tool_calls") or [],
            })

            # Turn 4: Second chained tool response roundtrip (also with stripped reasoning_content)
            calls3 = m3.get("tool_calls") or []
            assert calls3, f"expected tool call on turn 3, got {m3}"
            stripped_assistant3 = {
                "role": "assistant",
                "content": m3.get("content"),
                "tool_calls": calls3,
            }
            msg_turn4 = msg_turn3 + [
                stripped_assistant3,
                {
                    "role": "tool",
                    "tool_call_id": calls3[0]["id"],
                    "content": "capi.cpp:200: int mk_checkpoint_pos(void * hp) { return static_cast<Handle *>(hp)->eng->checkpoint_pos(); }",
                },
                {"role": "user", "content": "Summarize in one sentence what mk_checkpoint_pos returns."},
            ]
            r4 = complete({**body1, "messages": msg_turn4, "max_tokens": 128})
            turns.append({
                "turn": "4_second_tool_roundtrip_summary",
                "prompt_tokens": r4["usage"]["prompt_tokens"],
                "cached_tokens": r4["usage"]["prompt_tokens_details"]["cached_tokens"],
                "incremental_tokens": r4["usage"]["prompt_tokens"] - r4["usage"]["prompt_tokens_details"]["cached_tokens"],
                "prompt_ms": r4["timings"]["prompt_ms"],
                "completion_tokens": r4["usage"]["completion_tokens"],
                "predicted_ms": r4["timings"]["predicted_ms"],
                "predicted_per_second": r4["timings"]["predicted_per_second"],
                "content": r4["choices"][0]["message"].get("content"),
            })

            out_data = {"prefill_chunk": args.prefill_chunk, "ctx": args.ctx, "kv": args.kv, "turns": turns}
            (args.output / "results.json").write_text(json.dumps(out_data, indent=2) + "\n")
            for t in turns:
                print(
                    f"{t['turn']:36s} | cached {t['cached_tokens']:5d}/{t['prompt_tokens']:5d} "
                    f"| prompt {t['prompt_ms']:8.1f} ms | gen {t['completion_tokens']:3d} tok "
                    f"@ {t['predicted_per_second']:5.1f} tok/s",
                    flush=True,
                )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
