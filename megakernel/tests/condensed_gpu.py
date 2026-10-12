#!/usr/bin/env python3
"""Compare native DFlash2 with scalar MK and verify MTP stripping on real GGUFs."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from mk_server import Engine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lib", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--target-with-mtp", type=Path, required=True)
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--kv", choices=("q8", "q4"), default="q8")
    args = p.parse_args()
    os.environ["MK_PREFILL_CHUNK"] = "128"
    prompts = [
        ("Answer with just the integer: what is 17 + 28?", "45"),
        ("Answer with just the integer: what is 12 times 13?", "156"),
        ("Answer with just the integer: what is 144 divided by 12?", "12"),
        ("Responda somente o inteiro: quanto é 19 + 23?", "42"),
        ("Write a short Python function that returns the square of an integer.", None),
        ("Repeat exactly: café α → 中文 😀", None),
    ]
    results = {}
    for name, model, draft in (("with_mtp_native", args.target_with_mtp, args.draft),
                               ("no_mtp_native", args.target, args.draft),
                               ("no_mtp_scalar", args.target, None)):
        eng = Engine(str(args.lib.resolve()), str(model.resolve()), args.ctx,
                     7 if draft else 0, None, {"q8": 1, "q4": 2}[args.kv],
                     draft_model=str(draft.resolve()) if draft else None)
        eng.L.mk_free.argtypes = [ctypes.c_void_p]
        eng.L.mk_free.restype = None
        runs = []
        try:
            eng.set_sampling(0, .95, 20, 0, 123)
            for index, (prompt, expected) in enumerate(prompts):
                eng.reset()
                text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                toks = eng.tokenize(text)
                ids = [eng.prefill(toks)]
                start = time.perf_counter()
                while len(ids) < 64 and not eng.is_eog(ids[-1]):
                    for token in eng.step(64 - len(ids)):
                        ids.append(token)
                        if eng.is_eog(token):
                            break
                seconds = time.perf_counter() - start
                answer = b"".join(eng.piece(i) for i in ids if not eng.is_eog(i)).decode("utf-8", errors="replace")
                if expected is not None and answer.strip() != expected:
                    raise AssertionError(f"{name}: expected {expected!r}, got {answer!r}")
                runs.append({"prompt": prompt, "prompt_ids": toks, "output_ids": ids,
                             "text": answer, "decode_seconds": seconds,
                             "tokens_per_second": (len(ids) - 1) / seconds})
        finally:
            eng.L.mk_free(eng.h)
        results[name] = runs
    # MTP is disabled in both native runs, so physically stripping it must have no effect.
    for a, b in zip(results["with_mtp_native"], results["no_mtp_native"]):
        if a["output_ids"] != b["output_ids"] or a["prompt_ids"] != b["prompt_ids"]:
            raise AssertionError("removing unused MTP changed native output")
    results["native_scalar_matches"] = [a["output_ids"] == b["output_ids"]
        for a, b in zip(results["no_mtp_native"], results["no_mtp_scalar"])]
    results["ctx"] = args.ctx
    results["kv"] = args.kv
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print("PASS: four known answers per mode, six identical MTP/no-MTP generations; native/scalar matches:",
          results["native_scalar_matches"])


if __name__ == "__main__":
    main()
