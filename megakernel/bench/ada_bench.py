#!/usr/bin/env python3
"""Sequential, reproducible GPU comparison using identical prompt token IDs.

Writes every response, token sequence and stderr log; never assumes greedy
outputs must match exactly between Q8-activation and FP16-activation engines.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request


def post(base, endpoint, body):
    req = urllib.request.Request(base+endpoint, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=600) as r:
        return json.load(r)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-m","--model",required=True,type=Path)
    ap.add_argument("--llama-server",required=True,type=Path)
    ap.add_argument("--megakernel",type=Path,default=Path(__file__).resolve().parents[1]/"qwen38_mk")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--ctx",type=int,default=4096)
    ap.add_argument("--tokens",type=int,default=128)
    ap.add_argument("--reps",type=int,default=2)
    ap.add_argument("--drafts",default="0,1,2,3,4")
    ap.add_argument("--llama-draft",type=int,default=0)
    ap.add_argument("--port",type=int,default=18093)
    ap.add_argument("--prompt",default="Write a Python function that checks whether a number is prime.")
    ap.add_argument("--temperature",type=float,default=0.0)
    ap.add_argument("--top-p",type=float,default=.95)
    ap.add_argument("--top-k",type=int,default=20)
    ap.add_argument("--min-p",type=float,default=0.0)
    ap.add_argument("--seed",type=int,default=123)
    ap.add_argument("--prompt-file",type=Path)
    ap.add_argument("--draft-vocab",type=Path)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    prompt=args.prompt_file.read_text() if args.prompt_file else args.prompt
    text=f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    base=f"http://127.0.0.1:{args.port}"
    cmd=[str(args.llama_server.resolve()),"-m",str(args.model.resolve()),"-ngl","99","-fa","on","-ctk","q8_0","-ctv","q8_0","-c",str(args.ctx),"--parallel","1","--host","127.0.0.1","--port",str(args.port)]
    if args.llama_draft:cmd += ["--spec-type","draft-mtp","--spec-draft-n-max",str(args.llama_draft)]
    results={"model":str(args.model.resolve()),"context":args.ctx,"kv":"q8_0","llama_command":cmd,"llama":[],"megakernel":[]}
    gpu=subprocess.run(["nvidia-smi","--query-gpu=name,compute_cap,memory.total,clocks.max.graphics,clocks.max.memory,driver_version","--format=csv"],capture_output=True,text=True)
    sampling={"temperature":args.temperature,"top_p":args.top_p,"top_k":args.top_k,"min_p":args.min_p,"seed":args.seed}
    results["sampling"]=sampling
    results["gpu"]=gpu.stdout.strip()
    results["prompt"]=prompt
    results["draft_vocab"]=str(args.draft_vocab.resolve()) if args.draft_vocab else None
    log=(args.output/"llama.log").open("w")
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+300
        while True:
            if proc.poll() is not None:raise RuntimeError("llama-server exited; see llama.log")
            try:
                with urllib.request.urlopen(base+"/health",timeout=1) as r:
                    if r.status==200:break
            except Exception:pass
            if time.monotonic()>deadline:raise TimeoutError("llama-server startup")
            time.sleep(.25)
        ids=post(base,"/tokenize",{"content":text,"add_special":False,"parse_special":True})["tokens"]
        (args.output/"prompt.ids").write_text(" ".join(map(str,ids)))
        results["prompt_ids"]=ids
        # Warm GPU kernels/graphs with a separate request before measured reps.
        post(base,"/completion",{"prompt":ids,"n_predict":16,**sampling,"cache_prompt":False})
        for rep in range(args.reps):
            response=post(base,"/completion",{"prompt":ids,"n_predict":args.tokens,**sampling,"cache_prompt":False,"return_tokens":True,"repeat_penalty":1.0})
            results["llama"].append(response)
            (args.output/"results.json").write_text(json.dumps(results,indent=2))
            print("llama",rep,response["timings"]["predicted_per_second"],flush=True)
    finally:
        proc.terminate()
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        log.close()
    reference=results["llama"][0].get("tokens",[])
    (args.output/"reference.ids").write_text(" ".join(map(str,reference)))
    for draft in map(int,filter(None,args.drafts.split(","))):
        for rep in range(args.reps):
            cmd=[str(args.megakernel.resolve()),"-m",str(args.model.resolve()),"-c",str(args.ctx),"--kv","q8","-d",str(draft),"-n",str(args.tokens),"--prompt-ids",str((args.output/"prompt.ids").resolve()),"--ids","--quiet"]
            cmd += ["--temp",str(args.temperature),"--top-p",str(args.top_p),"--top-k",str(args.top_k),"--min-p",str(args.min_p),"--seed",str(args.seed)]
            if args.draft_vocab:cmd += ["--draft-vocab",str(args.draft_vocab.resolve())]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
            (args.output/f"mk-d{draft}-r{rep}.log").write_text(r.stderr)
            (args.output/f"mk-d{draft}-r{rep}.out").write_text(r.stdout)
            if r.returncode:raise RuntimeError(f"megakernel draft={draft} failed; see log")
            output=list(map(int,re.search(r"^IDS: (.*)$",r.stdout,re.M)[1].split()))
            speed=float(re.search(r"= ([\d.]+) tok/s",r.stderr)[1])
            prefix=0
            for a,b in zip(reference,output):
                if a!=b:break
                prefix+=1
            row={"draft":draft,"rep":rep,"tokens_per_second":speed,"ids":output,"reference_common_prefix":prefix}
            results["megakernel"].append(row)
            print("megakernel",draft,rep,speed,"common prefix",prefix,flush=True)
            (args.output/"results.json").write_text(json.dumps(results,indent=2))
    if reference and args.temperature == 0:
        r=subprocess.run([str(args.megakernel.resolve()),"-m",str(args.model.resolve()),"-c",str(args.ctx),"--kv","q8","-d","0","--prompt-ids",str((args.output/"prompt.ids").resolve()),"--score",str((args.output/"reference.ids").resolve())],capture_output=True,text=True,timeout=600)
        (args.output/"teacher-forced.log").write_text(r.stderr)
        results["teacher_forced"]={"returncode":r.returncode,"stdout":r.stdout}
        print(r.stdout.strip(),flush=True)
    (args.output/"results.json").write_text(json.dumps(results,indent=2))

if __name__=="__main__":main()
