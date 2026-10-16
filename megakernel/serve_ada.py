#!/usr/bin/env python3
"""Launch one of the locally validated RTX 4060 Ti inference configurations."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parent


def target_profile(target, llama_root):
    """Read variant metadata; full tensor validation remains in the CUDA loader."""
    sys.path.insert(0, str(llama_root / 'gguf-py'))
    import gguf
    reader = gguf.GGUFReader(target)
    def value(key, default=None):
        return reader.fields[key].contents() if key in reader.fields else default
    mtp = int(value('qwen35.nextn_predict_layers', 0))
    if value('general.architecture') != 'qwen35' or mtp not in (0, 1) or value('qwen35.block_count') != 64 + mtp:
        raise ValueError('expected qwen35 with 64 trunk layers and zero or one MTP layer')
    return {'name': value('general.name'), 'vocabulary': len(value('tokenizer.ggml.tokens')),
            'mtp_layers': mtp, 'model_context': int(value('qwen35.context_length'))}


def draft_depth(engine, requested, mtp_layers):
    if engine == 'mtp' and not mtp_layers:
        raise ValueError('target has no MTP head; select megakernel or megakernel-dflash2')
    depth = requested if requested is not None else (7 if engine in ('dflash2', 'megakernel-dflash2') else 4 if mtp_layers else 0)
    if engine == 'megakernel' and depth > 0 and not mtp_layers:
        raise ValueError('target has no MTP head; use --draft 0 or --engine megakernel-dflash2')
    return depth


def check_draft(target,draft,llama_root=None):
    llama_root = llama_root or WORKSPACE / 'llama.cpp'
    sys.path.insert(0,str(llama_root/'gguf-py'))
    import gguf
    t=gguf.GGUFReader(target); d=gguf.GGUFReader(draft)
    def value(reader,key):
        if key not in reader.fields: raise ValueError(f'missing GGUF metadata: {key}')
        return reader.fields[key].contents()
    if value(t,'general.architecture')!='qwen35' or value(d,'general.architecture')!='dflash':
        raise ValueError('expected qwen35 target and dflash draft')
    if value(t,'qwen35.embedding_length')!=value(d,'dflash.embedding_length'):
        raise ValueError('target/draft hidden dimensions differ')
    if value(d,'dflash.selector_top_k')<=0: raise ValueError('draft is not DFlash2')
    layers=value(d,'dflash.target_layers')
    if not layers or any(x<0 or x>=value(t,'qwen35.block_count') for x in layers):
        raise ValueError('draft feature taps are outside the target')
    for key in ('tokenizer.ggml.model','tokenizer.ggml.pre','tokenizer.ggml.tokens',
                'tokenizer.ggml.token_type','tokenizer.ggml.merges',
                'tokenizer.ggml.bos_token_id','tokenizer.ggml.eos_token_id',
                'tokenizer.ggml.padding_token_id','tokenizer.ggml.add_bos_token'):
        if value(t,key)!=value(d,key):
            raise ValueError(f'target/draft differ: {key}; an ASCII-condensed target requires a remapped DFlash2 draft from tools/prepare_condensed.py')
    return int(value(d,'dflash.block_size'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine',choices=('dflash2','mtp','megakernel','megakernel-dflash2'),default='dflash2')
    ap.add_argument('--model',type=Path)
    ap.add_argument('--draft-model',type=Path,default=WORKSPACE/'models-dflash2-qwen38/Qwen3.8-27B-DFlash2-Q4_K_M.gguf')
    ap.add_argument('--llama-root',type=Path,default=WORKSPACE/'llama.cpp',
                    help='llama.cpp source/build directory (default: ./llama.cpp)')
    ap.add_argument('--ctx',type=int,default=4096)
    ap.add_argument('--draft',type=int)
    ap.add_argument('--port',type=int,default=8080)
    ap.add_argument('--host',default='127.0.0.1')
    ap.add_argument('--prefill-chunk',default='auto')
    ap.add_argument('--verify-width', choices=('full', '3', 'auto'),
                    help='native DFlash2 verification width; auto is an experimental acceptance-based policy')
    ap.add_argument('--kv',choices=('q8','q4','f16','auto'),default='q8',
                    help='target KV format for megakernel engines; Q4 is experimental and lossy')
    ap.add_argument('--check',action='store_true',help='validate files and print the command without loading the GPU')
    args=ap.parse_args()
    if not args.model:
        candidates=list(WORKSPACE.glob('models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF/snapshots/*/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf'))
        if len(candidates)!=1: ap.error('specify --model; expected exactly one local target')
        args.model=candidates[0]
    if not args.model.is_file(): ap.error('target GGUF does not exist')
    try:
        profile = target_profile(args.model, args.llama_root.resolve())
        depth = draft_depth(args.engine, args.draft, profile['mtp_layers'])
    except (ValueError, KeyError, TypeError) as error:
        ap.error(str(error))
    if args.ctx<=0 or depth<0: ap.error('invalid context or draft depth')
    if args.verify_width is not None and args.engine != 'megakernel-dflash2':
        ap.error('--verify-width requires --engine megakernel-dflash2')
    if args.ctx > profile['model_context']: ap.error('requested context exceeds the model GGUF context')
    if args.engine in ('megakernel','megakernel-dflash2'):
        if args.engine=='megakernel' and depth>4: ap.error('megakernel supports at most four MTP proposals')
        cmd=[sys.executable,str(HERE/'server/mk_server.py'),'--model',str(args.model.resolve()),
             '--lib',str(HERE/'libqwen38mk.so'),'--ctx',str(args.ctx),'--draft',str(depth),
             '--kv',args.kv,'--prefill-chunk',args.prefill_chunk,'--host',args.host,'--port',str(args.port)]
        if args.engine=='megakernel-dflash2':
            if not args.draft_model.is_file(): ap.error('DFlash2 draft GGUF does not exist')
            block=check_draft(args.model,args.draft_model,args.llama_root.resolve())
            if not 1<=depth<block: ap.error(f'native DFlash2 depth must be in [1,{block-1}]')
            cmd+=['--draft-model',str(args.draft_model.resolve())]
            if args.verify_width is not None:
                cmd+=['--verify-width',args.verify_width]
    else:
        if args.kv != 'q8': ap.error('--kv f16/q4/auto currently requires a megakernel engine')
        binary=args.llama_root/'build/bin/llama-server'
        if not binary.is_file(): ap.error('local llama-server is not built')
        cmd=[str(binary),'-m',str(args.model.resolve()),'-ngl','99','-fa','on','-ctk','q8_0','-ctv','q8_0',
             '-c',str(args.ctx),'--parallel','1','--fit','off','--host',args.host,'--port',str(args.port),
             '--temp','1','--top-k','20','--top-p','0.95','--min-p','0']
        if args.engine=='dflash2':
            if not args.draft_model.is_file(): ap.error('DFlash2 draft GGUF does not exist')
            block=check_draft(args.model,args.draft_model,args.llama_root.resolve())
            if depth>=block: ap.error(f'draft depth must be smaller than trained block size {block}')
            cmd+=['--spec-type','draft-dflash','--spec-draft-model',str(args.draft_model.resolve()),
                  '--spec-draft-ngl','99','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0']
        else: cmd+=['--spec-type','draft-mtp']
        cmd+=['--spec-draft-n-max',str(depth)]
    print(json.dumps({'engine':args.engine,'target':profile,'command':cmd},ensure_ascii=False,indent=2),flush=True)
    if not args.check: os.execv(cmd[0],cmd)


if __name__=='__main__':main()
