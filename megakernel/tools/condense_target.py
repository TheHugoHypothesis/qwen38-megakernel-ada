#!/usr/bin/env python3
"""Apply an existing verified vocabulary subset to a standard no-MTP target.

For Swift 1.5, use the previously verified P1G target as --tokenizer-reference
and its adapted DFlash2 as --draft. Only the Swift embedding/output rows are
gathered; no trunk weights are borrowed from the reference, and no tensor is
dequantized. Every retained row and every other payload is checked exhaustively.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_condensed import VOCAB_WEIGHTS, export_gguf, file_hash, token_mapping, value, verify_source


def verify_draft(target, draft):
    if value(draft, 'general.architecture') != 'dflash':
        raise ValueError('expected a DFlash2 draft')
    for key, field in target.fields.items():
        if key.startswith('tokenizer.') and key != 'tokenizer.chat_template':
            if key not in draft.fields or field.contents() != value(draft, key):
                raise ValueError(f'condensed draft tokenizer mismatch: {key}')
    if (value(draft, 'dflash.embedding_length') != 5120 or
        value(draft, 'dflash.target_layers') != [6, 20, 34, 48, 62] or
        value(draft, 'dflash.selector_top_k') != 16 or value(draft, 'dflash.block_size') != 8):
        raise ValueError('unsupported DFlash2 dimensions/taps/selector')
    tensors = {t.name: t for t in draft.tensors}
    vocab = len(value(target, 'tokenizer.ggml.tokens'))
    for name in ('selector_predecessor.weight', 'selector_successor.weight'):
        if name not in tensors or list(tensors[name].shape) != [256, vocab]:
            raise ValueError(f'condensed draft selector shape mismatch: {name}')
    mask = value(draft, 'tokenizer.ggml.mask_token_id')
    if not 0 <= mask < vocab or value(draft, 'tokenizer.ggml.tokens')[mask] != '<|audio_start|>':
        raise ValueError('invalid DFlash2 mask token')
    return mask


def condense(source_path, reference_path, draft_path, output_dir, expected_source_sha256=None,
             expected_reference_sha256=None, expected_draft_sha256=None, schema_validator=None,
             source_license=None):
    import gguf
    import numpy as np
    source_path, reference_path, draft_path = map(Path, (source_path, reference_path, draft_path))
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite output directory {output_dir}')
    sources = {}
    for key, path, expected in [('target', source_path, expected_source_sha256),
                               ('vocabulary_reference', reference_path, expected_reference_sha256),
                               ('condensed_draft', draft_path, expected_draft_sha256)]:
        digest = file_hash(path)
        if expected is not None and digest != expected:
            raise ValueError(f'{key} SHA-256 differs from the pinned input')
        sources[key] = {'path': str(path.resolve()), 'sha256': digest, 'bytes': path.stat().st_size}
    source, reference, draft = [gguf.GGUFReader(p) for p in (source_path, reference_path, draft_path)]
    mtp = value(source, 'qwen35.nextn_predict_layers') if 'qwen35.nextn_predict_layers' in source.fields else 0
    if value(source, 'general.architecture') != 'qwen35' or value(source, 'qwen35.block_count') != 64 or mtp:
        raise ValueError('use a standard qwen35 target with 64 trunk layers and no MTP head')
    if value(source, 'qwen35.embedding_length') != 5120 or any(t.name.startswith('blk.64.') for t in source.tensors):
        raise ValueError('unsupported hidden width or unexpected MTP tensors')
    source_tensors = {t.name: t for t in source.tensors}
    if len(source_tensors) != 851 or len(source_tensors) != len(source.tensors):
        raise ValueError('expected 851 unique tensors in the standard source')
    for name in VOCAB_WEIGHTS:
        if name not in source_tensors or list(source_tensors[name].shape) != [5120, len(value(source, 'tokenizer.ggml.tokens'))]:
            raise ValueError(f'invalid source vocabulary tensor: {name}')
    mapping, _ = token_mapping(reference, source, require_mask=False)
    if len(mapping) >= len(value(source, 'tokenizer.ggml.tokens')) or len(mapping) % 4:
        raise ValueError('expected a smaller vocabulary divisible by four for MK scoring')
    if len(mapping) % 128 and source_tensors['output.weight'].tensor_type != gguf.GGMLQuantizationType.Q4_K:
        raise ValueError('unaligned condensed MK head requires Q4_K')
    if source_license:
        sources['license'] = {'path': str(Path(source_license).resolve()), 'sha256': file_hash(source_license)}
    if value(reference, 'tokenizer.chat_template') != value(source, 'tokenizer.chat_template'):
        raise ValueError('reference chat template differs; preserve and validate the source template first')
    # Every tokenizer field copied from the reference must have known semantics.
    arrays = {'tokenizer.ggml.tokens', 'tokenizer.ggml.token_type', 'tokenizer.ggml.merges'}
    if any(key.startswith('tokenizer.') and key not in reference.fields for key in source.fields):
        raise ValueError('reference omits source tokenizer metadata')
    for key, field in reference.fields.items():
        if key.startswith('tokenizer.') and key not in arrays and not key.endswith('_token_id'):
            if key not in source.fields or value(source, key) != field.contents():
                raise ValueError(f'reference changes non-vocabulary tokenizer metadata: {key}')
    mask = verify_draft(reference, draft)
    if schema_validator:
        subprocess.run([str(Path(schema_validator).resolve()), str(source_path.resolve())], check=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    name = source_path.stem + '-ASCII-Condensed-noMTP.gguf'
    target_path = output_dir / name
    print(f'Gathering {len(mapping):,} of {len(value(source, "tokenizer.ggml.tokens")):,} vocabulary rows; copying the source trunk', flush=True)
    target_report = export_gguf(source, target_path, tokenizer_from=reference, gather=mapping,
        gather_names=VOCAB_WEIGHTS, overrides={
            'general.name': value(source, 'general.name') + ' ASCII Condensed (no MTP)',
            'qwen35.nextn_predict_layers': 0,
            'tokenizer.chat_template': value(source, 'tokenizer.chat_template')})
    written = gguf.GGUFReader(target_path)
    print('Checking every retained embedding/output row and all non-vocabulary tensors against the source', flush=True)
    verify_source(written, source, mapping)
    # Preserve license, attribution, architecture and all other source metadata.
    changed = {'general.name', 'qwen35.nextn_predict_layers'}
    for key, field in source.fields.items():
        if not key.startswith(('GGUF.', 'tokenizer.')) and key not in changed:
            if value(written, key) != field.contents():
                raise ValueError(f'export changed source metadata: {key}')
    verify_draft(written, draft)
    if schema_validator:
        subprocess.run([str(Path(schema_validator).resolve()), str(target_path.resolve())], check=True)
    # Copy the already adapted draft. It is the same token space, not a Swift-trained draft.
    draft_output = output_dir / draft_path.name
    with draft_path.open('rb') as src, draft_output.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=8 << 20)
    if file_hash(draft_output) != sources['condensed_draft']['sha256']:
        raise ValueError('copied draft checksum differs')
    if source_license:
        shutil.copyfile(source_license, output_dir / 'LICENSE')
        if file_hash(output_dir / 'LICENSE') != sources['license']['sha256']:
            raise ValueError('copied license checksum differs')
    np.save(output_dir / 'new-to-original.npy', np.asarray(mapping, dtype=np.int32), allow_pickle=False)
    target_tensors = {t.name: t for t in written.tensors}
    weight_saving = sum(source_tensors[n].n_bytes - target_tensors[n].n_bytes for n in VOCAB_WEIGHTS)
    report = {'operation': 'reuse_verified_vocabulary_subset', 'sources': sources,
        'target': target_report, 'draft': {'path': str(draft_output.resolve()),
            'sha256': sources['condensed_draft']['sha256'], 'bytes': draft_output.stat().st_size},
        'original_vocab': len(value(source, 'tokenizer.ggml.tokens')), 'new_vocab': len(mapping),
        'mask_token_id': mask, 'mtp_layers': 0, 'mtp_payload_removed_bytes': 0,
        'target_vocab_weight_saving_bytes': weight_saving,
        'file_saving_bytes': source_path.stat().st_size - target_path.stat().st_size,
        'untouched_non_vocabulary_tensors': len(written.tensors) - len(VOCAB_WEIGHTS),
        'retained_rows_verified_per_vocab_tensor': len(mapping),
        'all_tensor_types_unchanged': True, 'all_other_source_metadata_preserved': True,
        'no_requantization_or_finetuning': True,
        'draft_retrained_for_source': False,
        'quality_note': 'Pruning changes tokenization and the target distribution; quality must be measured separately.'}
    (output_dir / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    (output_dir / 'SHA256SUMS').write_text(target_report['sha256'] + '  ' + target_path.name + '\n' +
        sources['condensed_draft']['sha256'] + '  ' + draft_output.name + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-target', type=Path, required=True)
    ap.add_argument('--tokenizer-reference', type=Path, required=True)
    ap.add_argument('--draft', type=Path, required=True, help='already adapted draft matching the retained vocabulary')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--llama-root', type=Path, required=True)
    ap.add_argument('--expected-source-sha256')
    ap.add_argument('--expected-reference-sha256')
    ap.add_argument('--expected-draft-sha256')
    ap.add_argument('--schema-validator', type=Path, help='optional built model_schema_test binary; validates before and after export')
    ap.add_argument('--source-license', type=Path, help='copy the original weight LICENSE into the output bundle')
    args = ap.parse_args()
    sys.path.insert(0, str(args.llama_root.resolve() / 'gguf-py'))
    condense(args.source_target, args.tokenizer_reference, args.draft, args.output_dir,
             args.expected_source_sha256, args.expected_reference_sha256,
             args.expected_draft_sha256, args.schema_validator, args.source_license)


if __name__ == '__main__':
    main()
