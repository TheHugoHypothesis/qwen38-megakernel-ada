#!/usr/bin/env python3
"""Export an MTP-free target and a DFlash2 draft in its condensed token-ID space.

Uses llama.cpp's gguf-py writer. Every tensor is copied as stored quantized
bytes; only selector vocabulary rows are gathered. Source files are read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


SELECTORS = {"selector_predecessor.weight", "selector_successor.weight"}
VOCAB_WEIGHTS = {"token_embd.weight", "output.weight"}


def value(reader, key):
    if key not in reader.fields:
        raise ValueError(f"missing metadata: {key}")
    return reader.fields[key].contents()


def payload_hash(data):
    return hashlib.sha256(memoryview(data).cast("B")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def token_mapping(target, draft, require_mask=True):
    """Validate a stable vocabulary subset, including types, specials and merges."""
    old = value(draft, "tokenizer.ggml.tokens")
    new = value(target, "tokenizer.ggml.tokens")
    if not 32 <= len(new) <= len(old) <= 248320 or len(new) % 2:
        raise ValueError("MK requires an even vocabulary with 32..248320 rows")
    if len(set(old)) != len(old) or len(set(new)) != len(new):
        raise ValueError("ambiguous duplicate tokens cannot be remapped safely")
    lookup = {token: i for i, token in enumerate(old)}
    try:
        mapping = [lookup[token] for token in new]
    except KeyError as e:
        raise ValueError(f"target contains a token absent from the draft: {e}") from e
    if mapping != sorted(mapping):
        raise ValueError("target token order is not a stable subset of the draft")
    old_types = value(draft, "tokenizer.ggml.token_type")
    new_types = value(target, "tokenizer.ggml.token_type")
    if len(old_types) != len(old) or len(new_types) != len(new):
        raise ValueError("token-type count does not match vocabulary")
    if [old_types[i] for i in mapping] != new_types:
        raise ValueError("retained token types differ")
    kept = set(mapping)
    if any(i not in kept for i, t in enumerate(old_types) if t != 1):
        raise ValueError("condensed target dropped a non-normal/special token")
    for key in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
        if value(target, key) != value(draft, key):
            raise ValueError(f"tokenizer family differs: {key}")
    surviving = set(new)
    original_merges = value(draft, "tokenizer.ggml.merges")
    expected = []
    for merge in original_merges:
        parts = merge.split(" ")
        if len(parts) != 2:
            raise ValueError("malformed BPE merge")
        if parts[0] in surviving and parts[1] in surviving and "".join(parts) in surviving:
            expected.append(merge)
    if value(target, "tokenizer.ggml.merges") != expected:
        raise ValueError("condensed merges do not preserve the filtered BPE priority")
    reverse = {old_id: new_id for new_id, old_id in enumerate(mapping)}
    for key, field in target.fields.items():
        if key.startswith("tokenizer.") and key.endswith("_token_id"):
            if key not in draft.fields:
                raise ValueError(f"cannot verify remapped special metadata: {key}")
            old_id = value(draft, key)
            if old_id not in reverse or value(target, key) != reverse[old_id]:
                raise ValueError(f"special token ID mapping differs: {key}")
    if "tokenizer.ggml.mask_token_id" in draft.fields:
        mask = value(draft, "tokenizer.ggml.mask_token_id")
        if mask not in reverse:
            raise ValueError("draft mask token was dropped")
        return mapping, reverse[mask]
    if require_mask:
        raise ValueError("missing metadata: tokenizer.ggml.mask_token_id")
    return mapping, None


def verify_source(target, original, mapping):
    """Exhaustively check all trunk bytes and every retained vocabulary row."""
    import numpy as np
    src = {t.name: t for t in original.tensors}
    if set(src) != {t.name for t in target.tensors}:
        raise ValueError("condensed/source tensor sets differ")
    if value(original, "tokenizer.ggml.tokens") != value(target, "tokenizer.ggml.tokens"):
        old_tokens = value(original, "tokenizer.ggml.tokens")
        if [old_tokens[i] for i in mapping] != value(target, "tokenizer.ggml.tokens"):
            raise ValueError("source vocabulary differs from the draft's original vocabulary")
    if value(original, "tokenizer.chat_template") != value(target, "tokenizer.chat_template"):
        raise ValueError("condensed chat template differs from source")
    for t in target.tensors:
        old = src[t.name]
        if t.tensor_type != old.tensor_type:
            raise ValueError(f"tensor type changed: {t.name}")
        if t.name in VOCAB_WEIGHTS:
            if int(t.shape[0]) != int(old.shape[0]) or int(t.shape[1]) != len(mapping):
                raise ValueError(f"invalid gathered shape: {t.name}")
            # Bound temporary memory; verify every row, not a random sample.
            for start in range(0, len(mapping), 4096):
                ids = mapping[start:start + 4096]
                if not np.array_equal(t.data[start:start + len(ids)], old.data[ids]):
                    raise ValueError(f"gathered bytes differ: {t.name}, row {start}")
        elif list(t.shape) != list(old.shape) or payload_hash(t.data) != payload_hash(old.data):
            raise ValueError(f"non-vocabulary tensor changed: {t.name}")


def export_gguf(reader, output, overrides=None, tokenizer_from=None, gather=None, drop_mtp=False,
                gather_names=None):
    """Write then verify a temporary GGUF before making the output visible."""
    import gguf
    import numpy as np
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if reader.endianess != gguf.GGUFEndian.LITTLE:
        raise ValueError("only little-endian GGUF is supported")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial", dir=output.parent)
    os.close(fd)
    writer = gguf.GGUFWriter(temp, value(reader, "general.architecture"))
    changes = dict(overrides or {})
    gather_names = SELECTORS if gather_names is None else set(gather_names)
    if not gather_names <= (SELECTORS | VOCAB_WEIGHTS):
        writer.close()
        Path(temp).unlink()
        raise ValueError("only vocabulary-indexed embedding/head/selector rows may be gathered")
    try:
        fields = dict(reader.fields)
        if tokenizer_from is not None:
            fields = {k: f for k, f in fields.items() if not k.startswith("tokenizer.")}
            fields.update({k: f for k, f in tokenizer_from.fields.items() if k.startswith("tokenizer.")})
        for key, field in fields.items():
            if key.startswith("GGUF.") or key == "general.architecture":
                continue
            v = changes.pop(key, field.contents())
            writer.add_key_value(key, v, field.types[0], field.types[-1] if len(field.types) > 1 else None)
        for key, v in changes.items():
            if isinstance(v, str):
                writer.add_string(key, v)
            elif isinstance(v, int):
                writer.add_uint32(key, v)
            else:
                raise ValueError(f"unsupported metadata override: {key}")
        if "general.alignment" in reader.fields:
            writer.data_alignment = int(value(reader, "general.alignment"))
        expected = {}
        removed = []
        for t in reader.tensors:
            if drop_mtp and t.name.startswith("blk.64."):
                removed.append(t.name)
                continue
            data = t.data
            if gather is not None and t.name in gather_names:
                width = 256 if t.name in SELECTORS else 5120
                if list(t.shape) != [width, len(value(reader, "tokenizer.ggml.tokens"))]:
                    raise ValueError(f"tensor is not vocabulary-indexed: {t.name}")
                if t.name in SELECTORS and t.tensor_type not in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K):
                    raise ValueError(f"unsupported selector format: {t.name}")
                data = np.ascontiguousarray(data[gather])
            writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)
            expected[t.name] = (int(t.tensor_type), payload_hash(data))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        written = gguf.GGUFReader(temp)
        if len(written.tensors) != len(expected):
            raise ValueError("exported tensor count differs")
        for t in written.tensors:
            if (int(t.tensor_type), payload_hash(t.data)) != expected[t.name]:
                raise ValueError(f"export changed tensor bytes: {t.name}")
        for key, field in fields.items():
            if key.startswith("tokenizer.") and key not in (overrides or {}):
                if value(written, key) != field.contents():
                    raise ValueError(f"export changed tokenizer metadata: {key}")
        for key, expected_value in (overrides or {}).items():
            if value(written, key) != expected_value:
                raise ValueError(f"export metadata override failed: {key}")
        digest = file_hash(temp)
        # Refuse an output created during the write rather than overwrite it.
        os.link(temp, output)
        os.unlink(temp)
        return {"path": str(output.resolve()), "sha256": digest,
                "bytes": output.stat().st_size, "tensor_count": len(expected), "removed_tensors": removed}
    finally:
        writer.close()
        if Path(temp).exists():
            Path(temp).unlink()


def prepare(target_path, draft_path, output_dir, source_path=None, expected_target_sha256=None):
    import gguf
    import numpy as np
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = gguf.GGUFReader(target_path)
    draft = gguf.GGUFReader(draft_path)
    target_hash = file_hash(target_path)
    if expected_target_sha256 and target_hash != expected_target_sha256:
        raise ValueError("downloaded target SHA-256 does not match the pinned release")
    if value(target, "general.architecture") != "qwen35" or value(draft, "general.architecture") != "dflash":
        raise ValueError("expected qwen35 target and DFlash2 draft")
    if value(target, "qwen35.block_count") != 65 or value(target, "qwen35.nextn_predict_layers") != 1:
        raise ValueError("expected a 64-trunk-layer target with one MTP block")
    if value(target, "qwen35.embedding_length") != 5120 or value(draft, "dflash.embedding_length") != 5120:
        raise ValueError("unsupported target/draft hidden width")
    if value(draft, "dflash.selector_top_k") != 16 or value(draft, "dflash.target_layers") != [6, 20, 34, 48, 62]:
        raise ValueError("unsupported DFlash2 selector/taps")
    if {t.name for t in draft.tensors if t.name in SELECTORS} != SELECTORS:
        raise ValueError("missing DFlash2 selector matrices")
    mapping, mask = token_mapping(target, draft)
    source_hash = None
    if source_path:
        original = gguf.GGUFReader(source_path)
        verify_source(target, original, mapping)
        source_hash = file_hash(source_path)
    target_output = output_dir / "Qwen3.8-27B-GSQ-RCO-ASCII-Condensed-noMTP.gguf"
    draft_output = output_dir / "Qwen3.8-27B-DFlash2-ASCII-Condensed-Q4_K_M.gguf"
    # Check all output names before writing either large file.
    for p in (target_output, draft_output, output_dir / "preparation.json", output_dir / "new-to-original.npy"):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
    target_report = export_gguf(target, target_output, drop_mtp=True, overrides={
        "qwen35.block_count": 64, "qwen35.nextn_predict_layers": 0,
        "general.name": value(target, "general.name") + " ASCII Condensed (no MTP)",
    })
    draft_report = export_gguf(draft, draft_output, tokenizer_from=target, gather=mapping, overrides={
        "tokenizer.ggml.mask_token_id": mask,
        "general.name": value(draft, "general.name") + " ASCII Condensed",
    })
    # Validate the final pair, including remapped special IDs and filtered merges.
    out_target, out_draft = gguf.GGUFReader(target_output), gguf.GGUFReader(draft_output)
    for key in target.fields:
        if key.startswith("tokenizer.") and value(out_target, key) != value(out_draft, key):
            raise ValueError(f"exported target/draft tokenizer mismatch: {key}")
    np.save(output_dir / "new-to-original.npy", np.asarray(mapping, dtype=np.int32), allow_pickle=False)
    report = {"target_source": {"path": str(Path(target_path).resolve()), "sha256": target_hash},
              "draft_source": {"path": str(Path(draft_path).resolve()), "sha256": file_hash(draft_path)},
              "original_target_sha256": source_hash,
              "exhaustive_original_tensor_and_vocab_row_check": bool(source_path),
              "new_vocab": len(mapping), "original_vocab": len(value(draft, "tokenizer.ggml.tokens")),
              "mask_token_id": mask, "target": target_report, "draft": draft_report,
              "removed_mtp_payload_bytes": sum(t.n_bytes for t in target.tensors if t.name.startswith("blk.64.")),
              "checks": "all exported tensor payloads and tokenizer fields verified; no requantization"}
    (output_dir / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-target", type=Path, help="original unpruned GSQ-RCO GGUF for exhaustive byte checks")
    parser.add_argument("--expected-target-sha256")
    parser.add_argument("--llama-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.llama_root.resolve() / "gguf-py"))
    report = prepare(args.target, args.draft, args.output_dir, args.source_target, args.expected_target_sha256)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
