"""Byte-preservation and remapping tests with small synthetic GGUF fixtures."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import gguf
import numpy as np

spec = importlib.util.spec_from_file_location("prepare_condensed", Path(__file__).parents[1] / "tools/prepare_condensed.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class CondensedExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tokens = [f"token{i}" for i in range(36)]
        self.tokens[:3] = ["a", "b", "ab"]
        self.tokens[10:13] = ["x", "y", "xy"]
        self.ids = [i for i in range(36) if i not in {10, 11, 12, 13}]
        self.draft = self.fixture("draft", list(range(36)), True)
        self.target = self.fixture("target", self.ids, False)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, name, ids, is_draft, width=256, mtp=True, mixed=False):
        path = self.root / (name + ".gguf")
        w = gguf.GGUFWriter(path, "dflash" if is_draft else "qwen35")
        w.add_string("general.name", name)
        w.add_uint32("qwen35.block_count", 65 if mtp else 64)
        w.add_uint32("qwen35.nextn_predict_layers", int(mtp))
        w.add_string("tokenizer.ggml.model", "gpt2")
        w.add_string("tokenizer.ggml.pre", "qwen35")
        w.add_array("tokenizer.ggml.tokens", [self.tokens[i] for i in ids])
        w.add_array("tokenizer.ggml.token_type", [3 if i >= 33 else 1 for i in ids])
        w.add_array("tokenizer.ggml.merges", ["a b", "x y"] if is_draft else ["a b"])
        w.add_uint32("tokenizer.ggml.bos_token_id", ids.index(34))
        w.add_uint32("tokenizer.ggml.eos_token_id", ids.index(35))
        if is_draft:
            w.add_uint32("tokenizer.ggml.mask_token_id", 33)
        w.add_string("tokenizer.chat_template", "unchanged template")
        for tensor in (prepare.SELECTORS if is_draft else prepare.VOCAB_WEIGHTS):
            kind = gguf.GGMLQuantizationType.IQ2_S if mixed and tensor == 'token_embd.weight' else gguf.GGMLQuantizationType.Q4_K
            row_bytes = width // 256 * (82 if kind == gguf.GGMLQuantizationType.IQ2_S else 144)
            data = np.arange(36 * row_bytes, dtype=np.uint32).astype(np.uint8).reshape(36, row_bytes)[ids]
            w.add_tensor(tensor, data, raw_dtype=kind)
        w.add_tensor("blk.0.attn_norm.weight", np.arange(256, dtype=np.float32))
        if not is_draft and mtp:
            w.add_tensor("blk.64.nextn.enorm.weight", np.arange(256, dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file(progress=False)
        w.close()
        return gguf.GGUFReader(path)

    def test_nonidentity_mapping_and_mask(self):
        mapping, mask = prepare.token_mapping(self.target, self.draft)
        self.assertEqual(mapping, self.ids)
        self.assertEqual(mask, 29)

    def test_export_strips_mtp_and_preserves_every_trunk_byte(self):
        path = self.root / "no-mtp.gguf"
        report = prepare.export_gguf(self.target, path, drop_mtp=True,
            overrides={"qwen35.block_count": 64, "qwen35.nextn_predict_layers": 0})
        out = gguf.GGUFReader(path)
        self.assertFalse(any(t.name.startswith("blk.64.") for t in out.tensors))
        self.assertEqual(prepare.value(out, "qwen35.block_count"), 64)
        old = {t.name: t for t in self.target.tensors}
        for t in out.tensors:
            self.assertTrue(np.array_equal(t.data, old[t.name].data))
        self.assertEqual(len(report["removed_tensors"]), 1)
        with self.assertRaises(FileExistsError):
            prepare.export_gguf(self.target, path)

    def test_selector_rows_and_tokenizer_follow_target_ids(self):
        path = self.root / "adapted-draft.gguf"
        prepare.export_gguf(self.draft, path, tokenizer_from=self.target,
            gather=self.ids, overrides={"tokenizer.ggml.mask_token_id": 29})
        out = gguf.GGUFReader(path)
        old = {t.name: t for t in self.draft.tensors}
        for t in out.tensors:
            expected = old[t.name].data[self.ids] if t.name in prepare.SELECTORS else old[t.name].data
            self.assertTrue(np.array_equal(t.data, expected))
        self.assertEqual(prepare.value(out, "tokenizer.ggml.tokens"), prepare.value(self.target, "tokenizer.ggml.tokens"))
        self.assertEqual(prepare.value(out, "tokenizer.ggml.mask_token_id"), 29)

    def test_changed_token_type_is_rejected(self):
        field = self.target.fields["tokenizer.ggml.token_type"]
        original = field.contents()
        class BadField:
            def contents(self):
                return [3] + original[1:]
        fields = dict(self.target.fields)
        self.target.fields["tokenizer.ggml.token_type"] = BadField()
        with self.assertRaisesRegex(ValueError, "token types differ"):
            prepare.token_mapping(self.target, self.draft)
        self.target.fields = fields

    def test_vocab_rows_preserve_mixed_quantized_source_bytes(self):
        source = self.fixture('refined-source', list(range(36)), False, width=5120, mtp=False, mixed=True)
        path = self.root / 'refined-condensed.gguf'
        report = prepare.export_gguf(source, path, tokenizer_from=self.target,
            gather=self.ids, gather_names=prepare.VOCAB_WEIGHTS)
        written = gguf.GGUFReader(path)
        prepare.verify_source(written, source, self.ids)
        self.assertEqual(report['removed_tensors'], [])
        src = {t.name: t for t in source.tensors}
        for t in written.tensors:
            self.assertEqual(t.tensor_type, src[t.name].tensor_type)
            if t.name in prepare.VOCAB_WEIGHTS:
                self.assertEqual(list(t.shape), [5120, 32])
        self.assertEqual(prepare.value(written, 'qwen35.block_count'), 64)
        self.assertEqual(prepare.value(written, 'qwen35.nextn_predict_layers'), 0)

    def test_maskless_source_token_mapping(self):
        source = self.fixture('source-without-mask', list(range(36)), False)
        # This synthetic source contains both merges; source/target special IDs differ.
        source.fields['tokenizer.ggml.merges'] = self.draft.fields['tokenizer.ggml.merges']
        mapping, mask = prepare.token_mapping(self.target, source, require_mask=False)
        self.assertEqual(mapping, self.ids)
        self.assertIsNone(mask)
        with self.assertRaisesRegex(ValueError, 'mask_token_id'):
            prepare.token_mapping(self.target, source)

    def test_non_vocab_gather_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'vocabulary-indexed'):
            prepare.export_gguf(self.target, self.root / 'bad.gguf', gather=self.ids,
                gather_names={'blk.0.attn_norm.weight'})


if __name__ == "__main__":
    unittest.main()
