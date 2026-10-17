"""Context workloads must preserve the remapped chat boundary and exact length."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('context_scaling', Path(__file__).parents[1] / 'bench/context_scaling.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class ContextPromptTests(unittest.TestCase):
    def test_original_and_condensed_boundaries(self):
        for end in (248046, 129746):
            source = [10, 11, 12, 20, 21, end, 30, 31]
            ids = bench.make_prompt(source, 128, None, end)
            self.assertEqual(len(ids), 128)
            self.assertEqual(ids[:3], source[:3])
            self.assertEqual(ids[-3:], source[-3:])
            self.assertEqual(ids.count(end), 1)
            self.assertEqual(set(ids[3:-3]), {20, 21})

    def test_header_and_suffix_must_fit(self):
        with self.assertRaisesRegex(ValueError, 'cannot fit'):
            bench.make_prompt([10, 11, 12, 20, 129746, 30, 31], 5, None, 129746)

    def test_plain_corpus_cycles_exactly(self):
        self.assertEqual(bench.make_prompt([1, 2, 3], 7, None, 129746), [1, 2, 3, 1, 2, 3, 1])

    def test_reasoning_cache_restores_stripped_assistant_thinking(self):
        mk_spec = importlib.util.spec_from_file_location('mk_server', Path(__file__).parents[1] / 'server/mk_server.py')
        mk = importlib.util.module_from_spec(mk_spec)
        mk_spec.loader.exec_module(mk)
        cache = {
            ('call', 'call_abc123'): 'Inspecting file header.',
            ('turn', 'Done.', ()): 'Verified output.',
        }
        msgs = [
            {'role': 'user', 'content': 'Read file'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_abc123', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path": "a.py"}'}}
            ]},
            {'role': 'tool', 'tool_call_id': 'call_abc123', 'content': 'print(1)'},
            {'role': 'assistant', 'content': 'Done.'},
            {'role': 'assistant', 'content': 'Explicit', 'reasoning_content': 'Custom'},
        ]
        norm = mk.normalize_messages(msgs, cache)
        self.assertEqual(norm[1].get('reasoning_content'), 'Inspecting file header.')
        self.assertEqual(norm[3].get('reasoning_content'), 'Verified output.')
        self.assertEqual(norm[4].get('reasoning_content'), 'Custom')


if __name__ == '__main__':
    unittest.main()
