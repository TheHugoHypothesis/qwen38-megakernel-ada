"""Reject corrupt traces and avoid comparing features after token histories diverge."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


def module(name):
    path=Path(__file__).parents[1]/'bench'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


analyze=module('analyze_feature_trace');agent=module('agent_context_bench')


class DiagnosticsTests(unittest.TestCase):
    def trace(self,directory,changed,corrupt=False):
        base=Path(directory);left=base/'left';right=base/'right'
        x=np.zeros((3,64,5120),np.float32);y=x.copy();y[changed]=.125
        x.tofile(left.with_suffix('.f32'));y.tofile(right.with_suffix('.f32'))
        left.with_suffix('.ids').write_text('10 11 12');right.with_suffix('.ids').write_text('10 99 12')
        prompt=base/'prompt.ids';prompt.write_text('1');out=base/'result.json'
        if corrupt:right.with_suffix('.f32').write_bytes(b'\0')
        args=['analyze',str(left),str(right),'--prompt-ids',str(prompt),'--output',str(out)]
        with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):analyze.main()
        return json.loads(out.read_text())

    def test_changed_token_position_is_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.trace(d,(2,5,7))
            self.assertEqual(r['common_positions_compared'],2)
            self.assertEqual(r['first_output_difference'],1)
            self.assertEqual(r['first_changed_features'],[])

    def test_earlier_feature_difference_is_localized(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.trace(d,(1,5,7));first=r['first_changed_features'][0]
            self.assertEqual((first['position'],first['layer_input'],first['different_elements']),(1,5,1))
            self.assertEqual(first['max_abs'],.125)

    def test_partial_dump_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'truncated'):self.trace(d,(1,5,7),corrupt=True)

    def test_tool_completion_is_distinguished_from_partial_reasoning(self):
        self.assertEqual(agent.classify_output('Thinking. <tool_call><function=read_file>')['output_kind'],'text_or_incomplete_reasoning')
        self.assertEqual(agent.classify_output('<tool_call><function=read_file></function></tool_call>'),
                         {'output_kind':'tool_call','tool_names':['read_file']})


if __name__=='__main__':unittest.main()
