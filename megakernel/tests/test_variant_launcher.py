"""Prevent unavailable MTP decoding from being selected for standard Swift GGUFs."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('serve_ada', Path(__file__).parents[1] / 'serve_ada.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class VariantLauncherTests(unittest.TestCase):
    def test_standard_target_defaults_to_scalar(self):
        self.assertEqual(launcher.draft_depth('megakernel', None, 0), 0)

    def test_external_draft_does_not_require_mtp(self):
        self.assertEqual(launcher.draft_depth('megakernel-dflash2', None, 0), 7)
        self.assertEqual(launcher.draft_depth('dflash2', None, 0), 7)

    def test_mtp_request_on_standard_target_is_rejected(self):
        for engine, depth in [('mtp', None), ('mtp', 0), ('megakernel', 4)]:
            with self.assertRaisesRegex(ValueError, 'no MTP head'):
                launcher.draft_depth(engine, depth, 0)

    def test_original_mtp_default_is_preserved(self):
        self.assertEqual(launcher.draft_depth('megakernel', None, 1), 4)
        self.assertEqual(launcher.draft_depth('megakernel', 0, 1), 0)


if __name__ == '__main__':
    unittest.main()
