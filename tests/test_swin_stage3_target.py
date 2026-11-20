import os
import unittest
from unittest.mock import patch

from backend.models import Models
from backend.segmentation import swin_stage3_layer_name


class SwinStage3TargetTests(unittest.TestCase):
    def setUp(self):
        loader = patch.object(Models, "load")
        loader.start()
        self.addCleanup(loader.stop)

    def test_stage_output_by_default(self):
        with patch.dict(os.environ):
            os.environ.pop("NEURO_SWIN_STAGE3_TARGET", None)
            models = Models()
        self.assertEqual(models.swin_stage3_target, "stage_output")
        self.assertEqual(swin_stage3_layer_name(models), "layers[2]")

    def test_block_norm1_override(self):
        with patch.dict(os.environ, NEURO_SWIN_STAGE3_TARGET="block_norm1"):
            models = Models()
        self.assertEqual(models.swin_stage3_target, "block_norm1")
        self.assertEqual(swin_stage3_layer_name(models), "layers[2].blocks[-1].norm1")

    def test_invalid_target_identifies_valid_choices(self):
        with patch.dict(os.environ, NEURO_SWIN_STAGE3_TARGET="invalid"):
            with self.assertRaisesRegex(ValueError, "NEURO_SWIN_STAGE3_TARGET.*block_norm1.*stage_output"):
                Models()
