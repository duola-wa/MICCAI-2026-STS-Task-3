"""CPU smoke tests for the true-3-D dental ROI architecture."""

from __future__ import annotations

import unittest

import torch

from mmdental.model import (
    DENTAL_ROI_3D_ARCHITECTURE,
    MMDental3DROIModel,
    build_model_from_config,
    build_model_from_schema,
)


class _Schema:
    tooth_labels = [str(index) for index in range(32)]
    diagnosis_codes = ["D0", "D1", "D2", "D3", "D4"]
    action_labels = ["A0", "A1", "A2", "A3"]
    medication_labels = ["M0", "M1", "M2"]
    text_dim = 8


class Model3DTest(unittest.TestCase):
    def _inputs(self):
        image = torch.rand(1, 2, 32, 32, 32)
        boxes = torch.zeros(1, 32, 6)
        boxes[0, 0] = torch.tensor([0.10, 0.50, 0.10, 0.50, 0.10, 0.50])
        boxes[0, 1] = torch.tensor([0.40, 0.90, 0.40, 0.90, 0.40, 0.90])
        quality = torch.zeros(1, 32, 10)
        quality[0, :2, 0] = 1.0
        quality[0, :2, 2:4] = 0.8
        return image, boxes, quality

    def test_forward_shapes_and_backward(self) -> None:
        model = MMDental3DROIModel(
            num_teeth=32,
            num_diagnosis=5,
            num_actions=4,
            num_medications=3,
            text_dim=8,
            token_dim=32,
            num_transformer_layers=1,
            num_attention_heads=4,
            dropout=0.0,
            base_channels=4,
            roi_pool_size=2,
        )
        image, boxes, quality = self._inputs()
        output = model(image, tooth_bboxes=boxes, tooth_quality=quality)
        self.assertEqual(output["teeth"].shape, (1, 32))
        self.assertEqual(output["diagnosis"].shape, (1, 5))
        self.assertEqual(output["tooth_diagnosis"].shape, (1, 32, 5))
        self.assertEqual(output["patient_feature"].shape, (1, 32))
        self.assertGreater(float(output["segmentation_quality"][0]), 0.0)
        self.assertTrue(torch.isfinite(output["text_embedding"]).all())
        (output["diagnosis"].mean() + output["tooth_diagnosis"].mean()).backward()
        self.assertIsNotNone(model.backbone.stem[0].weight.grad)

    def test_schema_alias_and_config_roundtrip(self) -> None:
        model = build_model_from_schema(
            _Schema(),
            model_type="dental_roi_3d",
            token_dim=32,
            num_attention_heads=4,
            tooth_transformer_layers=1,
            roi3d_base_channels=4,
            dropout=0.0,
        )
        self.assertIsInstance(model, MMDental3DROIModel)
        self.assertEqual(model.model_config["architecture"], DENTAL_ROI_3D_ARCHITECTURE)
        restored = build_model_from_config(model.model_config)
        self.assertIsInstance(restored, MMDental3DROIModel)
        self.assertEqual(restored.model_config, model.model_config)


if __name__ == "__main__":
    unittest.main()
