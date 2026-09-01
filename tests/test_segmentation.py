"""Tests for in-memory FDI mapping and the quality-gated tooth branch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmdental.model import MMDentalModel
from mmdental.segmentation import (
    ADULT_FDI_ORDER,
    extract_tooth_views,
    load_fdi_segmentation,
    map_nnunet_labels_to_fdi,
    preprocess_tooth_case,
)


class SegmentationMappingTests(unittest.TestCase):
    def test_complete_nnunet_to_fdi_mapping(self) -> None:
        source = np.arange(33, dtype=np.uint8)
        before = source.copy()
        mapped = map_nnunet_labels_to_fdi(source)
        self.assertEqual(mapped.dtype, np.uint8)
        self.assertEqual(mapped.tolist(), [0] + list(ADULT_FDI_ORDER))
        np.testing.assert_array_equal(source, before)

    def test_invalid_labels_are_rejected(self) -> None:
        for values in (
            np.asarray([-1, 0, 1]),
            np.asarray([0, 32, 33]),
            np.asarray([0.0, 1.5, 2.0]),
            np.asarray([0.0, np.nan]),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    map_nnunet_labels_to_fdi(values)

    def test_tooth_view_shapes_and_presence(self) -> None:
        volume = np.zeros((32, 32, 24), dtype=np.int16)
        volume[4:10, 5:11, 4:12] = 1200
        volume[20:27, 20:28, 10:19] = 1800
        segmentation = np.zeros_like(volume, dtype=np.uint8)
        segmentation[4:10, 5:11, 4:12] = 11
        segmentation[20:27, 20:28, 10:19] = 48
        views, quality = extract_tooth_views(
            volume,
            segmentation,
            voxel_sizes_xyz=(0.25, 0.25, 0.25),
            image_size=16,
            padding_mm=1.0,
            neighbor_offset=1,
            min_component_voxels=8,
        )
        self.assertEqual(views.shape, (32, 3, 3, 16, 16))
        self.assertEqual(quality.shape, (32, 10))
        self.assertGreater(float(views[0].sum()), 0.0)
        self.assertGreater(float(views[31].sum()), 0.0)
        self.assertEqual(int((quality[:, 0] > 0.5).sum()), 2)

    def test_preprocess_does_not_write_a_remapped_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "case.nii.gz"
            segmentation_dir = root / "prediction"
            segmentation_dir.mkdir()
            segmentation_path = segmentation_dir / "7.nii.gz"
            affine = np.eye(4, dtype=np.float64)
            volume = np.zeros((24, 24, 16), dtype=np.int16)
            volume[4:12, 4:12, 3:11] = 1000
            labels = np.zeros_like(volume, dtype=np.uint8)
            labels[4:12, 4:12, 3:11] = 1
            nib.save(nib.Nifti1Image(volume, affine), str(image_path))
            nib.save(nib.Nifti1Image(labels, affine), str(segmentation_path))
            original_mtime = segmentation_path.stat().st_mtime_ns
            record = SimpleNamespace(
                case_id="7",
                split="Train-Labeled",
                image_path=str(image_path),
            )
            destination = preprocess_tooth_case(
                record,
                segmentation_dir=segmentation_dir,
                cache_dir=root / "cache",
                image_size=16,
                padding_mm=1.0,
                min_component_voxels=8,
            )
            self.assertTrue(destination.is_file())
            self.assertEqual(segmentation_path.stat().st_mtime_ns, original_mtime)
            self.assertEqual(list(segmentation_dir.glob("*.nii.gz")), [segmentation_path])
            with np.load(destination, allow_pickle=False) as payload:
                self.assertNotIn("segmentation", payload.files)
                self.assertNotIn("fdi_segmentation", payload.files)
                self.assertEqual(tuple(payload["tooth_views"].shape), (32, 3, 3, 16, 16))
                self.assertEqual(payload["fdi_labels"].tolist(), list(ADULT_FDI_ORDER))

    def test_affine_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.int16), np.eye(4))
            shifted = np.eye(4)
            shifted[0, 3] = 2.0
            path = root / "mask.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.uint8), shifted), str(path))
            with self.assertRaises(ValueError):
                load_fdi_segmentation(path, reference)


class ToothBranchTests(unittest.TestCase):
    def test_zero_quality_is_exact_global_fallback(self) -> None:
        torch.manual_seed(5)
        model = MMDentalModel(
            num_teeth=52,
            num_diagnosis=7,
            num_actions=3,
            num_medications=2,
            text_dim=8,
            token_dim=32,
            num_transformer_layers=1,
            num_attention_heads=4,
            dropout=0.0,
            max_slices=2,
            spatial_pool_size=1,
            imagenet_pretrained=False,
            use_tooth_branch=True,
            tooth_transformer_layers=1,
        ).eval()
        global_views = torch.rand(1, 3, 1, 3, 32, 32)
        tooth_views = torch.rand(1, 32, 3, 3, 16, 16)
        tooth_quality = torch.zeros(1, 32, 10)
        with torch.no_grad():
            global_feature = model.encode(global_views)
            expected = {
                name: head(global_feature)
                for name, head in model.heads.items()
            }
            output = model(
                global_views,
                tooth_views=tooth_views,
                tooth_quality=tooth_quality,
            )
        for name in ("teeth", "diagnosis", "actions", "medications", "sex"):
            torch.testing.assert_close(output[name], expected[name], rtol=0.0, atol=0.0)
        self.assertEqual(float(output["segmentation_quality"].item()), 0.0)

    def test_pair_head_shape_and_batchnorm_are_stable(self) -> None:
        model = MMDentalModel(
            num_teeth=52,
            num_diagnosis=5,
            num_actions=3,
            num_medications=2,
            text_dim=8,
            token_dim=32,
            num_transformer_layers=1,
            num_attention_heads=4,
            dropout=0.0,
            max_slices=2,
            spatial_pool_size=1,
            use_tooth_branch=True,
            use_tooth_pair_head=True,
        )
        model.train()
        batch_norms = [
            module for module in model.backbone.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        ]
        self.assertTrue(batch_norms)
        self.assertTrue(all(not module.training for module in batch_norms))
        with torch.no_grad():
            output = model(
                torch.rand(1, 3, 1, 3, 32, 32),
                tooth_views=torch.rand(1, 32, 3, 3, 16, 16),
                tooth_quality=torch.ones(1, 32, 10),
            )
        self.assertEqual(output["tooth_diagnosis"].shape, (1, 32, 5))

        model.freeze_global_model(True)
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.transformer.training)
        self.assertFalse(model.dropout.training)
        self.assertTrue(model.tooth_transformer.training)
        self.assertTrue(all(
            parameter.requires_grad == name.startswith("tooth_")
            for name, parameter in model.named_parameters()
        ))


if __name__ == "__main__":
    unittest.main()
