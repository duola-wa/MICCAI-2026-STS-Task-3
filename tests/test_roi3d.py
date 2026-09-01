"""Tests for physical-margin true-3D dental ROI preprocessing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmdental.roi3d import (
    DentalROI3DDataset,
    dental_roi_cache_path_for_case,
    extract_dental_roi3d,
    load_roi3d_cache,
    preprocess_roi3d_case,
)
from mmdental.segmentation import ADULT_FDI_ORDER


class DentalROIExtractionTests(unittest.TestCase):
    def test_physical_margin_letterbox_and_surrounding_lesion_are_preserved(self) -> None:
        shape = (40, 50, 30)  # XYZ
        volume = np.full(shape, -1000, dtype=np.int16)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[10:14, 20:24, 8:12] = 11
        volume[10:14, 20:24, 8:12] = 1000
        # This bright lesion is outside the tooth mask but inside the requested
        # physical-margin crop and therefore must survive in image channel 0.
        volume[14:16, 22:24, 10:12] = 3000

        payload = extract_dental_roi3d(
            volume,
            segmentation,
            voxel_sizes_xyz=(2.0, 1.0, 0.5),
            output_shape_dhw=(24, 32, 40),
            global_margin_mm=4.0,
            tooth_margin_mm=2.0,
            min_component_voxels=8,
        )
        self.assertEqual(payload["image"].dtype, np.float16)
        self.assertEqual(payload["image"].shape, (2, 24, 32, 40))
        np.testing.assert_array_equal(
            payload["crop_bounds_xyz"],
            np.asarray([[8, 16], [16, 28], [0, 20]], dtype=np.int32),
        )
        # Physical crop extent DHW is 10x12x16 mm; one scale is used for all
        # axes and the remainder is letterbox padding.
        np.testing.assert_array_equal(payload["resized_shape_dhw"], [24, 29, 38])
        self.assertEqual(int(payload["fallback_code"][0]), 0)
        self.assertEqual(int((payload["tooth_quality"][:, 0] > 0.5).sum()), 1)
        self.assertGreater(float(payload["tooth_bboxes"][0].min()), 0.0)
        self.assertGreater(
            float(payload["image"][0][payload["image"][1] < 0.5].max()),
            0.9,
        )
        self.assertEqual(payload["fdi_labels"].tolist(), list(ADULT_FDI_ORDER))

    def test_empty_segmentation_falls_back_to_full_cbct(self) -> None:
        volume = np.zeros((12, 14, 10), dtype=np.int16)
        segmentation = np.zeros_like(volume, dtype=np.uint8)
        payload = extract_dental_roi3d(
            volume,
            segmentation,
            voxel_sizes_xyz=(1.0, 1.0, 1.0),
            output_shape_dhw=(16, 16, 16),
            min_component_voxels=8,
        )
        self.assertEqual(int(payload["fallback_code"][0]), 2)
        np.testing.assert_array_equal(payload["crop_bounds_xyz"], [[0, 12], [0, 14], [0, 10]])
        self.assertEqual(float(payload["image"][1].sum()), 0.0)
        self.assertEqual(float(payload["tooth_bboxes"].sum()), 0.0)
        self.assertEqual(float(payload["tooth_quality"].sum()), 0.0)

    def test_subthreshold_labels_use_explicit_raw_mask_fallback(self) -> None:
        volume = np.zeros((20, 20, 20), dtype=np.int16)
        segmentation = np.zeros_like(volume, dtype=np.uint8)
        segmentation[9:11, 9:11, 9:11] = 11
        payload = extract_dental_roi3d(
            volume,
            segmentation,
            voxel_sizes_xyz=(1.0, 1.0, 1.0),
            output_shape_dhw=(16, 16, 16),
            global_margin_mm=2.0,
            min_component_voxels=128,
        )
        self.assertEqual(int(payload["fallback_code"][0]), 1)
        self.assertGreater(float(payload["image"][1].sum()), 0.0)
        self.assertEqual(float(payload["tooth_quality"].sum()), 0.0)


class DentalROICacheTests(unittest.TestCase):
    def test_atomic_preprocess_does_not_modify_or_duplicate_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "case.nii.gz"
            segmentation_dir = root / "prediction"
            segmentation_dir.mkdir()
            segmentation_path = segmentation_dir / "7.nii.gz"
            affine = np.diag([0.5, 0.5, 1.0, 1.0])
            volume = np.full((24, 24, 16), -1000, dtype=np.int16)
            volume[4:12, 4:12, 3:11] = 1000
            labels = np.zeros_like(volume, dtype=np.uint8)
            labels[4:12, 4:12, 3:11] = 1  # raw nnU-Net label -> FDI 11 in RAM
            nib.save(nib.Nifti1Image(volume, affine), str(image_path))
            nib.save(nib.Nifti1Image(labels, affine), str(segmentation_path))
            original_bytes = segmentation_path.read_bytes()
            original_mtime = segmentation_path.stat().st_mtime_ns
            record = SimpleNamespace(
                case_id="7",
                split="Train-Labeled",
                image_path=str(image_path),
            )

            destination = preprocess_roi3d_case(
                record,
                segmentation_dir=segmentation_dir,
                cache_dir=root / "cache",
                output_shape_dhw=(16, 24, 24),
                global_margin_mm=2.0,
                tooth_margin_mm=1.0,
                min_component_voxels=8,
            )
            self.assertEqual(
                destination,
                dental_roi_cache_path_for_case(root / "cache", record),
            )
            self.assertTrue(destination.is_file())
            self.assertFalse(list(destination.parent.glob("*.tmp-*")))
            self.assertEqual(segmentation_path.stat().st_mtime_ns, original_mtime)
            self.assertEqual(segmentation_path.read_bytes(), original_bytes)
            self.assertEqual(list(segmentation_dir.glob("*.nii.gz")), [segmentation_path])

            payload = load_roi3d_cache(destination, expected_output_shape=(16, 24, 24))
            self.assertEqual(payload["image"].shape, (2, 16, 24, 24))
            self.assertEqual(payload["tooth_bboxes"].shape, (32, 6))
            self.assertEqual(payload["tooth_quality"].shape, (32, 10))

            dataset = DentalROI3DDataset(
                [record],
                cache_dir=root / "cache",
                training=True,
                segmentation_dropout=1.0,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["image"].shape), (2, 16, 24, 24))
            self.assertGreater(float(item["image"][0].sum()), 0.0)
            self.assertEqual(float(item["image"][1].sum()), 0.0)
            self.assertEqual(float(item["tooth_bboxes"].sum()), 0.0)
            self.assertEqual(float(item["tooth_quality"].sum()), 0.0)

    def test_corrupt_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.npz"
            np.savez(path, image=np.zeros((2, 8, 8, 8), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, "missing keys"):
                load_roi3d_cache(path)


if __name__ == "__main__":
    unittest.main()

