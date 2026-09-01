"""True-3D, nnU-Net-guided dental ROI caching.

The nnU-Net prediction is used as a *locator*, not as a hard image mask.  A
physical-margin crop is taken from the original CBCT so periapical and other
near-tooth findings remain visible.  The crop and a binary tooth-mask channel
are resized together with a physical-aspect-preserving 3D letterbox.

Raw nnU-Net labels are mapped to FDI notation in memory by
``load_fdi_segmentation``.  This module never writes a remapped NIfTI file.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset

from .segmentation import (
    ADULT_FDI_ORDER,
    TOOTH_QUALITY_DIM,
    load_fdi_segmentation,
    segmentation_path_for_case,
)


DENTAL_ROI_CACHE_FORMAT_VERSION = 1
DEFAULT_ROI3D_SHAPE: Tuple[int, int, int] = (96, 160, 160)


def dental_roi_cache_path_for_case(cache_dir: Path, record: Any) -> Path:
    """Return ``cache/dental_roi_3d/<split>/<case>.npz``."""

    return (
        Path(cache_dir)
        / "dental_roi_3d"
        / str(record.split)
        / "{}.npz".format(record.case_id)
    )


def _validate_shape(name: str, values: Sequence[int], minimum: int = 1) -> Tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("{} must contain exactly D H W, got {}".format(name, values))
    shape = tuple(int(value) for value in values)
    if any(value < minimum for value in shape):
        raise ValueError("{} values must be at least {}, got {}".format(name, minimum, shape))
    return shape


def _bounds_from_mask(mask: np.ndarray) -> Optional[Tuple[slice, slice, slice]]:
    coordinates = np.nonzero(mask)
    if not coordinates or coordinates[0].size == 0:
        return None
    return tuple(
        slice(int(axis.min()), int(axis.max()) + 1)
        for axis in coordinates
    )  # type: ignore[return-value]


def _expand_bounds_mm(
    bounds: Sequence[slice],
    shape_xyz: Sequence[int],
    spacing_xyz: Sequence[float],
    margin_mm: float,
) -> Tuple[slice, slice, slice]:
    if margin_mm < 0:
        raise ValueError("Physical margins must be non-negative")
    output = []
    for axis, current in enumerate(bounds):
        spacing = float(spacing_xyz[axis])
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("Invalid voxel spacing {} on axis {}".format(spacing, axis))
        padding = int(math.ceil(float(margin_mm) / spacing))
        output.append(
            slice(
                max(0, int(current.start) - padding),
                min(int(shape_xyz[axis]), int(current.stop) + padding),
            )
        )
    return tuple(output)  # type: ignore[return-value]


def _largest_component(
    fdi_segmentation: np.ndarray,
    fdi: int,
    initial_bounds: Sequence[slice],
) -> Tuple[Optional[np.ndarray], Optional[Tuple[slice, slice, slice]], int, int]:
    """Return the largest local component, its global bounds and voxel counts."""

    local_mask = fdi_segmentation[tuple(initial_bounds)] == int(fdi)
    total_voxels = int(local_mask.sum())
    if total_voxels == 0:
        return None, None, 0, 0
    components, num_components = ndimage.label(
        local_mask,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    if num_components == 1:
        component = local_mask
        largest_voxels = total_voxels
    else:
        sizes = np.bincount(components.reshape(-1))
        sizes[0] = 0
        largest_label = int(np.argmax(sizes))
        largest_voxels = int(sizes[largest_label])
        component = components == largest_label
    local_bounds = _bounds_from_mask(component)
    if local_bounds is None:
        return None, None, 0, total_voxels
    global_bounds = tuple(
        slice(
            int(initial_bounds[axis].start) + int(local_bounds[axis].start),
            int(initial_bounds[axis].start) + int(local_bounds[axis].stop),
        )
        for axis in range(3)
    )
    return component, global_bounds, largest_voxels, total_voxels  # type: ignore[return-value]


def _analyse_segmentation(
    fdi_segmentation: np.ndarray,
    spacing_xyz: Sequence[float],
    global_margin_mm: float,
    tooth_margin_mm: float,
    min_component_voxels: int,
) -> Dict[str, Any]:
    """Clean tooth components and compute global/per-FDI geometry in XYZ."""

    if fdi_segmentation.ndim != 3:
        raise ValueError("FDI segmentation must be 3D, got {}".format(fdi_segmentation.shape))
    if min_component_voxels < 1:
        raise ValueError("min_component_voxels must be positive")
    shape_xyz = tuple(int(value) for value in fdi_segmentation.shape)
    if any(value < 1 for value in shape_xyz):
        raise ValueError("FDI segmentation has an empty dimension: {}".format(shape_xyz))
    spacing_xyz = tuple(float(value) for value in spacing_xyz)
    if len(spacing_xyz) != 3 or any(
        not math.isfinite(value) or value <= 0 for value in spacing_xyz
    ):
        raise ValueError("Invalid XYZ voxel sizes: {}".format(spacing_xyz))

    cleaned_mask = np.zeros(shape_xyz, dtype=np.bool_)
    tooth_quality = np.zeros((len(ADULT_FDI_ORDER), TOOTH_QUALITY_DIM), dtype=np.float32)
    tooth_bounds: list[Optional[Tuple[slice, slice, slice]]] = [
        None for _ in ADULT_FDI_ORDER
    ]
    objects = ndimage.find_objects(fdi_segmentation, max_label=48)
    shape_array = np.asarray(shape_xyz, dtype=np.float32)
    total_image_voxels = max(1, int(np.prod(shape_xyz)))

    for tooth_index, fdi in enumerate(ADULT_FDI_ORDER):
        initial_bounds = objects[fdi - 1] if len(objects) >= fdi else None
        if initial_bounds is None:
            continue
        component, bounds, largest_voxels, total_label_voxels = _largest_component(
            fdi_segmentation,
            fdi,
            initial_bounds,
        )
        if component is None or bounds is None or largest_voxels < min_component_voxels:
            continue

        local_clean = cleaned_mask[tuple(initial_bounds)]
        np.logical_or(local_clean, component, out=local_clean)
        expanded = _expand_bounds_mm(
            bounds,
            shape_xyz,
            spacing_xyz,
            tooth_margin_mm,
        )
        tooth_bounds[tooth_index] = expanded

        extents = np.asarray(
            [int(current.stop) - int(current.start) for current in bounds],
            dtype=np.float32,
        )
        centers = np.asarray(
            [(int(current.start) + int(current.stop) - 1) / 2.0 for current in bounds],
            dtype=np.float32,
        )
        bbox_voxels = max(1, int(np.prod(extents)))
        tooth_quality[tooth_index] = np.asarray(
            [
                1.0,
                math.log1p(largest_voxels) / math.log1p(total_image_voxels),
                np.clip(float(largest_voxels) / float(bbox_voxels), 0.0, 1.0),
                np.clip(
                    float(largest_voxels) / float(max(1, total_label_voxels)),
                    0.0,
                    1.0,
                ),
                *np.clip(extents / shape_array, 0.0, 1.0).tolist(),
                *np.clip(
                    centers / np.maximum(shape_array - 1.0, 1.0),
                    0.0,
                    1.0,
                ).tolist(),
            ],
            dtype=np.float32,
        )

    # Prefer cleaned, plausible components.  With a very poor segmentation,
    # fall back to all predicted tooth voxels; with an empty segmentation, use
    # the full CBCT.  The fallback is explicit in cache metadata.
    crop_source = cleaned_mask
    fallback_code = 0
    if not bool(cleaned_mask.any()):
        raw_mask = fdi_segmentation > 0
        if bool(raw_mask.any()):
            crop_source = raw_mask
            cleaned_mask = raw_mask
            fallback_code = 1  # all labels were below the component threshold
        else:
            fallback_code = 2  # completely empty segmentation

    union_bounds = _bounds_from_mask(crop_source)
    if union_bounds is None:
        union_bounds = tuple(slice(0, size) for size in shape_xyz)
    crop_bounds = _expand_bounds_mm(
        union_bounds,
        shape_xyz,
        spacing_xyz,
        global_margin_mm,
    )
    if any(int(current.stop) <= int(current.start) for current in crop_bounds):
        raise RuntimeError("Computed an empty dental crop: {}".format(crop_bounds))
    return {
        "cleaned_mask": cleaned_mask,
        "crop_bounds_xyz": crop_bounds,
        "tooth_bounds_xyz": tooth_bounds,
        "tooth_quality": tooth_quality,
        "fallback_code": fallback_code,
    }


def _letterbox_3d(
    intensity_dhw: np.ndarray,
    mask_dhw: np.ndarray,
    spacing_dhw: Sequence[float],
    output_shape_dhw: Sequence[int],
) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]:
    """Resize image and mask together while preserving physical 3D aspect."""

    output_shape = _validate_shape("output_shape_dhw", output_shape_dhw, minimum=8)
    if intensity_dhw.ndim != 3 or mask_dhw.ndim != 3:
        raise ValueError("Letterbox inputs must both be 3D")
    if tuple(intensity_dhw.shape) != tuple(mask_dhw.shape):
        raise ValueError(
            "Image/mask crop mismatch: {} versus {}".format(
                intensity_dhw.shape,
                mask_dhw.shape,
            )
        )
    source_shape = np.asarray(intensity_dhw.shape, dtype=np.float64)
    spacing = np.asarray(tuple(float(value) for value in spacing_dhw), dtype=np.float64)
    if spacing.shape != (3,) or not bool(np.isfinite(spacing).all()) or bool((spacing <= 0).any()):
        raise ValueError("Invalid DHW spacing: {}".format(tuple(spacing.tolist())))
    physical_extent = source_shape * spacing
    target = np.asarray(output_shape, dtype=np.float64)
    pixels_per_mm = float(np.min(target / physical_extent))
    resized_shape_array = np.rint(physical_extent * pixels_per_mm).astype(np.int64)
    resized_shape_array = np.clip(resized_shape_array, 1, np.asarray(output_shape))
    resized_shape = tuple(int(value) for value in resized_shape_array)

    image_tensor = torch.from_numpy(
        np.ascontiguousarray(intensity_dhw, dtype=np.float32)
    )[None, None]
    mask_tensor = torch.from_numpy(
        np.ascontiguousarray(mask_dhw, dtype=np.float32)
    )[None, None]
    resized_image = F.interpolate(
        image_tensor,
        size=resized_shape,
        mode="trilinear",
        align_corners=False,
    )
    resized_mask = F.interpolate(mask_tensor, size=resized_shape, mode="nearest")

    pad_total = tuple(output_shape[axis] - resized_shape[axis] for axis in range(3))
    pad_before = tuple(value // 2 for value in pad_total)
    pad_after = tuple(pad_total[axis] - pad_before[axis] for axis in range(3))
    # torch pad order is W-left/right, H-left/right, D-left/right.
    padding = (
        pad_before[2],
        pad_after[2],
        pad_before[1],
        pad_after[1],
        pad_before[0],
        pad_after[0],
    )
    resized_image = F.pad(resized_image, padding, mode="constant", value=0.0)
    resized_mask = F.pad(resized_mask, padding, mode="constant", value=0.0)
    output = torch.cat((resized_image, resized_mask), dim=1)[0]
    if tuple(output.shape) != (2,) + output_shape:
        raise RuntimeError("Unexpected letterbox output shape {}".format(tuple(output.shape)))
    return output.numpy().astype(np.float16), resized_shape, pad_before


def _transform_tooth_bounds(
    tooth_bounds_xyz: Sequence[Optional[Tuple[slice, slice, slice]]],
    crop_bounds_xyz: Sequence[slice],
    source_shape_dhw: Sequence[int],
    resized_shape_dhw: Sequence[int],
    pad_before_dhw: Sequence[int],
    output_shape_dhw: Sequence[int],
) -> np.ndarray:
    """Transform 32 physical-margin boxes to normalized letterbox coordinates."""

    boxes = np.zeros((len(ADULT_FDI_ORDER), 6), dtype=np.float32)
    source_shape = np.asarray(source_shape_dhw, dtype=np.float64)
    resized_shape = np.asarray(resized_shape_dhw, dtype=np.float64)
    pads = np.asarray(pad_before_dhw, dtype=np.float64)
    target = np.asarray(output_shape_dhw, dtype=np.float64)
    factors = resized_shape / source_shape

    for tooth_index, bounds_xyz in enumerate(tooth_bounds_xyz):
        if bounds_xyz is None:
            continue
        # Convert global XYZ slice boundaries to crop-local DHW boundaries.
        local_xyz_start = np.asarray(
            [
                max(int(bounds_xyz[axis].start), int(crop_bounds_xyz[axis].start))
                - int(crop_bounds_xyz[axis].start)
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        local_xyz_stop = np.asarray(
            [
                min(int(bounds_xyz[axis].stop), int(crop_bounds_xyz[axis].stop))
                - int(crop_bounds_xyz[axis].start)
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        start_dhw = local_xyz_start[::-1]
        stop_dhw = local_xyz_stop[::-1]
        if bool((stop_dhw <= start_dhw).any()):
            continue
        start_target = pads + start_dhw * factors
        stop_target = pads + stop_dhw * factors
        start_normalized = np.clip(start_target / target, 0.0, 1.0)
        stop_normalized = np.clip(stop_target / target, 0.0, 1.0)
        if bool((stop_normalized <= start_normalized).any()):
            continue
        boxes[tooth_index] = np.asarray(
            [
                start_normalized[0],
                stop_normalized[0],
                start_normalized[1],
                stop_normalized[1],
                start_normalized[2],
                stop_normalized[2],
            ],
            dtype=np.float32,
        )
    return boxes


def _assemble_roi(
    volume_crop_xyz: np.ndarray,
    mask_crop_xyz: np.ndarray,
    spacing_xyz: Sequence[float],
    crop_bounds_xyz: Sequence[slice],
    tooth_bounds_xyz: Sequence[Optional[Tuple[slice, slice, slice]]],
    tooth_quality: np.ndarray,
    fallback_code: int,
    source_shape_xyz: Sequence[int],
    output_shape_dhw: Sequence[int],
    window_min: float,
    window_max: float,
) -> Dict[str, np.ndarray]:
    if not math.isfinite(window_min) or not math.isfinite(window_max) or window_max <= window_min:
        raise ValueError("window_max must be greater than window_min")
    intensity_xyz = np.asarray(volume_crop_xyz, dtype=np.float32)
    np.nan_to_num(intensity_xyz, copy=False, nan=window_min, posinf=window_max, neginf=window_min)
    np.clip(intensity_xyz, window_min, window_max, out=intensity_xyz)
    intensity_xyz -= float(window_min)
    intensity_xyz /= float(window_max - window_min)
    mask_xyz = np.asarray(mask_crop_xyz, dtype=np.float32)

    # nibabel arrays are XYZ.  The network cache is D,H,W = Z,Y,X.
    intensity_dhw = np.ascontiguousarray(np.transpose(intensity_xyz, (2, 1, 0)))
    mask_dhw = np.ascontiguousarray(np.transpose(mask_xyz, (2, 1, 0)))
    spacing_dhw = tuple(float(value) for value in spacing_xyz)[::-1]
    image, resized_shape, pad_before = _letterbox_3d(
        intensity_dhw,
        mask_dhw,
        spacing_dhw,
        output_shape_dhw,
    )
    tooth_bboxes = _transform_tooth_bounds(
        tooth_bounds_xyz,
        crop_bounds_xyz,
        intensity_dhw.shape,
        resized_shape,
        pad_before,
        output_shape_dhw,
    )
    crop_bounds_array = np.asarray(
        [[int(current.start), int(current.stop)] for current in crop_bounds_xyz],
        dtype=np.int32,
    )
    return {
        "format_version": np.asarray([DENTAL_ROI_CACHE_FORMAT_VERSION], dtype=np.int16),
        "image": image,
        "tooth_bboxes": tooth_bboxes,
        "tooth_quality": np.asarray(tooth_quality, dtype=np.float32),
        "fdi_labels": np.asarray(ADULT_FDI_ORDER, dtype=np.uint8),
        "source_shape_xyz": np.asarray(source_shape_xyz, dtype=np.int32),
        "voxel_sizes_xyz": np.asarray(spacing_xyz, dtype=np.float32),
        "crop_bounds_xyz": crop_bounds_array,
        "output_shape_dhw": np.asarray(output_shape_dhw, dtype=np.int32),
        "resized_shape_dhw": np.asarray(resized_shape, dtype=np.int32),
        "pad_before_dhw": np.asarray(pad_before, dtype=np.int32),
        "fallback_code": np.asarray([fallback_code], dtype=np.uint8),
    }


def extract_dental_roi3d(
    volume_xyz: np.ndarray,
    fdi_segmentation: np.ndarray,
    voxel_sizes_xyz: Sequence[float],
    output_shape_dhw: Sequence[int] = DEFAULT_ROI3D_SHAPE,
    global_margin_mm: float = 16.0,
    tooth_margin_mm: float = 8.0,
    window_min: float = -1000.0,
    window_max: float = 3000.0,
    min_component_voxels: int = 128,
) -> Dict[str, np.ndarray]:
    """Build one 3D ROI payload from in-memory XYZ arrays.

    ``image[0]`` contains the full original CBCT intensities inside the dental
    crop, including non-tooth voxels.  ``image[1]`` is only an attention hint.
    """

    volume_xyz = np.asanyarray(volume_xyz)
    fdi_segmentation = np.asanyarray(fdi_segmentation)
    if volume_xyz.ndim != 3 or fdi_segmentation.ndim != 3:
        raise ValueError("CBCT and FDI segmentation must both be 3D")
    if tuple(volume_xyz.shape) != tuple(fdi_segmentation.shape):
        raise ValueError(
            "CBCT/segmentation mismatch: {} versus {}".format(
                volume_xyz.shape,
                fdi_segmentation.shape,
            )
        )
    analysis = _analyse_segmentation(
        fdi_segmentation,
        voxel_sizes_xyz,
        global_margin_mm,
        tooth_margin_mm,
        min_component_voxels,
    )
    crop_bounds = analysis["crop_bounds_xyz"]
    return _assemble_roi(
        volume_xyz[tuple(crop_bounds)],
        analysis["cleaned_mask"][tuple(crop_bounds)],
        voxel_sizes_xyz,
        crop_bounds,
        analysis["tooth_bounds_xyz"],
        analysis["tooth_quality"],
        analysis["fallback_code"],
        volume_xyz.shape,
        output_shape_dhw,
        window_min,
        window_max,
    )


def validate_roi3d_payload(
    payload: Mapping[str, np.ndarray],
    expected_output_shape: Optional[Sequence[int]] = None,
    source: str = "ROI cache",
) -> None:
    required = {
        "format_version",
        "image",
        "tooth_bboxes",
        "tooth_quality",
        "fdi_labels",
        "source_shape_xyz",
        "voxel_sizes_xyz",
        "crop_bounds_xyz",
        "output_shape_dhw",
        "resized_shape_dhw",
        "pad_before_dhw",
        "fallback_code",
    }
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError("{} is missing keys {}".format(source, sorted(missing)))
    version = int(np.asarray(payload["format_version"]).reshape(-1)[0])
    if version != DENTAL_ROI_CACHE_FORMAT_VERSION:
        raise ValueError("Unsupported ROI cache version {} in {}".format(version, source))
    image = np.asarray(payload["image"])
    output_shape = _validate_shape("cached output_shape_dhw", payload["output_shape_dhw"], 8)
    if expected_output_shape is not None:
        expected = _validate_shape("expected_output_shape", expected_output_shape, 8)
        if output_shape != expected:
            raise ValueError(
                "ROI shape setting mismatch in {}: {} versus expected {}".format(
                    source,
                    output_shape,
                    expected,
                )
            )
    if image.dtype != np.float16 or image.shape != (2,) + output_shape:
        raise ValueError(
            "Invalid ROI image in {}: dtype={}, shape={}".format(source, image.dtype, image.shape)
        )
    if not bool(np.isfinite(image).all()):
        raise ValueError("ROI image contains non-finite values in {}".format(source))
    bboxes = np.asarray(payload["tooth_bboxes"])
    if bboxes.shape != (len(ADULT_FDI_ORDER), 6) or not bool(np.isfinite(bboxes).all()):
        raise ValueError("Invalid tooth_bboxes in {}: {}".format(source, bboxes.shape))
    if bool((bboxes < 0).any()) or bool((bboxes > 1).any()):
        raise ValueError("tooth_bboxes fall outside [0, 1] in {}".format(source))
    quality = np.asarray(payload["tooth_quality"])
    if quality.shape != (len(ADULT_FDI_ORDER), TOOTH_QUALITY_DIM):
        raise ValueError("Invalid tooth_quality in {}: {}".format(source, quality.shape))
    if not bool(np.isfinite(quality).all()):
        raise ValueError("tooth_quality contains non-finite values in {}".format(source))
    fdi_labels = tuple(int(value) for value in np.asarray(payload["fdi_labels"]).tolist())
    if fdi_labels != ADULT_FDI_ORDER:
        raise ValueError("Unexpected FDI order in {}: {}".format(source, fdi_labels))
    crop_bounds = np.asarray(payload["crop_bounds_xyz"])
    source_shape = np.asarray(payload["source_shape_xyz"])
    if crop_bounds.shape != (3, 2) or source_shape.shape != (3,):
        raise ValueError("Invalid crop/source geometry in {}".format(source))
    if bool((crop_bounds[:, 0] < 0).any()) or bool((crop_bounds[:, 1] > source_shape).any()):
        raise ValueError("Crop lies outside the source image in {}".format(source))
    if bool((crop_bounds[:, 1] <= crop_bounds[:, 0]).any()):
        raise ValueError("Empty crop bounds in {}".format(source))


def load_roi3d_cache(
    path: Path,
    expected_output_shape: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    """Load and strictly validate one cache, returning owned NumPy arrays."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Missing 3D dental ROI cache: {}".format(path))
    try:
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    except Exception as error:
        raise ValueError("Could not read ROI cache {}: {}".format(path, error)) from error
    validate_roi3d_payload(payload, expected_output_shape, source=str(path))
    return payload


def _augment_roi3d(image: torch.Tensor) -> torch.Tensor:
    """Intensity-only augmentation; mask geometry and FDI laterality stay fixed."""

    intensity = image[0]
    if torch.rand(1).item() < 0.9:
        scale = 0.9 + 0.2 * torch.rand(1).item()
        bias = -0.04 + 0.08 * torch.rand(1).item()
        intensity = intensity * scale + bias
    if torch.rand(1).item() < 0.5:
        intensity = intensity + torch.randn_like(intensity) * (
            0.003 + 0.012 * torch.rand(1).item()
        )
    image[0] = intensity.clamp_(0.0, 1.0)
    return image


class DentalROI3DDataset(Dataset):
    """Dataset for validated ``dental_roi_3d`` caches.

    Segmentation dropout removes only the locator channel and its geometry.
    The full cropped CBCT intensity remains available as a graceful fallback.
    """

    def __init__(
        self,
        records: Sequence[Any],
        cache_dir: Path,
        label_schema: Optional[Any] = None,
        training: bool = False,
        segmentation_dropout: float = 0.0,
    ) -> None:
        self.records = list(records)
        self.cache_dir = Path(cache_dir)
        self.label_schema = label_schema
        self.training = bool(training)
        self.segmentation_dropout = float(segmentation_dropout)
        if not 0.0 <= self.segmentation_dropout <= 1.0:
            raise ValueError("segmentation_dropout must be in [0, 1]")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        path = dental_roi_cache_path_for_case(self.cache_dir, record)
        payload = load_roi3d_cache(path)
        image = torch.from_numpy(np.asarray(payload["image"], dtype=np.float32))
        tooth_bboxes = torch.from_numpy(
            np.asarray(payload["tooth_bboxes"], dtype=np.float32)
        )
        tooth_quality = torch.from_numpy(
            np.asarray(payload["tooth_quality"], dtype=np.float32)
        )
        if self.training:
            image = _augment_roi3d(image)
            if torch.rand(1).item() < self.segmentation_dropout:
                image[1].zero_()
                tooth_bboxes.zero_()
                tooth_quality.zero_()
        item: Dict[str, Any] = {
            "image": image,
            "tooth_bboxes": tooth_bboxes,
            "tooth_quality": tooth_quality,
            "case_id": record.case_id,
            "record_index": index,
        }
        if self.label_schema is not None:
            targets = self.label_schema.encode_record(record)
            item["targets"] = {
                name: torch.as_tensor(value)
                for name, value in targets.items()
            }
        return item


def preprocess_roi3d_case(
    record: Any,
    segmentation_dir: Path,
    cache_dir: Path,
    output_shape_dhw: Sequence[int] = DEFAULT_ROI3D_SHAPE,
    global_margin_mm: float = 16.0,
    tooth_margin_mm: float = 8.0,
    window_min: float = -1000.0,
    window_max: float = 3000.0,
    min_component_voxels: int = 128,
    affine_tolerance: float = 1e-3,
    overwrite: bool = False,
) -> Path:
    """Atomically build one compact true-3D cache from CBCT and nnU-Net output."""

    output_shape = _validate_shape("output_shape_dhw", output_shape_dhw, minimum=8)
    destination = dental_roi_cache_path_for_case(cache_dir, record)
    if destination.is_file() and not overwrite:
        load_roi3d_cache(destination, expected_output_shape=output_shape)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    image_path = Path(record.image_path)
    if not image_path.is_file():
        raise FileNotFoundError("Missing CBCT image: {}".format(image_path))
    image = nib.load(str(image_path))
    if len(image.shape) != 3:
        raise ValueError("Expected a 3D CBCT, got {} for {}".format(image.shape, image_path))
    voxel_sizes = tuple(float(value) for value in image.header.get_zooms()[:3])
    fdi_segmentation = load_fdi_segmentation(
        segmentation_path_for_case(segmentation_dir, record),
        reference_image=image,
        affine_tolerance=affine_tolerance,
    )
    analysis = _analyse_segmentation(
        fdi_segmentation,
        voxel_sizes,
        global_margin_mm,
        tooth_margin_mm,
        min_component_voxels,
    )
    crop_bounds = analysis["crop_bounds_xyz"]
    # Slice the nibabel proxy so the full CBCT is not materialized when the
    # storage backend supports partial reads.
    volume_crop = np.asanyarray(image.dataobj[tuple(crop_bounds)])
    mask_crop = analysis["cleaned_mask"][tuple(crop_bounds)]
    payload = _assemble_roi(
        volume_crop,
        mask_crop,
        voxel_sizes,
        crop_bounds,
        analysis["tooth_bounds_xyz"],
        analysis["tooth_quality"],
        analysis["fallback_code"],
        image.shape,
        output_shape,
        window_min,
        window_max,
    )
    validate_roi3d_payload(payload, expected_output_shape=output_shape, source=str(destination))

    temporary = destination.with_name(destination.name + ".tmp-{}".format(os.getpid()))
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
