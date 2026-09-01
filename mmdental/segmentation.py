"""In-memory nnU-Net label mapping and segmentation-guided tooth views.

The raw segmentation is never rewritten.  During preprocessing, labels 1..32
are mapped to adult FDI notation in RAM, cleaned per tooth, and used only to
extract compact CBCT crops plus numeric quality features.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


ADULT_FDI_ORDER: Tuple[int, ...] = tuple(
    quadrant * 10 + position
    for quadrant in range(1, 5)
    for position in range(1, 9)
)
NNUNET_LABEL_TO_FDI = np.asarray((0,) + ADULT_FDI_ORDER, dtype=np.uint8)
TOOTH_QUALITY_DIM = 10
TOOTH_CACHE_FORMAT_VERSION = 1


def map_nnunet_labels_to_fdi(labels: np.ndarray) -> np.ndarray:
    """Map semantic labels 1..32 to FDI 11..48 without touching the source file.

    Mapping:
        1..8 -> 11..18, 9..16 -> 21..28,
        17..24 -> 31..38, 25..32 -> 41..48.
    Background 0 remains 0.  The returned uint8 array is newly allocated in
    memory and is not written by this function.
    """

    array = np.asanyarray(labels)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("Segmentation labels must be numeric, got {}".format(array.dtype))
    if np.issubdtype(array.dtype, np.floating):
        if not bool(np.isfinite(array).all()):
            raise ValueError("Segmentation contains NaN or infinite values")
        rounded = np.rint(array)
        if not bool(np.array_equal(array, rounded)):
            raise ValueError("Segmentation contains non-integer values")
        array = rounded
    if array.size:
        minimum = int(array.min())
        maximum = int(array.max())
        if minimum < 0 or maximum > 32:
            raise ValueError(
                "Expected nnU-Net labels in [0, 32], got range [{}, {}]".format(
                    minimum, maximum
                )
            )
    indices = array.astype(np.int16, copy=False)
    return NNUNET_LABEL_TO_FDI[indices]


def segmentation_path_for_case(segmentation_dir: Path, case: Any) -> Path:
    case_id = str(getattr(case, "case_id", case)).strip()
    return Path(segmentation_dir) / "{}.nii.gz".format(case_id)


def tooth_cache_path_for_case(cache_dir: Path, record: Any) -> Path:
    return (
        Path(cache_dir)
        / "tooth_views"
        / str(record.split)
        / "{}.npz".format(record.case_id)
    )


def load_fdi_segmentation(
    segmentation_path: Path,
    reference_image: nib.spatialimages.SpatialImage,
    affine_tolerance: float = 1e-3,
) -> np.ndarray:
    """Load one prediction and return an in-memory FDI array on the CBCT grid."""

    segmentation_path = Path(segmentation_path)
    if not segmentation_path.is_file():
        raise FileNotFoundError("Missing nnU-Net prediction: {}".format(segmentation_path))
    segmentation_image = nib.load(str(segmentation_path))
    if len(segmentation_image.shape) != 3:
        raise ValueError(
            "Expected a 3D segmentation, got {} in {}".format(
                segmentation_image.shape, segmentation_path
            )
        )
    if tuple(segmentation_image.shape) != tuple(reference_image.shape):
        raise ValueError(
            "CBCT/segmentation shape mismatch for {}: {} versus {}".format(
                segmentation_path,
                tuple(reference_image.shape),
                tuple(segmentation_image.shape),
            )
        )
    if not np.allclose(
        np.asarray(segmentation_image.affine),
        np.asarray(reference_image.affine),
        rtol=0.0,
        atol=float(affine_tolerance),
    ):
        raise ValueError(
            "CBCT/segmentation affine mismatch for {}. Refusing an implicit reorientation; "
            "verify the nnU-Net export geometry.".format(segmentation_path)
        )
    raw_labels = np.asanyarray(segmentation_image.dataobj)
    return map_nnunet_labels_to_fdi(raw_labels)


def _letterbox_stack(stack: np.ndarray, image_size: int) -> np.ndarray:
    """Resize a [3,H,W] intensity stack without changing its aspect ratio."""

    tensor = torch.from_numpy(np.ascontiguousarray(stack, dtype=np.float32)).unsqueeze(0)
    source_height, source_width = tensor.shape[-2:]
    scale = float(image_size) / float(max(source_height, source_width))
    resized_height = max(1, int(round(source_height * scale)))
    resized_width = max(1, int(round(source_width * scale)))
    resized = F.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    pad_height = image_size - resized_height
    pad_width = image_size - resized_width
    resized = F.pad(
        resized,
        (
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        ),
        mode="constant",
        value=0.0,
    )
    return resized.squeeze(0).numpy().astype(np.float16)


def _expanded_slices(
    bounds: Sequence[slice],
    shape: Sequence[int],
    voxel_sizes: Sequence[float],
    padding_mm: float,
) -> Tuple[slice, slice, slice]:
    output = []
    for axis, current in enumerate(bounds):
        spacing = max(float(voxel_sizes[axis]), 1e-6)
        padding = int(math.ceil(float(padding_mm) / spacing))
        output.append(
            slice(
                max(0, int(current.start) - padding),
                min(int(shape[axis]), int(current.stop) + padding),
            )
        )
    return tuple(output)  # type: ignore[return-value]


def _central_stack(
    volume_xyz: np.ndarray,
    crop_slices: Sequence[slice],
    center_xyz: Sequence[int],
    axis: int,
    neighbor_offset: int,
    window_min: float,
    window_max: float,
) -> np.ndarray:
    channels = []
    scale = max(float(window_max) - float(window_min), 1.0)
    for offset in (-neighbor_offset, 0, neighbor_offset):
        index = int(np.clip(center_xyz[axis] + offset, 0, volume_xyz.shape[axis] - 1))
        if axis == 0:
            plane = volume_xyz[index, crop_slices[1], crop_slices[2]]
        elif axis == 1:
            plane = volume_xyz[crop_slices[0], index, crop_slices[2]]
        else:
            plane = volume_xyz[crop_slices[0], crop_slices[1], index]
        # Match the orientation used after the global XYZ->ZYX conversion:
        # axial [Y,X], coronal [Z,X], sagittal [Z,Y].
        plane = np.asarray(plane, dtype=np.float32).T
        np.clip(plane, window_min, window_max, out=plane)
        channels.append((plane - float(window_min)) / scale)
    return np.stack(channels, axis=0)


def _largest_component_bounds(
    fdi_segmentation: np.ndarray,
    fdi: int,
    initial_bounds: Sequence[slice],
) -> Tuple[Tuple[slice, slice, slice], int, int]:
    """Return largest-component global bounds, largest size, and total size."""

    local_mask = fdi_segmentation[tuple(initial_bounds)] == int(fdi)
    total_voxels = int(local_mask.sum())
    if total_voxels == 0:
        return tuple(initial_bounds), 0, 0  # type: ignore[return-value]
    components, num_components = ndimage.label(
        local_mask,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    if num_components <= 1:
        return tuple(initial_bounds), total_voxels, total_voxels  # type: ignore[return-value]
    sizes = np.bincount(components.reshape(-1))
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    largest_voxels = int(sizes[largest_label])
    component_objects = ndimage.find_objects(components, max_label=num_components)
    local_bounds = component_objects[largest_label - 1]
    if local_bounds is None:
        return tuple(initial_bounds), 0, total_voxels  # type: ignore[return-value]
    global_bounds = tuple(
        slice(
            int(initial_bounds[axis].start) + int(local_bounds[axis].start),
            int(initial_bounds[axis].start) + int(local_bounds[axis].stop),
        )
        for axis in range(3)
    )
    return global_bounds, largest_voxels, total_voxels  # type: ignore[return-value]


def extract_tooth_views(
    volume_xyz: np.ndarray,
    fdi_segmentation: np.ndarray,
    voxel_sizes_xyz: Sequence[float],
    image_size: int = 96,
    padding_mm: float = 8.0,
    neighbor_offset: int = 2,
    window_min: float = -1000.0,
    window_max: float = 3000.0,
    min_component_voxels: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create 32 FDI-aligned three-plane tooth views and mask-quality features.

    Returns:
        tooth_views: [32, 3 planes, 3 channels, H, W], float16.
        tooth_quality: [32, 10], float32 in approximately [0, 1].

    Quality columns are: presence, normalized log-volume, bounding-box fill,
    largest-component purity, XYZ extent fractions, and XYZ center positions.
    """

    volume_xyz = np.asanyarray(volume_xyz)
    fdi_segmentation = np.asanyarray(fdi_segmentation)
    if volume_xyz.ndim != 3 or fdi_segmentation.ndim != 3:
        raise ValueError("CBCT and segmentation must both be 3D")
    if tuple(volume_xyz.shape) != tuple(fdi_segmentation.shape):
        raise ValueError(
            "CBCT/segmentation array mismatch: {} versus {}".format(
                tuple(volume_xyz.shape), tuple(fdi_segmentation.shape)
            )
        )
    if image_size < 16:
        raise ValueError("image_size must be at least 16")
    if min_component_voxels < 1:
        raise ValueError("min_component_voxels must be positive")

    tooth_views = np.zeros(
        (len(ADULT_FDI_ORDER), 3, 3, image_size, image_size),
        dtype=np.float16,
    )
    tooth_quality = np.zeros(
        (len(ADULT_FDI_ORDER), TOOTH_QUALITY_DIM),
        dtype=np.float32,
    )
    objects = ndimage.find_objects(fdi_segmentation, max_label=48)
    total_image_voxels = max(1, int(np.prod(volume_xyz.shape)))

    for tooth_index, fdi in enumerate(ADULT_FDI_ORDER):
        initial_bounds = objects[fdi - 1] if len(objects) >= fdi else None
        if initial_bounds is None:
            continue
        bounds, largest_voxels, total_label_voxels = _largest_component_bounds(
            fdi_segmentation, fdi, initial_bounds
        )
        if largest_voxels < min_component_voxels:
            continue

        extents = np.asarray(
            [int(current.stop) - int(current.start) for current in bounds],
            dtype=np.float32,
        )
        centers = np.asarray(
            [(int(current.start) + int(current.stop) - 1) / 2.0 for current in bounds],
            dtype=np.float32,
        )
        shape = np.asarray(volume_xyz.shape, dtype=np.float32)
        bbox_voxels = max(1, int(np.prod(extents)))
        fill_ratio = float(largest_voxels) / float(bbox_voxels)
        component_purity = float(largest_voxels) / float(max(1, total_label_voxels))
        tooth_quality[tooth_index] = np.asarray(
            [
                1.0,
                math.log1p(largest_voxels) / math.log1p(total_image_voxels),
                np.clip(fill_ratio, 0.0, 1.0),
                np.clip(component_purity, 0.0, 1.0),
                *np.clip(extents / shape, 0.0, 1.0).tolist(),
                *np.clip(centers / np.maximum(shape - 1.0, 1.0), 0.0, 1.0).tolist(),
            ],
            dtype=np.float32,
        )

        expanded = _expanded_slices(
            bounds,
            volume_xyz.shape,
            voxel_sizes_xyz,
            padding_mm,
        )
        center_indices = [int(round(value)) for value in centers]
        # Match the global branch's semantic order: axial, coronal, sagittal.
        for view_index, axis in enumerate((2, 1, 0)):
            stack = _central_stack(
                volume_xyz,
                expanded,
                center_indices,
                axis,
                neighbor_offset,
                window_min,
                window_max,
            )
            tooth_views[tooth_index, view_index] = _letterbox_stack(stack, image_size)

    return tooth_views, tooth_quality


def preprocess_tooth_case(
    record: Any,
    segmentation_dir: Path,
    cache_dir: Path,
    image_size: int = 96,
    padding_mm: float = 8.0,
    neighbor_offset: int = 2,
    window_min: float = -1000.0,
    window_max: float = 3000.0,
    min_component_voxels: int = 128,
    affine_tolerance: float = 1e-3,
    overwrite: bool = False,
) -> Path:
    """Build one compact tooth-view cache without saving a remapped mask."""

    destination = tooth_cache_path_for_case(cache_dir, record)
    if destination.is_file() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    image = nib.load(str(record.image_path))
    if len(image.shape) != 3:
        raise ValueError("Expected 3D CBCT, got {} for {}".format(image.shape, record.image_path))
    segmentation_path = segmentation_path_for_case(segmentation_dir, record)
    fdi_segmentation = load_fdi_segmentation(
        segmentation_path,
        reference_image=image,
        affine_tolerance=affine_tolerance,
    )
    volume_xyz = np.asanyarray(image.dataobj)
    voxel_sizes = tuple(float(value) for value in image.header.get_zooms()[:3])
    tooth_views, tooth_quality = extract_tooth_views(
        volume_xyz=volume_xyz,
        fdi_segmentation=fdi_segmentation,
        voxel_sizes_xyz=voxel_sizes,
        image_size=image_size,
        padding_mm=padding_mm,
        neighbor_offset=neighbor_offset,
        window_min=window_min,
        window_max=window_max,
        min_component_voxels=min_component_voxels,
    )

    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            format_version=np.asarray([TOOTH_CACHE_FORMAT_VERSION], dtype=np.int16),
            tooth_views=tooth_views,
            tooth_quality=tooth_quality,
            fdi_labels=np.asarray(ADULT_FDI_ORDER, dtype=np.uint8),
        )
    os.replace(str(temporary), str(destination))
    return destination
