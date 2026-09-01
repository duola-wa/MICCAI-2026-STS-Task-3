"""Data loading, patient-level record aggregation, and CBCT view caching."""

from __future__ import annotations

import csv
import html
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .segmentation import (
    ADULT_FDI_ORDER,
    TOOTH_CACHE_FORMAT_VERSION,
    TOOTH_QUALITY_DIM,
    tooth_cache_path_for_case,
)


CSV_NAMES = {
    "Train-Labeled": "Train-Labeled.csv",
    "Train-Unlabeled": "Train-Unlabeled.csv",
}

CSV_COLUMNS = [
    "Filename",
    "Sex",
    "Age",
    "Main appeal",
    "Subsequent",
    "Present medical history",
    "Past medical history",
    "Oral Check",
    "Diagnosis",
    "Treatment plan",
    "Handle",
    "Doctor advices",
]

TEXT_COLUMNS = CSV_COLUMNS[3:]
TARGET_TEXT_COLUMNS = [
    "Oral Check",
    "Diagnosis",
    "Treatment plan",
    "Handle",
    "Doctor advices",
]


def canonical_case_id(value: Any) -> str:
    """Normalize a CSV/file case identifier without changing non-numeric IDs."""
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return text


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _join_unique(values: Iterable[Any], separator: str = " [VISIT] ") -> str:
    output: List[str] = []
    seen = set()
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return separator.join(output)


@dataclass
class CaseRecord:
    case_id: str
    split: str
    image_path: str
    sex: str = ""
    age: float = float("nan")
    fields: Optional[Dict[str, str]] = None
    report: str = ""
    num_visits: int = 0

    def __post_init__(self) -> None:
        self.case_id = canonical_case_id(self.case_id)
        if self.fields is None:
            self.fields = {column: "" for column in TEXT_COLUMNS}

    @property
    def target_text(self) -> str:
        assert self.fields is not None
        parts = []
        for column in TARGET_TEXT_COLUMNS:
            text = self.fields.get(column, "")
            if text:
                parts.append("{}: {}".format(column, text))
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if isinstance(payload.get("age"), float) and math.isnan(payload["age"]):
            payload["age"] = None
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CaseRecord":
        data = dict(payload)
        if data.get("age") is None:
            data["age"] = float("nan")
        return cls(**data)


def discover_image_paths(data_root: Path, split: str) -> Dict[str, Path]:
    split_dir = Path(data_root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError("Missing split directory: {}".format(split_dir))

    image_paths: Dict[str, Path] = {}
    for path in sorted(split_dir.glob("*/*.nii.gz"), key=lambda item: canonical_case_id(item.parent.name)):
        case_id = canonical_case_id(path.parent.name)
        if canonical_case_id(path.name[:-7]) != case_id:
            raise ValueError("Folder/image ID mismatch: {}".format(path))
        if case_id in image_paths:
            raise ValueError("Duplicate image for case {} in {}".format(case_id, split))
        image_paths[case_id] = path
    if not image_paths:
        raise FileNotFoundError("No .nii.gz cases found under {}".format(split_dir))
    return image_paths


def _build_report(rows: pd.DataFrame) -> str:
    visits: List[str] = []
    for visit_index, (_, row) in enumerate(rows.iterrows(), start=1):
        sections = []
        for column in TEXT_COLUMNS:
            value = clean_text(row.get(column, ""))
            if value:
                sections.append("{}: {}".format(column, value))
        if sections:
            visits.append("Visit {}. {}".format(visit_index, " ".join(sections)))
    return " ".join(visits)


def load_split_records(data_root: Path, split: str) -> List[CaseRecord]:
    """Load one split, aggregating all visit rows by Filename."""
    data_root = Path(data_root)
    image_paths = discover_image_paths(data_root, split)
    csv_name = CSV_NAMES.get(split)
    if csv_name is None:
        return [
            CaseRecord(case_id=case_id, split=split, image_path=str(path.resolve()))
            for case_id, path in image_paths.items()
        ]

    csv_path = data_root / split / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError("Missing record CSV: {}".format(csv_path))
    frame = pd.read_csv(csv_path, dtype={"Filename": str}, keep_default_na=False)
    missing_columns = [column for column in CSV_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError("{} is missing columns: {}".format(csv_path, missing_columns))
    frame["Filename"] = frame["Filename"].map(canonical_case_id)

    records: List[CaseRecord] = []
    csv_case_ids = set()
    for case_id, rows in frame.groupby("Filename", sort=False):
        case_id = canonical_case_id(case_id)
        csv_case_ids.add(case_id)
        if case_id not in image_paths:
            raise ValueError("CSV case {} has no image in {}".format(case_id, split))

        fields = {column: _join_unique(rows[column].tolist()) for column in TEXT_COLUMNS}
        sex = _join_unique(rows["Sex"].tolist(), separator=" ").lower()
        ages = pd.to_numeric(rows["Age"], errors="coerce").dropna()
        age = float(ages.iloc[0]) if len(ages) else float("nan")
        records.append(
            CaseRecord(
                case_id=case_id,
                split=split,
                image_path=str(image_paths[case_id].resolve()),
                sex=sex,
                age=age,
                fields=fields,
                report=_build_report(rows),
                num_visits=int(len(rows)),
            )
        )

    image_only = sorted(set(image_paths) - csv_case_ids)
    if image_only:
        raise ValueError("Images without CSV records in {}: {}".format(split, image_only[:10]))
    return records


def load_supervised_records(data_root: Path, use_unlabeled_records: bool = False) -> List[CaseRecord]:
    records = load_split_records(data_root, "Train-Labeled")
    if use_unlabeled_records:
        records.extend(load_split_records(data_root, "Train-Unlabeled"))
    return records


def cache_path_for_case(cache_dir: Path, record: CaseRecord) -> Path:
    return Path(cache_dir) / record.split / "{}.npy".format(record.case_id)


def _sample_centers(length: int, num_slices: int, margin_fraction: float) -> np.ndarray:
    if num_slices < 1:
        raise ValueError("num_slices must be positive")
    margin = int(round(length * margin_fraction))
    low = max(0, margin)
    high = min(length - 1, length - 1 - margin)
    if high <= low:
        low, high = 0, length - 1
    return np.rint(np.linspace(low, high, num_slices)).astype(np.int64)


def _extract_view_stack(
    volume_dhw: np.ndarray,
    axis: int,
    num_slices: int,
    image_size: int,
    neighbor_offset: int,
    margin_fraction: float,
    window_min: float,
    window_max: float,
) -> np.ndarray:
    centers = _sample_centers(volume_dhw.shape[axis], num_slices, margin_fraction)
    output = np.empty((num_slices, 3, image_size, image_size), dtype=np.float16)
    offsets = (-neighbor_offset, 0, neighbor_offset)
    scale = max(window_max - window_min, 1.0)

    for slice_index, center in enumerate(centers):
        channels = []
        for offset in offsets:
            index = int(np.clip(center + offset, 0, volume_dhw.shape[axis] - 1))
            plane = np.take(volume_dhw, index, axis=axis).astype(np.float32, copy=False)
            np.clip(plane, window_min, window_max, out=plane)
            plane = (plane - window_min) / scale
            channels.append(plane)
        stack = np.ascontiguousarray(np.stack(channels, axis=0))
        tensor = torch.from_numpy(stack).unsqueeze(0)
        source_height, source_width = tensor.shape[-2:]
        scale_factor = float(image_size) / float(max(source_height, source_width))
        resized_height = max(1, int(round(source_height * scale_factor)))
        resized_width = max(1, int(round(source_width * scale_factor)))
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
        output[slice_index] = resized.squeeze(0).numpy().astype(np.float16)
    return output


def preprocess_case(
    record: CaseRecord,
    cache_dir: Path,
    num_slices: int = 12,
    image_size: int = 224,
    neighbor_offset: int = 2,
    margin_fraction: float = 0.08,
    window_min: float = -1000.0,
    window_max: float = 3000.0,
    overwrite: bool = False,
) -> Path:
    """Convert one 3D CBCT into cached axial/coronal/sagittal 2.5D views."""
    destination = cache_path_for_case(cache_dir, record)
    if destination.is_file() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    image = nib.load(record.image_path)
    if len(image.shape) != 3:
        raise ValueError("Expected 3D NIfTI, got {} for {}".format(image.shape, record.image_path))
    # nibabel returns X,Y,Z. Convert once to D,H,W = Z,Y,X.
    volume_xyz = np.asanyarray(image.dataobj)
    volume_dhw = np.ascontiguousarray(np.transpose(volume_xyz, (2, 1, 0)))
    del volume_xyz

    views = [
        _extract_view_stack(
            volume_dhw,
            axis=axis,
            num_slices=num_slices,
            image_size=image_size,
            neighbor_offset=neighbor_offset,
            margin_fraction=margin_fraction,
            window_min=window_min,
            window_max=window_max,
        )
        for axis in (0, 1, 2)
    ]
    cached = np.stack(views, axis=0)  # [view=3, slice, channel=3, H, W]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, cached, allow_pickle=False)
    os.replace(str(temporary), str(destination))
    return destination


def save_manifest(records: Sequence[CaseRecord], cache_dir: Path, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "split", "image_path", "cache_path", "cache_exists"],
        )
        writer.writeheader()
        for record in records:
            cache_path = cache_path_for_case(cache_dir, record)
            writer.writerow(
                {
                    "case_id": record.case_id,
                    "split": record.split,
                    "image_path": record.image_path,
                    "cache_path": str(cache_path.resolve()),
                    "cache_exists": int(cache_path.is_file()),
                }
            )


def save_case_records(records: Sequence[CaseRecord], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump([record.to_dict() for record in records], handle, ensure_ascii=False, indent=2)


def load_case_records(path: Path) -> List[CaseRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [CaseRecord.from_dict(item) for item in payload]


def augment_cached_views(views: torch.Tensor) -> torch.Tensor:
    """Label-preserving augmentation. Deliberately avoids flips/90-degree rotation."""
    if torch.rand(1).item() < 0.9:
        scale = 0.9 + 0.2 * torch.rand(1).item()
        bias = -0.05 + 0.1 * torch.rand(1).item()
        views = views * scale + bias
    if torch.rand(1).item() < 0.5:
        views = views + torch.randn_like(views) * (0.005 + 0.015 * torch.rand(1).item())
    if torch.rand(1).item() < 0.4:
        shift_y = int(torch.randint(-6, 7, (1,)).item())
        shift_x = int(torch.randint(-6, 7, (1,)).item())
        views = torch.roll(views, shifts=(shift_y, shift_x), dims=(-2, -1))
    if views.shape[1] > 4 and torch.rand(1).item() < 0.3:
        drop_index = int(torch.randint(0, views.shape[1], (1,)).item())
        views[:, drop_index] = 0
    return views.clamp_(0.0, 1.0)


def augment_tooth_views(views: torch.Tensor) -> torch.Tensor:
    """Intensity-only augmentation that preserves tooth identity and geometry."""
    if torch.rand(1).item() < 0.9:
        scale = 0.9 + 0.2 * torch.rand(1).item()
        bias = -0.04 + 0.08 * torch.rand(1).item()
        views = views * scale + bias
    if torch.rand(1).item() < 0.5:
        views = views + torch.randn_like(views) * (0.004 + 0.012 * torch.rand(1).item())
    return views.clamp_(0.0, 1.0)


class CachedViewDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CaseRecord],
        cache_dir: Path,
        label_schema: Optional[Any] = None,
        training: bool = False,
        include_tooth_data: bool = False,
        segmentation_dropout: float = 0.0,
    ) -> None:
        self.records = list(records)
        self.cache_dir = Path(cache_dir)
        self.label_schema = label_schema
        self.training = training
        self.include_tooth_data = bool(include_tooth_data)
        self.segmentation_dropout = float(segmentation_dropout)
        if not 0.0 <= self.segmentation_dropout <= 1.0:
            raise ValueError("segmentation_dropout must be in [0, 1]")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        cache_path = cache_path_for_case(self.cache_dir, record)
        if not cache_path.is_file():
            raise FileNotFoundError(
                "Missing cache {}. Run prepare_data.py first.".format(cache_path)
            )
        array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        views = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
        if views.ndim != 5 or views.shape[0] != 3 or views.shape[2] != 3:
            raise ValueError("Invalid cached view shape {} in {}".format(tuple(views.shape), cache_path))
        if self.training:
            views = augment_cached_views(views)

        item: Dict[str, Any] = {
            "image": views,
            "case_id": record.case_id,
            "record_index": index,
        }
        if self.include_tooth_data:
            tooth_path = tooth_cache_path_for_case(self.cache_dir, record)
            if not tooth_path.is_file():
                raise FileNotFoundError(
                    "Missing tooth cache {}. Run prepare_teeth.py first.".format(tooth_path)
                )
            with np.load(tooth_path, allow_pickle=False) as payload:
                required = {"format_version", "tooth_views", "tooth_quality", "fdi_labels"}
                missing = required.difference(payload.files)
                if missing:
                    raise ValueError(
                        "Tooth cache {} is missing keys {}".format(tooth_path, sorted(missing))
                    )
                version = int(np.asarray(payload["format_version"]).reshape(-1)[0])
                if version != TOOTH_CACHE_FORMAT_VERSION:
                    raise ValueError(
                        "Unsupported tooth cache version {} in {}".format(version, tooth_path)
                    )
                fdi_labels = tuple(int(value) for value in payload["fdi_labels"].tolist())
                if fdi_labels != ADULT_FDI_ORDER:
                    raise ValueError(
                        "Unexpected FDI order in {}: {}".format(tooth_path, fdi_labels)
                    )
                tooth_array = np.array(payload["tooth_views"], dtype=np.float32, copy=True)
                quality_array = np.array(payload["tooth_quality"], dtype=np.float32, copy=True)
            expected_prefix = (len(ADULT_FDI_ORDER), 3, 3)
            if tooth_array.ndim != 5 or tuple(tooth_array.shape[:3]) != expected_prefix:
                raise ValueError(
                    "Invalid tooth-view shape {} in {}".format(tooth_array.shape, tooth_path)
                )
            if quality_array.shape != (len(ADULT_FDI_ORDER), TOOTH_QUALITY_DIM):
                raise ValueError(
                    "Invalid tooth-quality shape {} in {}".format(quality_array.shape, tooth_path)
                )
            tooth_views = torch.from_numpy(tooth_array)
            if self.training:
                tooth_views = augment_tooth_views(tooth_views)
            tooth_quality = torch.from_numpy(quality_array)
            if self.training and torch.rand(1).item() < self.segmentation_dropout:
                # Modality dropout keeps the global path useful and makes a bad
                # or unavailable segmentation an exact quality-gated fallback.
                tooth_views.zero_()
                tooth_quality.zero_()
            item["tooth_views"] = tooth_views
            item["tooth_quality"] = tooth_quality
        if self.label_schema is not None:
            targets = self.label_schema.encode_record(record)
            item["targets"] = {
                name: torch.as_tensor(value)
                for name, value in targets.items()
            }
        return item
