"""Prepare true-3D dental-arch ROI caches guided by nnU-Net predictions."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from mmdental.data import load_split_records
from mmdental.paths import default_cache_dir, default_data_root, default_segmentation_dir
from mmdental.roi3d import (
    DEFAULT_ROI3D_SHAPE,
    DENTAL_ROI_CACHE_FORMAT_VERSION,
    dental_roi_cache_path_for_case,
    load_roi3d_cache,
    preprocess_roi3d_case,
)
from mmdental.segmentation import ADULT_FDI_ORDER, segmentation_path_for_case


VALID_SPLITS = ("Train-Labeled", "Train-Unlabeled", "Validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--segmentation-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        # The 3D diagnosis model is supervised.  Unlabeled cases are opt-in so
        # a normal run does not fail when they have no nnU-Net predictions.
        default=["Train-Labeled", "Validation"],
    )
    parser.add_argument(
        "--output-shape",
        nargs=3,
        type=int,
        metavar=("D", "H", "W"),
        default=list(DEFAULT_ROI3D_SHAPE),
    )
    parser.add_argument("--global-margin-mm", type=float, default=16.0)
    parser.add_argument("--tooth-margin-mm", type=float, default=8.0)
    parser.add_argument("--window-min", type=float, default=-1000.0)
    parser.add_argument("--window-max", type=float, default=3000.0)
    parser.add_argument("--min-component-voxels", type=int, default=128)
    parser.add_argument("--affine-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel case processes. Keep this small because CBCT volumes are large.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N cases.")
    return parser.parse_args()


def cache_config(args: argparse.Namespace, segmentation_dir: Path) -> Dict[str, Any]:
    return {
        "format_version": DENTAL_ROI_CACHE_FORMAT_VERSION,
        "source_segmentation_dir": str(segmentation_dir.resolve()),
        "mapping": {
            "nnunet_labels": list(range(1, 33)),
            "fdi_labels": list(ADULT_FDI_ORDER),
        },
        "image_channels": ["windowed_cbct", "binary_cleaned_tooth_mask"],
        "output_shape_dhw": [int(value) for value in args.output_shape],
        "global_margin_mm": float(args.global_margin_mm),
        "tooth_margin_mm": float(args.tooth_margin_mm),
        "window_min": float(args.window_min),
        "window_max": float(args.window_max),
        "min_component_voxels": int(args.min_component_voxels),
        "affine_tolerance": float(args.affine_tolerance),
        "physical_aspect_preserving_letterbox": True,
        "stores_original_cbct_context": True,
        "stores_remapped_nifti": False,
    }


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _worker(record: Any, kwargs: Dict[str, Any]) -> Tuple[str, str, str, int, int]:
    destination = preprocess_roi3d_case(record=record, **kwargs)
    payload = load_roi3d_cache(
        destination,
        expected_output_shape=kwargs["output_shape_dhw"],
    )
    present_teeth = int((payload["tooth_quality"][:, 0] > 0.5).sum())
    fallback_code = int(payload["fallback_code"].reshape(-1)[0])
    return record.split, record.case_id, str(destination.resolve()), present_teeth, fallback_code


def _manifest_row(
    record: Any,
    segmentation_dir: Path,
    result: Tuple[str, str, str, int, int],
) -> Dict[str, Any]:
    _, _, destination, present_teeth, fallback_code = result
    return {
        "case_id": record.case_id,
        "split": record.split,
        "image_path": str(Path(record.image_path).resolve()),
        "segmentation_path": str(
            segmentation_path_for_case(segmentation_dir, record).resolve()
        ),
        "roi3d_cache_path": destination,
        "present_teeth": present_teeth,
        "fallback_code": fallback_code,
    }


def _validate_arguments(args: argparse.Namespace) -> None:
    if len(args.output_shape) != 3 or any(int(value) < 8 for value in args.output_shape):
        raise ValueError("--output-shape must contain three values of at least 8")
    if args.global_margin_mm < 0 or args.tooth_margin_mm < 0:
        raise ValueError("ROI margins must be non-negative")
    if args.window_max <= args.window_min:
        raise ValueError("--window-max must be greater than --window-min")
    if args.min_component_voxels < 1:
        raise ValueError("--min-component-voxels must be positive")
    if args.affine_tolerance < 0:
        raise ValueError("--affine-tolerance must be non-negative")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")


def main() -> None:
    args = parse_args()
    _validate_arguments(args)
    segmentation_dir = (
        args.segmentation_dir
        if args.segmentation_dir is not None
        else default_segmentation_dir(args.data_root)
    )
    segmentation_dir = Path(segmentation_dir)
    if not segmentation_dir.is_dir():
        raise FileNotFoundError("Segmentation directory does not exist: {}".format(segmentation_dir))

    roi_root = Path(args.cache_dir) / "dental_roi_3d"
    roi_root.mkdir(parents=True, exist_ok=True)
    config_path = roi_root / "cache_config.json"
    requested_config = cache_config(args, segmentation_dir)
    if config_path.is_file() and not args.overwrite:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != requested_config:
            raise RuntimeError(
                "3D ROI settings differ from {}. Use --overwrite or a new --cache-dir.".format(
                    config_path
                )
            )

    records: List[Any] = []
    for split in args.splits:
        records.extend(load_split_records(args.data_root, split))
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No cases selected")

    missing_predictions = [
        segmentation_path_for_case(segmentation_dir, record)
        for record in records
        if not segmentation_path_for_case(segmentation_dir, record).is_file()
    ]
    if missing_predictions:
        example = ", ".join(str(path) for path in missing_predictions[:3])
        raise FileNotFoundError(
            "{} of {} selected cases have no nnU-Net prediction. Examples: {}. "
            "Select only available splits with --splits or create those predictions first.".format(
                len(missing_predictions),
                len(records),
                example,
            )
        )

    _atomic_write_json(config_path, requested_config)
    output_shape = tuple(int(value) for value in args.output_shape)
    worker_kwargs = {
        "segmentation_dir": segmentation_dir,
        "cache_dir": Path(args.cache_dir),
        "output_shape_dhw": output_shape,
        "global_margin_mm": float(args.global_margin_mm),
        "tooth_margin_mm": float(args.tooth_margin_mm),
        "window_min": float(args.window_min),
        "window_max": float(args.window_max),
        "min_component_voxels": int(args.min_component_voxels),
        "affine_tolerance": float(args.affine_tolerance),
        "overwrite": bool(args.overwrite),
    }
    print(
        "Preparing {} true-3D dental ROI cases with shape {} using {} worker(s)".format(
            len(records),
            output_shape,
            args.workers,
        )
    )
    print(
        "Original CBCT context is retained around teeth (global margin {:.1f} mm); "
        "no remapped NIfTI will be written.".format(args.global_margin_mm)
    )

    failures: List[Tuple[str, str, str]] = []
    manifest_rows: List[Dict[str, Any]] = []
    start = time.time()
    if args.workers == 1:
        for index, record in enumerate(records, start=1):
            case_start = time.time()
            try:
                result = _worker(record, worker_kwargs)
                row = _manifest_row(record, segmentation_dir, result)
                manifest_rows.append(row)
                print(
                    "[{}/{}] {}:{} teeth={} fallback={} -> {} ({:.1f}s)".format(
                        index,
                        len(records),
                        record.split,
                        record.case_id,
                        row["present_teeth"],
                        row["fallback_code"],
                        row["roi3d_cache_path"],
                        time.time() - case_start,
                    )
                )
            except Exception as error:
                failures.append((record.split, record.case_id, repr(error)))
                print("[ERROR] {}:{} {}".format(record.split, record.case_id, error))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_worker, record, worker_kwargs): record
                for record in records
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                record = futures[future]
                try:
                    result = future.result()
                    row = _manifest_row(record, segmentation_dir, result)
                    manifest_rows.append(row)
                    print(
                        "[{}/{}] {}:{} teeth={} fallback={} -> {}".format(
                            completed,
                            len(records),
                            record.split,
                            record.case_id,
                            row["present_teeth"],
                            row["fallback_code"],
                            row["roi3d_cache_path"],
                        )
                    )
                except Exception as error:
                    failures.append((record.split, record.case_id, repr(error)))
                    print("[ERROR] {}:{} {}".format(record.split, record.case_id, error))

    manifest_rows.sort(key=lambda row: (VALID_SPLITS.index(row["split"]), str(row["case_id"])))
    manifest_path = roi_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "split",
            "image_path",
            "segmentation_path",
            "roi3d_cache_path",
            "present_teeth",
            "fallback_code",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    fallback_counts = {
        code: sum(int(row["fallback_code"]) == code for row in manifest_rows)
        for code in (0, 1, 2)
    }
    print(
        "Finished in {:.1f} minutes; prepared={}, failures={}, "
        "fallbacks(clean/raw/full)={}/{}/{}".format(
            (time.time() - start) / 60.0,
            len(manifest_rows),
            len(failures),
            fallback_counts[0],
            fallback_counts[1],
            fallback_counts[2],
        )
    )
    print("Manifest: {}".format(manifest_path.resolve()))
    if failures:
        raise RuntimeError("3D ROI preprocessing failures: {}".format(failures[:5]))


if __name__ == "__main__":
    main()

