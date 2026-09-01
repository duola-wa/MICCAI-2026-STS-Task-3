"""Prepare compact segmentation-guided tooth views without writing new masks."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mmdental.data import load_split_records
from mmdental.paths import default_cache_dir, default_data_root, default_segmentation_dir
from mmdental.segmentation import (
    ADULT_FDI_ORDER,
    TOOTH_CACHE_FORMAT_VERSION,
    preprocess_tooth_case,
    segmentation_path_for_case,
    tooth_cache_path_for_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--segmentation-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["Train-Labeled", "Train-Unlabeled", "Validation"],
        default=["Train-Labeled", "Train-Unlabeled", "Validation"],
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--padding-mm", type=float, default=8.0)
    parser.add_argument("--neighbor-offset", type=int, default=2)
    parser.add_argument("--window-min", type=float, default=-1000.0)
    parser.add_argument("--window-max", type=float, default=3000.0)
    parser.add_argument("--min-component-voxels", type=int, default=128)
    parser.add_argument("--affine-tolerance", type=float, default=1e-3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process only N cases.")
    return parser.parse_args()


def cache_config(args: argparse.Namespace, segmentation_dir: Path) -> Dict[str, Any]:
    return {
        "format_version": TOOTH_CACHE_FORMAT_VERSION,
        "source_segmentation_dir": str(segmentation_dir.resolve()),
        "mapping": {
            "nnunet_labels": list(range(1, 33)),
            "fdi_labels": list(ADULT_FDI_ORDER),
        },
        "num_teeth": len(ADULT_FDI_ORDER),
        "num_views": 3,
        "channels": 3,
        "image_size": int(args.image_size),
        "padding_mm": float(args.padding_mm),
        "neighbor_offset": int(args.neighbor_offset),
        "window_min": float(args.window_min),
        "window_max": float(args.window_max),
        "min_component_voxels": int(args.min_component_voxels),
        "affine_tolerance": float(args.affine_tolerance),
        "stores_remapped_segmentation": False,
    }


def main() -> None:
    args = parse_args()
    segmentation_dir = (
        args.segmentation_dir
        if args.segmentation_dir is not None
        else default_segmentation_dir(args.data_root)
    )
    if not segmentation_dir.is_dir():
        raise FileNotFoundError("Segmentation directory does not exist: {}".format(segmentation_dir))

    tooth_root = args.cache_dir / "tooth_views"
    tooth_root.mkdir(parents=True, exist_ok=True)
    config_path = tooth_root / "cache_config.json"
    requested_config = cache_config(args, segmentation_dir)
    if config_path.is_file() and not args.overwrite:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != requested_config:
            raise RuntimeError(
                "Tooth-cache settings differ from {}. Use --overwrite or a new --cache-dir.".format(
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
        str(segmentation_path_for_case(segmentation_dir, record))
        for record in records
        if not segmentation_path_for_case(segmentation_dir, record).is_file()
    ]
    if missing_predictions:
        raise FileNotFoundError(
            "{} nnU-Net predictions are missing. First missing: {}".format(
                len(missing_predictions), missing_predictions[0]
            )
        )

    config_path.write_text(json.dumps(requested_config, indent=2), encoding="utf-8")
    print(
        "Preparing {} segmentation-guided cases from {}".format(
            len(records), segmentation_dir.resolve()
        )
    )
    print("FDI mapping is in memory only; no remapped NIfTI will be written.")
    failures = []
    manifest_rows = []
    start = time.time()
    for index, record in enumerate(records, start=1):
        case_start = time.time()
        try:
            destination = preprocess_tooth_case(
                record=record,
                segmentation_dir=segmentation_dir,
                cache_dir=args.cache_dir,
                image_size=args.image_size,
                padding_mm=args.padding_mm,
                neighbor_offset=args.neighbor_offset,
                window_min=args.window_min,
                window_max=args.window_max,
                min_component_voxels=args.min_component_voxels,
                affine_tolerance=args.affine_tolerance,
                overwrite=args.overwrite,
            )
            with np.load(destination, allow_pickle=False) as payload:
                present_teeth = int((payload["tooth_quality"][:, 0] > 0.5).sum())
            manifest_rows.append(
                {
                    "case_id": record.case_id,
                    "split": record.split,
                    "image_path": record.image_path,
                    "segmentation_path": str(
                        segmentation_path_for_case(segmentation_dir, record).resolve()
                    ),
                    "tooth_cache_path": str(destination.resolve()),
                    "present_teeth": present_teeth,
                }
            )
            print(
                "[{}/{}] {}:{} teeth={} -> {} ({:.1f}s)".format(
                    index,
                    len(records),
                    record.split,
                    record.case_id,
                    present_teeth,
                    destination,
                    time.time() - case_start,
                )
            )
        except Exception as error:
            failures.append((record.split, record.case_id, repr(error)))
            print("[ERROR] {}:{} {}".format(record.split, record.case_id, error))

    manifest_path = tooth_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "split",
            "image_path",
            "segmentation_path",
            "tooth_cache_path",
            "present_teeth",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        "Finished in {:.1f} minutes; prepared={}, failures={}".format(
            (time.time() - start) / 60.0,
            len(manifest_rows),
            len(failures),
        )
    )
    if failures:
        raise RuntimeError("Tooth preprocessing failures: {}".format(failures[:5]))


if __name__ == "__main__":
    main()
