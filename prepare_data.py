"""Precompute compact three-plane 2.5D CBCT views for training and inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from mmdental.data import load_split_records, preprocess_case, save_manifest
from mmdental.paths import default_cache_dir, default_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["Train-Labeled", "Train-Unlabeled", "Validation"],
        default=["Train-Labeled", "Train-Unlabeled", "Validation"],
    )
    parser.add_argument("--num-slices", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--neighbor-offset", type=int, default=2)
    parser.add_argument("--margin-fraction", type=float, default=0.08)
    parser.add_argument("--window-min", type=float, default=-1000.0)
    parser.add_argument("--window-max", type=float, default=3000.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process only N cases (smoke test).")
    return parser.parse_args()


def cache_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "num_views": 3,
        "num_slices": args.num_slices,
        "channels": 3,
        "image_size": args.image_size,
        "resize_mode": "letterbox",
        "neighbor_offset": args.neighbor_offset,
        "margin_fraction": args.margin_fraction,
        "window_min": args.window_min,
        "window_max": args.window_max,
    }


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.cache_dir / "cache_config.json"
    requested_config = cache_config(args)
    if config_path.is_file() and not args.overwrite:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != requested_config:
            raise RuntimeError(
                "Cache settings differ from {}. Use a new --cache-dir or --overwrite.".format(config_path)
            )

    records: List[Any] = []
    for split in args.splits:
        records.extend(load_split_records(args.data_root, split))
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No cases selected")

    config_path.write_text(json.dumps(requested_config, indent=2), encoding="utf-8")
    print("Preparing {} cases into {}".format(len(records), args.cache_dir.resolve()))
    failures = []
    start = time.time()
    for index, record in enumerate(records, start=1):
        case_start = time.time()
        try:
            destination = preprocess_case(
                record,
                cache_dir=args.cache_dir,
                num_slices=args.num_slices,
                image_size=args.image_size,
                neighbor_offset=args.neighbor_offset,
                margin_fraction=args.margin_fraction,
                window_min=args.window_min,
                window_max=args.window_max,
                overwrite=args.overwrite,
            )
            print(
                "[{}/{}] {}:{} -> {} ({:.1f}s)".format(
                    index,
                    len(records),
                    record.split,
                    record.case_id,
                    destination,
                    time.time() - case_start,
                )
            )
        except Exception as error:  # continue to report all corrupted cases
            failures.append((record.split, record.case_id, repr(error)))
            print("[ERROR] {}:{} {}".format(record.split, record.case_id, error))

    save_manifest(records, args.cache_dir, args.cache_dir / "manifest.csv")
    print("Finished in {:.1f} minutes; failures={}".format((time.time() - start) / 60.0, len(failures)))
    if failures:
        raise RuntimeError("Preprocessing failures: {}".format(failures[:5]))


if __name__ == "__main__":
    main()
