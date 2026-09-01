"""Check the Ubuntu/PyTorch runtime and the expected MMDental directory layout."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from mmdental.paths import (
    PROJECT_ROOT,
    default_cache_dir,
    default_data_root,
    default_runs_dir,
    default_segmentation_dir,
)


SPLITS = ("Train-Labeled", "Train-Unlabeled", "Validation")
REQUIRED_IMPORTS = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("nibabel", "nibabel"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("scipy", "scipy"),
)
OPTIONAL_IMPORTS = (("iterstrat", "iterative-stratification"),)


def canonical_case_id(value: str) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--segmentation-dir", type=Path, default=None)
    parser.add_argument("--require-segmentations", action="store_true")
    parser.add_argument(
        "--required-segmentation-splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
    )
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def gibibytes(value: int) -> float:
    return float(value) / (1024.0 ** 3)


def inspect_csv(path: Path) -> Tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    case_ids = {str(row.get("Filename", "")).strip() for row in rows}
    case_ids.discard("")
    return len(rows), len(case_ids)


def print_nvidia_smi() -> None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        print("[WARNING] nvidia-smi was not found on PATH")
        return
    command = [
        executable,
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print("[WARNING] nvidia-smi failed: {}".format(error))
        return
    output = result.stdout.strip() or result.stderr.strip()
    status = "OK" if result.returncode == 0 else "WARNING"
    print("[{}] nvidia-smi: {}".format(status, output or "no output"))


def inspect_required_packages() -> List[str]:
    errors: List[str] = []
    versions = []
    for import_name, distribution_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(import_name)
            versions.append("{}={}".format(distribution_name, package_version(distribution_name)))
        except Exception as error:
            message = "{} import failed: {!r}".format(distribution_name, error)
            errors.append(message)
            print("[ERROR] {}".format(message))
    if versions:
        print("[OK] Python packages: {}".format(", ".join(versions)))
    for import_name, distribution_name in OPTIONAL_IMPORTS:
        try:
            importlib.import_module(import_name)
            print("[OK] Optional package: {}={}".format(distribution_name, package_version(distribution_name)))
        except Exception as error:
            print(
                "[WARNING] Optional {} import failed ({!r}); training will use ordinary KFold".format(
                    distribution_name, error
                )
            )
    return errors


def inspect_torch(require_cuda: bool) -> List[str]:
    errors: List[str] = []
    try:
        import torch
    except Exception as error:
        errors.append("PyTorch import failed: {!r}".format(error))
        print("[ERROR] {}".format(errors[-1]))
        return errors

    try:
        import torchvision
    except Exception as error:
        errors.append("torchvision import failed: {!r}".format(error))
        print("[ERROR] {}".format(errors[-1]))
        return errors

    print("[OK] torch={}, torchvision={}".format(torch.__version__, torchvision.__version__))
    print(
        "[INFO] torch CUDA build={}, cuDNN={}, CUDA available={}".format(
            torch.version.cuda,
            torch.backends.cudnn.version(),
            torch.cuda.is_available(),
        )
    )
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            print(
                "[OK] cuda:{} {} capability={}.{} VRAM={:.1f} GiB".format(
                    index,
                    properties.name,
                    properties.major,
                    properties.minor,
                    gibibytes(properties.total_memory),
                )
            )
        try:
            from mmdental.amp import autocast_context

            left = torch.randn((64, 64), device="cuda")
            right = torch.randn((64, 64), device="cuda")
            with autocast_context(True, "cuda"):
                result = left @ right
            if not bool(torch.isfinite(result).all().item()):
                raise RuntimeError("non-finite CUDA test result")
            torch.cuda.synchronize()
            print("[OK] CUDA tensor and AMP smoke operation passed")
        except Exception as error:
            errors.append("CUDA tensor/AMP operation failed: {!r}".format(error))
            print("[ERROR] {}".format(errors[-1]))
    elif require_cuda:
        errors.append("CUDA is required but torch.cuda.is_available() is False")
        print("[ERROR] {}".format(errors[-1]))
    else:
        print("[WARNING] CUDA is unavailable; full training will be impractically slow")
    return errors


def inspect_data(
    data_root: Path,
    cache_dir: Path,
    segmentation_dir: Path,
    require_segmentations: bool,
    required_segmentation_splits: Sequence[str],
) -> List[str]:
    errors: List[str] = []
    print("[INFO] project root: {}".format(PROJECT_ROOT))
    print("[INFO] data root:    {}".format(data_root))
    print("[INFO] cache dir:    {}".format(cache_dir))
    print("[INFO] segmentation: {}".format(segmentation_dir))
    print("[INFO] runs dir:     {}".format(default_runs_dir()))
    if not data_root.is_dir():
        errors.append("Data root does not exist: {}".format(data_root))
        print("[ERROR] {}".format(errors[-1]))
        return errors

    total_images = 0
    total_caches = 0
    total_tooth_caches = 0
    total_roi3d_caches = 0
    split_case_ids = {}
    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            errors.append("Missing split directory: {}".format(split_dir))
            print("[ERROR] {}".format(errors[-1]))
            continue
        image_count = sum(1 for _ in split_dir.glob("*/*.nii.gz"))
        cache_count = sum(1 for _ in (cache_dir / split).glob("*.npy"))
        tooth_cache_count = sum(1 for _ in (cache_dir / "tooth_views" / split).glob("*.npz"))
        roi3d_cache_count = sum(
            1 for _ in (cache_dir / "dental_roi_3d" / split).glob("*.npz")
        )
        case_ids = {
            canonical_case_id(path.parent.name)
            for path in split_dir.glob("*/*.nii.gz")
        }
        split_case_ids[split] = case_ids
        total_images += image_count
        total_caches += cache_count
        total_tooth_caches += tooth_cache_count
        total_roi3d_caches += roi3d_cache_count
        details = "images={}, caches={}, tooth_caches={}, roi3d_caches={}".format(
            image_count, cache_count, tooth_cache_count, roi3d_cache_count
        )
        csv_path = split_dir / "{}.csv".format(split)
        if csv_path.is_file():
            try:
                rows, cases = inspect_csv(csv_path)
                details += ", csv_rows={}, csv_cases={}".format(rows, cases)
            except Exception as error:
                errors.append("Cannot read {}: {!r}".format(csv_path, error))
        print("[OK] {}: {}".format(split, details))

    expected = {"Train-Labeled": 50, "Train-Unlabeled": 200, "Validation": 50}
    for split, expected_count in expected.items():
        actual = sum(1 for _ in (data_root / split).glob("*/*.nii.gz"))
        if (data_root / split).is_dir() and actual != expected_count:
            print(
                "[WARNING] {} has {} images; the inspected release had {}".format(
                    split, actual, expected_count
                )
            )
    print("[INFO] cache progress: {}/{} cases".format(total_caches, total_images))
    print("[INFO] tooth-cache progress: {}/{} cases".format(total_tooth_caches, total_images))
    print("[INFO] 3D-ROI cache progress: {}/{} cases".format(total_roi3d_caches, total_images))

    all_case_ids = set()
    for split, case_ids in split_case_ids.items():
        overlap = all_case_ids.intersection(case_ids)
        if overlap:
            errors.append(
                "Flat segmentation names are ambiguous because case IDs overlap across splits: {}".format(
                    sorted(overlap)[:10]
                )
            )
        all_case_ids.update(case_ids)
    if not segmentation_dir.is_dir():
        message = "Segmentation directory does not exist: {}".format(segmentation_dir)
        if require_segmentations:
            errors.append(message)
            print("[ERROR] {}".format(message))
        else:
            print("[WARNING] {}".format(message))
    else:
        segmentation_ids = {}
        for path in segmentation_dir.glob("*.nii.gz"):
            case_id = canonical_case_id(path.name[:-7])
            if case_id in segmentation_ids:
                errors.append(
                    "Duplicate canonical segmentation ID {}: {}, {}".format(
                        case_id, segmentation_ids[case_id], path
                    )
                )
            segmentation_ids[case_id] = path
        required_case_ids = set()
        for split in required_segmentation_splits:
            required_case_ids.update(split_case_ids.get(split, set()))
        missing = sorted(required_case_ids.difference(segmentation_ids))
        extra = sorted(set(segmentation_ids).difference(all_case_ids))
        status = "OK" if not missing else "ERROR" if require_segmentations else "WARNING"
        print(
            "[{}] nnU-Net predictions for {}: matched={}/{}, missing={}, extra={}".format(
                status,
                "+".join(required_segmentation_splits),
                len(required_case_ids) - len(missing),
                len(required_case_ids),
                len(missing),
                len(extra),
            )
        )
        if missing:
            message = "Missing segmentation case IDs: {}".format(missing[:10])
            if require_segmentations:
                errors.append(message)
            else:
                print("[WARNING] {}".format(message))
        if extra:
            print("[WARNING] Extra segmentation case IDs: {}".format(extra[:10]))
    return errors


def main() -> None:
    args = parse_args()
    segmentation_dir = (
        args.segmentation_dir.expanduser()
        if args.segmentation_dir is not None
        else default_segmentation_dir(args.data_root.expanduser())
    )
    print("[INFO] Python {}".format(sys.version.replace("\n", " ")))
    print("[INFO] OS {}".format(platform.platform()))
    print_nvidia_smi()
    errors = inspect_required_packages()
    errors.extend(inspect_torch(args.require_cuda))
    errors.extend(
        inspect_data(
            args.data_root.expanduser(),
            args.cache_dir.expanduser(),
            segmentation_dir,
            args.require_segmentations,
            args.required_segmentation_splits,
        )
    )

    disk_target = PROJECT_ROOT if PROJECT_ROOT.exists() else Path.home()
    usage = shutil.disk_usage(str(disk_target))
    print(
        "[INFO] filesystem free={:.1f} GiB, total={:.1f} GiB ({})".format(
            gibibytes(usage.free), gibibytes(usage.total), disk_target
        )
    )
    if errors:
        print("ENVIRONMENT CHECK FAILED: {} error(s)".format(len(errors)))
        raise SystemExit(1)
    print("ENVIRONMENT CHECK PASSED")


if __name__ == "__main__":
    main()
