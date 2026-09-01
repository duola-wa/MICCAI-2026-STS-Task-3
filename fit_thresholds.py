"""Fit one decision threshold per entity head from five-fold out-of-fold predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmdental.amp import autocast_context
from mmdental.data import CachedViewDataset, load_supervised_records
from mmdental.engine import move_model_inputs
from mmdental.labels import LabelSchema
from mmdental.model import build_model_from_config
from mmdental.paths import default_cache_dir, default_data_root, default_runs_dir
from mmdental.roi3d import DentalROI3DDataset
from train import split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--schema-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output", type=Path, default=default_runs_dir() / "thresholds.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def micro_f1(probability: np.ndarray, target: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    prediction = probability >= threshold
    truth = target >= 0.5
    valid = mask.astype(bool).reshape(
        (len(mask),) + (1,) * (prediction.ndim - 1)
    )
    prediction = prediction & valid
    truth = truth & valid
    tp = float(np.logical_and(prediction, truth).sum())
    fp = float(np.logical_and(prediction, np.logical_not(truth)).sum())
    fn = float(np.logical_and(np.logical_not(prediction), truth).sum())
    denominator = 2.0 * tp + fp + fn
    return 2.0 * tp / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    schema = LabelSchema.load(args.schema_dir)
    head_names = ["teeth", "diagnosis", "actions", "medications"]
    first_checkpoint = torch.load(args.checkpoints[0], map_location="cpu")
    if bool(first_checkpoint.get("model_config", {}).get("use_tooth_pair_head", False)):
        head_names.append("tooth_diagnosis")
    collected: Dict[str, Dict[str, List[np.ndarray]]] = {
        name: {"probability": [], "target": [], "mask": []} for name in head_names
    }
    seen_folds = set()
    seen_validation_case_keys = set()
    shared_split_config = None
    shared_model_config = None
    expected_case_keys = set(schema.source_case_ids)

    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("schema_signature") != schema.signature():
            raise ValueError("Schema mismatch: {}".format(checkpoint_path))
        saved_args = checkpoint.get("args", {})
        fold = int(saved_args.get("fold", -1))
        if fold < 0:
            raise ValueError("OOF threshold fitting requires fold checkpoints, not fold=-1")
        num_folds = int(saved_args.get("num_folds", 5))
        seed = int(saved_args.get("seed", 42))
        use_unlabeled = bool(saved_args.get("use_unlabeled_records", False))
        limit_cases = int(saved_args.get("limit_cases", 0))
        split_config = (num_folds, seed, use_unlabeled, limit_cases)
        if shared_split_config is None:
            shared_split_config = split_config
        elif split_config != shared_split_config:
            raise ValueError("Fold checkpoints use inconsistent num_folds/seed/data sources")
        if fold in seen_folds:
            raise ValueError("Duplicate fold checkpoint: {}".format(fold))
        seen_folds.add(fold)
        records = load_supervised_records(args.data_root, use_unlabeled)
        if limit_cases > 0:
            records = records[:limit_cases]
        record_by_key = {
            "{}:{}".format(record.split, record.case_id): record
            for record in records
        }
        saved_validation_keys = checkpoint.get("validation_case_keys")
        if saved_validation_keys is not None:
            if len(saved_validation_keys) != len(set(saved_validation_keys)):
                raise ValueError("Checkpoint has duplicate validation case IDs: {}".format(checkpoint_path))
            missing = [key for key in saved_validation_keys if key not in record_by_key]
            if missing:
                raise ValueError(
                    "Checkpoint validation cases are missing from the selected data: {}".format(
                        missing[:5]
                    )
                )
            validation_records = [record_by_key[key] for key in saved_validation_keys]
        else:
            print(
                "WARNING: {} is a legacy checkpoint without saved validation IDs; "
                "reconstructing fold {} from the current environment.".format(checkpoint_path, fold)
            )
            _, validation_records = split_records(records, schema, fold, num_folds, seed)
        validation_case_keys = {
            "{}:{}".format(record.split, record.case_id)
            for record in validation_records
        }
        overlap = seen_validation_case_keys.intersection(validation_case_keys)
        if overlap:
            raise ValueError("OOF validation cases occur in multiple folds: {}".format(sorted(overlap)[:5]))
        seen_validation_case_keys.update(validation_case_keys)
        config = dict(checkpoint["model_config"])
        config["imagenet_pretrained"] = False
        if shared_model_config is None:
            shared_model_config = config
        elif config != shared_model_config:
            raise ValueError("Fold checkpoints use inconsistent model/segmentation configurations")
        architecture = str(config.get("architecture", "multiview_2p5d_v1"))
        use_roi3d = architecture == "dental_roi_3d_v1"
        use_tooth_branch = bool(config.get("use_tooth_branch", False))
        if use_roi3d:
            dataset = DentalROI3DDataset(
                validation_records,
                args.cache_dir,
                schema,
                training=False,
            )
        else:
            dataset = CachedViewDataset(
                validation_records,
                args.cache_dir,
                schema,
                training=False,
                include_tooth_data=use_tooth_branch,
            )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model = build_model_from_config(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        with torch.no_grad():
            for batch in loader:
                image, model_kwargs = move_model_inputs(batch, device)
                with autocast_context(args.amp, device):
                    output = model(image, **model_kwargs)
                for name in head_names:
                    collected[name]["probability"].append(torch.sigmoid(output[name]).cpu().numpy())
                    collected[name]["target"].append(batch["targets"][name].numpy())
                    collected[name]["mask"].append(batch["targets"]["{}_mask".format(name)].numpy())
        print("Collected fold {} from {}".format(fold, checkpoint_path))

    expected_folds = set(range(shared_split_config[0])) if shared_split_config else set()
    if seen_folds != expected_folds:
        raise ValueError(
            "OOF threshold fitting requires every fold exactly once; got {}, expected {}".format(
                sorted(seen_folds), sorted(expected_folds)
            )
        )
    if expected_case_keys and seen_validation_case_keys != expected_case_keys:
        missing = sorted(expected_case_keys - seen_validation_case_keys)
        unexpected = sorted(seen_validation_case_keys - expected_case_keys)
        raise ValueError(
            "OOF validation IDs do not exactly cover the schema cases; missing={}, unexpected={}".format(
                missing[:5], unexpected[:5]
            )
        )

    thresholds: Dict[str, Any] = {}
    candidates = np.linspace(0.1, 0.9, 33)
    for name in head_names:
        probability = np.concatenate(collected[name]["probability"], axis=0)
        target = np.concatenate(collected[name]["target"], axis=0)
        mask = np.concatenate(collected[name]["mask"], axis=0)
        scores = [micro_f1(probability, target, mask, float(value)) for value in candidates]
        best_index = int(np.argmax(scores))
        thresholds[name] = {
            "threshold": round(float(candidates[best_index]), 4),
            "oof_micro_f1": round(float(scores[best_index]), 6),
        }
        print("{} threshold={:.3f} F1={:.4f}".format(name, candidates[best_index], scores[best_index]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print("Wrote {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
