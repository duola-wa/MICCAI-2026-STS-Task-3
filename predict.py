"""Run fold-ensemble inference and create diagnostics plus an official JSON/ZIP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmdental.amp import autocast_context
from mmdental.data import CSV_COLUMNS, CachedViewDataset, load_split_records
from mmdental.engine import move_model_inputs
from mmdental.labels import LabelSchema
from mmdental.model import build_model_from_config
from mmdental.paths import (
    default_cache_dir,
    default_data_root,
    default_predictions_dir,
)
from mmdental.reporting import (
    PredictionEntities,
    decode_multilabel,
    fallback_entities_from_record,
    merge_with_retrieved_record,
    retrieve_nearest,
)
from mmdental.roi3d import DentalROI3DDataset
from mmdental.submission import write_official_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--schema-dir", type=Path, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument(
        "--split",
        choices=["Train-Labeled", "Train-Unlabeled", "Validation"],
        default="Validation",
    )
    parser.add_argument("--output-dir", type=Path, default=default_predictions_dir())
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--report-mode", choices=["template", "retrieval", "hybrid"], default="template")
    parser.add_argument("--tooth-threshold", type=float, default=0.45)
    parser.add_argument("--diagnosis-threshold", type=float, default=0.35)
    parser.add_argument("--action-threshold", type=float, default=0.35)
    parser.add_argument("--medication-threshold", type=float, default=0.45)
    parser.add_argument("--tooth-diagnosis-threshold", type=float, default=0.5)
    parser.add_argument("--max-tooth-diagnosis-pairs", type=int, default=16)
    parser.add_argument(
        "--tooth-pair-mode",
        choices=["auto", "off", "on"],
        default="auto",
        help="Inject auxiliary tooth/diagnosis pairs only when OOF F1 is reliable (auto >= 0.15).",
    )
    parser.add_argument("--thresholds-json", type=Path, default=None)
    parser.add_argument(
        "--retrieval-fallback",
        action="store_true",
        help="Experimental: fill empty entities from the nearest training case.",
    )
    parser.add_argument("--limit-cases", type=int, default=0, help="Development only: predict first N cases.")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def resolve_schema_dir(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> Path:
    if args.schema_dir is not None:
        return args.schema_dir
    relative = checkpoint.get("schema_relative_dir")
    if relative:
        candidate = (args.checkpoints[0].resolve().parent / relative).resolve()
        if (candidate / "schema.json").is_file():
            return candidate
    saved = checkpoint.get("schema_dir")
    if saved and (Path(saved) / "schema.json").is_file():
        return Path(saved)
    for path in args.checkpoints[0].resolve().parents:
        candidate = path / "schema"
        if (candidate / "schema.json").is_file():
            return candidate
    raise FileNotFoundError("Could not locate schema. Pass --schema-dir explicitly.")


def load_models(
    checkpoint_paths: Sequence[Path],
    schema: LabelSchema,
    device: torch.device,
) -> List[torch.nn.Module]:
    models = []
    reference_config = None
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("schema_signature") != schema.signature():
            raise ValueError("{} uses a different label schema".format(path))
        config = dict(checkpoint["model_config"])
        config["imagenet_pretrained"] = False
        if reference_config is None:
            reference_config = config
        elif config != reference_config:
            raise ValueError("Ensemble checkpoints have different model configurations")
        model = build_model_from_config(config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        models.append(model)
        print("Loaded {}".format(path))
    return models


def average_outputs(
    models: Sequence[torch.nn.Module],
    images: torch.Tensor,
    amp: bool,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    names = ["teeth", "diagnosis", "actions", "medications", "sex", "age", "text_embedding"]
    accumulated: Dict[str, torch.Tensor] = {}
    model_kwargs = model_kwargs or {}
    for model in models:
        with autocast_context(amp, images.device):
            output = model(images, **model_kwargs)
        if "tooth_diagnosis" in output and "tooth_diagnosis" not in names:
            names.append("tooth_diagnosis")
        for name in names:
            accumulated[name] = accumulated.get(name, torch.zeros_like(output[name])) + output[name]
    for name in names:
        accumulated[name] = accumulated[name] / float(len(models))
    accumulated["text_embedding"] = torch.nn.functional.normalize(
        accumulated["text_embedding"], dim=-1
    )
    return accumulated


def main() -> None:
    args = parse_args()
    tooth_pair_oof_f1 = 0.0
    if args.thresholds_json is not None:
        thresholds = json.loads(args.thresholds_json.read_text(encoding="utf-8"))
        args.tooth_threshold = float(thresholds["teeth"]["threshold"])
        args.diagnosis_threshold = float(thresholds["diagnosis"]["threshold"])
        args.action_threshold = float(thresholds["actions"]["threshold"])
        args.medication_threshold = float(thresholds["medications"]["threshold"])
        if "tooth_diagnosis" in thresholds:
            args.tooth_diagnosis_threshold = float(
                thresholds["tooth_diagnosis"]["threshold"]
            )
            tooth_pair_oof_f1 = float(
                thresholds["tooth_diagnosis"].get("oof_micro_f1", 0.0)
            )
    use_pair_predictions = args.tooth_pair_mode == "on" or (
        args.tooth_pair_mode == "auto" and tooth_pair_oof_f1 >= 0.15
    )
    if args.tooth_pair_mode == "auto":
        print(
            "Auxiliary tooth/diagnosis pair injection: {} (OOF F1={:.4f})".format(
                "enabled" if use_pair_predictions else "disabled",
                tooth_pair_oof_f1,
            )
        )
    device = select_device(args.device)
    first_checkpoint = torch.load(args.checkpoints[0], map_location="cpu")
    schema_dir = resolve_schema_dir(args, first_checkpoint)
    schema = LabelSchema.load(schema_dir)
    models = load_models(args.checkpoints, schema, device)
    architecture = str(models[0].model_config.get("architecture", "multiview_2p5d_v1"))
    use_roi3d = architecture == "dental_roi_3d_v1"
    use_tooth_branch = bool(getattr(models[0], "use_tooth_branch", False))
    if use_roi3d:
        print("Using true-3-D nnU-Net-guided dental-arch ROI checkpoint")
    if use_tooth_branch:
        print("Using FDI segmentation-guided tooth branch from checkpoint configuration")
    retrieval_records, retrieval_embeddings = schema.load_retrieval_bank()
    if args.report_mode != "template" or args.retrieval_fallback:
        print(
            "WARNING: retrieval mode is experimental and may copy clinically incorrect text from another patient."
        )

    records = load_split_records(args.data_root, args.split)
    if args.limit_cases > 0:
        records = records[: args.limit_cases]
    if use_roi3d:
        dataset = DentalROI3DDataset(
            records,
            args.cache_dir,
            label_schema=None,
            training=False,
        )
    else:
        dataset = CachedViewDataset(
            records,
            args.cache_dir,
            label_schema=None,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images, model_kwargs = move_model_inputs(batch, device)
            outputs = average_outputs(
                models,
                images,
                args.amp,
                model_kwargs=model_kwargs,
            )
            probabilities = {
                name: torch.sigmoid(outputs[name]).cpu().numpy()
                for name in ["teeth", "diagnosis", "actions", "medications"]
            }
            pair_probabilities = (
                torch.sigmoid(outputs["tooth_diagnosis"]).cpu().numpy()
                if use_pair_predictions and "tooth_diagnosis" in outputs
                else None
            )
            sex_logits = outputs["sex"].cpu().numpy()
            normalized_age = outputs["age"].cpu().numpy()
            text_embeddings = outputs["text_embedding"].cpu().numpy()

            for row_index, case_id in enumerate(batch["case_id"]):
                nearest, similarity = retrieve_nearest(
                    text_embeddings[row_index], retrieval_records, retrieval_embeddings
                )
                predicted_teeth = decode_multilabel(
                    probabilities["teeth"][row_index],
                    schema.tooth_labels,
                    args.tooth_threshold,
                    ensure_one=False,
                )
                predicted_diagnoses = decode_multilabel(
                    probabilities["diagnosis"][row_index],
                    schema.diagnosis_codes,
                    args.diagnosis_threshold,
                    ensure_one=False,
                )
                diagnosis_pairs: List[Tuple[str, str]] = []
                if pair_probabilities is not None:
                    candidates = []
                    for tooth_index, tooth in enumerate(schema.tooth_labels[:32]):
                        for diagnosis_index, code in enumerate(schema.diagnosis_codes):
                            probability = float(
                                pair_probabilities[row_index, tooth_index, diagnosis_index]
                            )
                            if probability >= args.tooth_diagnosis_threshold:
                                candidates.append((probability, tooth, code))
                    candidates.sort(reverse=True)
                    diagnosis_pairs = [
                        (tooth, code)
                        for _, tooth, code in candidates[: args.max_tooth_diagnosis_pairs]
                    ]
                    predicted_teeth = sorted(
                        set(predicted_teeth).union(tooth for tooth, _ in diagnosis_pairs),
                        key=schema.tooth_labels.index,
                    )
                    predicted_diagnoses = sorted(
                        set(predicted_diagnoses).union(code for _, code in diagnosis_pairs),
                        key=schema.diagnosis_codes.index,
                    )
                entities = PredictionEntities(
                    teeth=predicted_teeth,
                    diagnosis_codes=predicted_diagnoses,
                    actions=decode_multilabel(
                        probabilities["actions"][row_index],
                        schema.action_labels,
                        args.action_threshold,
                        ensure_one=False,
                    ),
                    medications=decode_multilabel(
                        probabilities["medications"][row_index],
                        schema.medication_labels,
                        args.medication_threshold,
                        ensure_one=False,
                    ),
                    sex="female" if int(np.argmax(sex_logits[row_index])) == 1 else "male",
                    age=float(np.clip(normalized_age[row_index] * schema.age_std + schema.age_mean, 5.0, 90.0)),
                    diagnosis_pairs=diagnosis_pairs,
                )
                if args.retrieval_fallback:
                    entities = fallback_entities_from_record(entities, nearest)
                prediction = merge_with_retrieved_record(
                    case_id=case_id,
                    entities=entities,
                    schema=schema,
                    nearest=nearest,
                    mode=args.report_mode,
                )
                prediction["nearest_training_case"] = nearest.case_id
                prediction["retrieval_similarity"] = round(similarity, 6)
                predictions.append(prediction)
                debug_rows.append(
                    {
                        "case_id": case_id,
                        "nearest_training_case": nearest.case_id,
                        "retrieval_similarity": similarity,
                        "num_teeth": len(entities.teeth),
                        "num_diagnoses": len(entities.diagnosis_codes),
                        "num_actions": len(entities.actions),
                        "num_medications": len(entities.medications),
                        "num_tooth_diagnosis_pairs": len(diagnosis_pairs),
                    }
                )
                print("Predicted case {} (nearest={}, similarity={:.3f})".format(case_id, nearest.case_id, similarity))

    jsonl_path = args.output_dir / "predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    csv_path = args.output_dir / "submission.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for prediction in predictions:
            row = {
                "Filename": prediction["case_id"],
                "Sex": prediction["sex"],
                "Age": prediction["age"],
            }
            row.update(prediction["fields"])
            writer.writerow(row)

    debug_path = args.output_dir / "retrieval_debug.csv"
    with debug_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(debug_rows[0].keys()) if debug_rows else ["case_id"])
        writer.writeheader()
        writer.writerows(debug_rows)

    official_json_path, official_zip_path = write_official_submission(
        predictions, args.output_dir
    )
    print("Wrote {} cases to {} and {}".format(len(predictions), jsonl_path, csv_path))
    print("Official Codabench JSON: {}".format(official_json_path))
    print("Official Codabench ZIP: {}".format(official_zip_path))
    print("Upload submission.zip, not submission.csv or predictions.jsonl.")


if __name__ == "__main__":
    main()
