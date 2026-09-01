"""Audit structural validity and collapse indicators in Validation predictions."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
from zipfile import ZipFile

from mmdental.data import load_split_records
from mmdental.paths import default_data_root, default_predictions_dir
from mmdental.submission import OFFICIAL_SUBMISSION_FIELDS, build_official_payload


def parse_args() -> argparse.Namespace:
    output_dir = default_predictions_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        default=output_dir / "predictions.jsonl",
    )
    parser.add_argument(
        "--official-json",
        type=Path,
        default=output_dir / "predictions.json",
    )
    parser.add_argument(
        "--submission-zip",
        type=Path,
        default=output_dir / "submission.zip",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("{} line {} is not an object".format(path, line_number))
            rows.append(value)
    return rows


def summarize_counts(predictions: List[Dict[str, Any]], name: str) -> None:
    counts = [len(prediction.get(name) or []) for prediction in predictions]
    print(
        "{} count: min={}, mean={:.2f}, max={}, empty={}/{}".format(
            name,
            min(counts),
            statistics.mean(counts),
            max(counts),
            sum(value == 0 for value in counts),
            len(counts),
        )
    )


def main() -> None:
    args = parse_args()
    errors: List[str] = []
    warnings: List[str] = []
    predictions = load_jsonl(args.predictions_jsonl)
    expected_ids = {
        record.case_id for record in load_split_records(args.data_root, "Validation")
    }
    predicted_ids = [str(prediction.get("case_id", "")) for prediction in predictions]
    duplicate_ids = sorted(
        case_id for case_id, count in Counter(predicted_ids).items() if count > 1
    )
    if duplicate_ids:
        errors.append("duplicate case IDs: {}".format(duplicate_ids[:5]))
    missing_ids = sorted(expected_ids - set(predicted_ids))
    unexpected_ids = sorted(set(predicted_ids) - expected_ids)
    if missing_ids or unexpected_ids:
        errors.append(
            "case ID mismatch; missing={}, unexpected={}".format(
                missing_ids[:5], unexpected_ids[:5]
            )
        )
    print(
        "Cases: predictions={}, unique={}, expected={}".format(
            len(predictions), len(set(predicted_ids)), len(expected_ids)
        )
    )

    try:
        generated_payload = build_official_payload(predictions)
    except ValueError as error:
        errors.append(str(error))
        generated_payload = {}

    official_payload = json.loads(args.official_json.read_text(encoding="utf-8"))
    if official_payload != generated_payload:
        errors.append("predictions.json does not match predictions.jsonl official fields")
    if set(official_payload) != expected_ids:
        errors.append("predictions.json does not cover exactly the Validation IDs")
    for case_id, fields in official_payload.items():
        if tuple(fields) != OFFICIAL_SUBMISSION_FIELDS:
            errors.append("case {} does not have the exact seven official fields".format(case_id))
            break

    with ZipFile(args.submission_zip, "r") as archive:
        members = archive.namelist()
        if members != ["predictions.json"]:
            errors.append("submission.zip members are {}".format(members))
        else:
            zipped_payload = json.loads(archive.read("predictions.json").decode("utf-8"))
            if zipped_payload != official_payload:
                errors.append("ZIP predictions.json differs from the standalone file")
    print("Official fields: {}".format(", ".join(OFFICIAL_SUBMISSION_FIELDS)))
    print("ZIP members: {}".format(members))

    for name in ("tooth_notations", "diagnosis_codes", "treatment_actions", "medications"):
        summarize_counts(predictions, name)

    signatures = Counter(
        (
            tuple(prediction.get("tooth_notations") or []),
            tuple(prediction.get("diagnosis_codes") or []),
            tuple(prediction.get("treatment_actions") or []),
            tuple(prediction.get("medications") or []),
        )
        for prediction in predictions
    )
    top_signature_count = signatures.most_common(1)[0][1]
    print(
        "Entity signatures: unique={}, most_common={}/{}".format(
            len(signatures), top_signature_count, len(predictions)
        )
    )
    if top_signature_count > 0.5 * len(predictions):
        warnings.append("more than half of cases share the same entity prediction")

    nearest = Counter(
        str(prediction.get("nearest_training_case", "")) for prediction in predictions
    )
    top_nearest, top_nearest_count = nearest.most_common(1)[0]
    similarities = [float(prediction.get("retrieval_similarity", 0.0)) for prediction in predictions]
    print(
        "Retrieval: unique_neighbors={}, top_neighbor={} ({}/{}), similarity={:.3f}..{:.3f}".format(
            len(nearest),
            top_nearest,
            top_nearest_count,
            len(predictions),
            min(similarities),
            max(similarities),
        )
    )
    if top_nearest_count > 0.5 * len(predictions):
        warnings.append("retrieval embedding is collapsed onto case {}".format(top_nearest))

    for field in OFFICIAL_SUBMISSION_FIELDS:
        values = [str(official_payload.get(case_id, {}).get(field, "")).strip() for case_id in predicted_ids]
        empty = sum(not value for value in values)
        unique = len(set(values))
        top_count = Counter(values).most_common(1)[0][1]
        print(
            "Field {!r}: empty={}, unique={}, most_common={}/{}".format(
                field, empty, unique, top_count, len(values)
            )
        )
        if empty:
            errors.append("field {!r} has {} empty cases".format(field, empty))
        if top_count > 0.8 * len(values):
            warnings.append("field {!r} has very low diversity".format(field))

    for message in warnings:
        print("[WARNING] {}".format(message))
    for message in errors:
        print("[ERROR] {}".format(message))
    if errors:
        print("PREDICTION AUDIT FAILED: {} error(s), {} warning(s)".format(len(errors), len(warnings)))
        raise SystemExit(1)
    print("PREDICTION AUDIT PASSED WITH {} WARNING(S)".format(len(warnings)))


if __name__ == "__main__":
    main()
