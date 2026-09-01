"""Official Codabench Task 3 JSON/ZIP submission writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple
from zipfile import ZIP_DEFLATED, ZipFile


OFFICIAL_SUBMISSION_FIELDS = (
    "Main appeal",
    "Present medical history",
    "Oral Check",
    "Diagnosis",
    "Treatment plan",
    "Handle",
    "Doctor advices",
)


def build_official_payload(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    payload: Dict[str, Dict[str, str]] = {}
    for prediction in predictions:
        case_id = str(prediction.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("Prediction has an empty case_id")
        if case_id in payload:
            raise ValueError("Duplicate prediction for case {}".format(case_id))
        source_fields = prediction.get("fields") or {}
        fields = {
            name: str(source_fields.get(name, "")).strip()
            for name in OFFICIAL_SUBMISSION_FIELDS
        }
        empty = [name for name, value in fields.items() if not value]
        if empty:
            raise ValueError(
                "Case {} has empty official fields: {}".format(case_id, empty)
            )
        payload[case_id] = fields
    if not payload:
        raise ValueError("No predictions were provided")
    return payload


def write_official_submission(
    predictions: Sequence[Dict[str, Any]], output_dir: Path
) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_official_payload(predictions)
    json_path = output_dir / "predictions.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    zip_path = output_dir / "submission.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="predictions.json")
    with ZipFile(zip_path, "r") as archive:
        members = archive.namelist()
        if members != ["predictions.json"]:
            raise RuntimeError("Invalid submission archive members: {}".format(members))
        archived = json.loads(archive.read("predictions.json").decode("utf-8"))
    if archived != payload:
        raise RuntimeError("Archived predictions.json differs from the generated payload")
    return json_path, zip_path
