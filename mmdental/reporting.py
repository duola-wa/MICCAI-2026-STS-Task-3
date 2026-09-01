"""Entity decoding, nearest-record retrieval, and conservative report rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import CaseRecord, TEXT_COLUMNS
from .labels import FDI_TOKEN_PATTERN, ICD_PATTERN, LabelSchema, extract_actions, extract_diagnosis_codes, extract_medications, extract_tooth_notations


@dataclass
class PredictionEntities:
    teeth: List[str]
    diagnosis_codes: List[str]
    actions: List[str]
    medications: List[str]
    sex: str
    age: float
    diagnosis_pairs: Optional[List[Tuple[str, str]]] = None


def decode_multilabel(
    probabilities: np.ndarray,
    labels: Sequence[str],
    threshold: float,
    ensure_one: bool = False,
) -> List[str]:
    probabilities = np.asarray(probabilities).reshape(-1)
    selected = [label for label, probability in zip(labels, probabilities) if probability >= threshold]
    if ensure_one and not selected and len(labels):
        selected = [labels[int(np.argmax(probabilities))]]
    return selected


def retrieve_nearest(
    predicted_embedding: np.ndarray,
    retrieval_records: Sequence[CaseRecord],
    retrieval_embeddings: np.ndarray,
) -> Tuple[CaseRecord, float]:
    vector = np.asarray(predicted_embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        vector = vector / norm
    scores = np.asarray(retrieval_embeddings, dtype=np.float32).dot(vector)
    index = int(np.argmax(scores))
    return retrieval_records[index], float(scores[index])


def _clean_retrieved_text(text: str) -> str:
    text = ICD_PATTERN.sub("", text or "")
    text = re.sub(r"\*\s*" + FDI_TOKEN_PATTERN + r"(?!\d)", "", text)
    text = re.sub(
        r"\b(?:tooth|teeth|FDI)\s*(?:number|no\.?|#)?\s*" + FDI_TOKEN_PATTERN + r"(?!\d)",
        "the involved tooth",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    return text


def _diagnosis_text(entities: PredictionEntities, schema: LabelSchema) -> str:
    if not entities.diagnosis_codes:
        return "No diagnosis code predicted."
    pieces = []
    paired_codes = set()
    grouped_pairs: Dict[str, List[str]] = {}
    for tooth, code in entities.diagnosis_pairs or []:
        grouped_pairs.setdefault(code, []).append(tooth)
        paired_codes.add(code)
    for code, teeth in grouped_pairs.items():
        name = schema.diagnosis_names.get(code, code)
        label = "{} ({})".format(name, code) if name != code else code
        pieces.append("Teeth {}: {}".format(", ".join(teeth), label))
    for code in entities.diagnosis_codes:
        if code in paired_codes:
            continue
        name = schema.diagnosis_names.get(code, code)
        pieces.append("{} ({})".format(name, code) if name != code else code)
    return "; ".join(pieces) + "."


def _display_items(keys: Sequence[str], display_map: Dict[str, str]) -> str:
    if not keys:
        return "None predicted."
    return "; ".join(display_map.get(key, key) for key in keys) + "."


def render_structured_report(entities: PredictionEntities, schema: LabelSchema) -> str:
    tooth_text = ", ".join(entities.teeth) if entities.teeth else "None predicted"
    return " ".join(
        [
            "Tooth notations: {}.".format(tooth_text),
            "Diagnosis: {}".format(_diagnosis_text(entities, schema)),
            "Treatment actions: {}".format(_display_items(entities.actions, schema.action_display)),
            "Medications: {}".format(_display_items(entities.medications, schema.medication_display)),
            "Doctor advice: Routine clinical follow-up is recommended if discomfort occurs.",
        ]
    )


def merge_with_retrieved_record(
    case_id: str,
    entities: PredictionEntities,
    schema: LabelSchema,
    nearest: Optional[CaseRecord],
    mode: str = "hybrid",
) -> Dict[str, Any]:
    if mode not in {"template", "retrieval", "hybrid"}:
        raise ValueError("Unknown report mode: {}".format(mode))
    structured_report = render_structured_report(entities, schema)
    nearest_fields = (nearest.fields or {}) if nearest is not None else {}
    fields = {column: "" for column in TEXT_COLUMNS}

    if mode in {"retrieval", "hybrid"} and nearest is not None:
        for column in TEXT_COLUMNS:
            fields[column] = _clean_retrieved_text(nearest_fields.get(column, ""))

    predicted_tooth_prefix = "Predicted tooth notations: {}.".format(
        ", ".join(entities.teeth) if entities.teeth else "none"
    )
    predicted_diagnosis = _diagnosis_text(entities, schema)
    predicted_actions = _display_items(entities.actions, schema.action_display)
    predicted_medications = _display_items(entities.medications, schema.medication_display)
    involved_teeth = ", ".join(entities.teeth) if entities.teeth else "unspecified teeth"

    if mode == "template":
        fields["Main appeal"] = (
            "Dental concern involving {}; clinical evaluation requested.".format(involved_teeth)
        )
        fields["Present medical history"] = (
            "CBCT-based assessment was performed for suspected dental disease involving {}.".format(
                involved_teeth
            )
        )
        fields["Oral Check"] = predicted_tooth_prefix
        fields["Diagnosis"] = predicted_diagnosis
        fields["Treatment plan"] = predicted_actions
        fields["Handle"] = "Treatment actions: {} Medications: {}".format(
            predicted_actions, predicted_medications
        )
        fields["Doctor advices"] = "Routine clinical follow-up if discomfort occurs."
    elif mode == "hybrid":
        if not fields["Main appeal"]:
            fields["Main appeal"] = "Dental concern involving {}.".format(involved_teeth)
        if not fields["Present medical history"]:
            fields["Present medical history"] = (
                "CBCT-based assessment was performed for suspected dental disease."
            )
        fields["Oral Check"] = "{} {}".format(predicted_tooth_prefix, fields["Oral Check"]).strip()
        fields["Diagnosis"] = "{} {}".format(predicted_diagnosis, fields["Diagnosis"]).strip()
        fields["Treatment plan"] = "{} {}".format(predicted_actions, fields["Treatment plan"]).strip()
        fields["Handle"] = "Treatment actions: {} Medications: {} {}".format(
            predicted_actions, predicted_medications, fields["Handle"]
        ).strip()

    report_parts = [structured_report]
    if mode in {"retrieval", "hybrid"}:
        for column in ["Oral Check", "Diagnosis", "Treatment plan", "Handle", "Doctor advices"]:
            if fields[column]:
                report_parts.append("{}: {}".format(column, fields[column]))

    return {
        "case_id": str(case_id),
        "report": " ".join(report_parts),
        "tooth_notations": list(entities.teeth),
        "diagnosis_codes": list(entities.diagnosis_codes),
        "tooth_diagnosis_pairs": [
            {"tooth": tooth, "diagnosis_code": code}
            for tooth, code in (entities.diagnosis_pairs or [])
        ],
        "treatment_actions": list(entities.actions),
        "medications": list(entities.medications),
        "sex": entities.sex,
        "age": round(float(entities.age), 1),
        "fields": fields,
    }


def fallback_entities_from_record(entities: PredictionEntities, record: CaseRecord) -> PredictionEntities:
    fields = record.fields or {}
    if not entities.teeth:
        entities.teeth = extract_tooth_notations(
            " ".join(fields.get(name, "") for name in ["Oral Check", "Diagnosis", "Treatment plan", "Handle"])
        )
    if not entities.diagnosis_codes:
        entities.diagnosis_codes = extract_diagnosis_codes(fields.get("Diagnosis", ""))
    if not entities.actions:
        entities.actions = extract_actions(
            " ".join(fields.get(name, "") for name in ["Treatment plan", "Handle", "Doctor advices"])
        )
    if not entities.medications:
        entities.medications = extract_medications(
            " ".join(fields.get(name, "") for name in ["Handle", "Doctor advices"])
        )
    return entities
