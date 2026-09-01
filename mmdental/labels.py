"""Clinical entity parsing and patient-level target encoding."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from .data import CaseRecord


ADULT_FDI_TEETH = [
    "{}{}".format(quadrant, position)
    for quadrant in range(1, 5)
    for position in range(1, 9)
]
PRIMARY_FDI_TEETH = [
    "{}{}".format(quadrant, position)
    for quadrant in range(5, 9)
    for position in range(1, 6)
]
FDI_TEETH = ADULT_FDI_TEETH + PRIMARY_FDI_TEETH
FDI_TOKEN_PATTERN = r"(?:[1-4][1-8]|[5-8][1-5])"


ACTION_DEFINITIONS: List[Dict[str, Any]] = [
    {"key": "root_canal_treatment", "display": "Root canal treatment", "patterns": [r"root canal", r"endodont", r"root filling", r"\bRCT\b"]},
    {"key": "crown_restoration", "display": "Crown restoration", "patterns": [r"\bcrown\b"]},
    {"key": "implant_treatment", "display": "Implant treatment", "patterns": [r"implant"]},
    {"key": "tooth_extraction", "display": "Tooth extraction", "patterns": [r"extract", r"tooth removal", r"recommended removal", r"pull out", r"remove(?:d|s|)\s+(?:the\s+)?tooth"]},
    {"key": "filling_restoration", "display": "Filling/restoration", "patterns": [r"\bfilling\b", r"filled", r"resin restoration", r"restorative treatment"]},
    {"key": "periodontal_scaling", "display": "Periodontal scaling", "patterns": [r"scal(?:ing|e)", r"periodontal treatment", r"supragingival"]},
    {"key": "orthodontic_treatment", "display": "Orthodontic treatment", "patterns": [r"orthodont", r"aligner", r"bracket"]},
    {"key": "denture_prosthesis", "display": "Denture/prosthesis", "patterns": [r"denture", r"prosthe", r"removable partial"]},
    {"key": "suture_management", "display": "Suture management", "patterns": [r"sutur"]},
    {"key": "incision_drainage", "display": "Incision and drainage", "patterns": [r"incision", r"drainage"]},
    {"key": "disinfection", "display": "Local disinfection", "patterns": [r"disinfect", r"iodophor", r"povidone"]},
    {"key": "anesthesia", "display": "Local anesthesia", "patterns": [r"anesthe", r"articaine", r"lidocaine"]},
    {"key": "imaging_followup", "display": "Imaging examination", "patterns": [r"\bcbct\b", r"radiograph", r"take (?:a )?pictures?", r"filming"]},
    {"key": "medication", "display": "Medication", "patterns": [r"antibiotic", r"prescri", r"medication"]},
    {"key": "follow_up", "display": "Clinical follow-up", "patterns": [r"follow[- ]?up", r"re-examination", r"review visit", r"return to (?:the )?clinic"]},
]


MEDICATION_DEFINITIONS: List[Dict[str, Any]] = [
    {"key": "articaine", "display": "Articaine", "patterns": [r"articaine", r"atecaine"]},
    {"key": "lidocaine", "display": "Lidocaine", "patterns": [r"lidocaine"]},
    {"key": "povidone_iodine", "display": "Povidone-iodine", "patterns": [r"povidone[- ]?iodine", r"iodophor"]},
    {"key": "chlorhexidine", "display": "Chlorhexidine", "patterns": [r"chlorhexidine"]},
    {"key": "amoxicillin", "display": "Amoxicillin", "patterns": [r"amoxicillin"]},
    {"key": "metronidazole", "display": "Metronidazole", "patterns": [r"metronidazole"]},
    {"key": "ibuprofen", "display": "Ibuprofen", "patterns": [r"ibuprofen"]},
    {"key": "cephalosporin", "display": "Cephalosporin", "patterns": [r"cephalospor"]},
    {"key": "penicillin", "display": "Penicillin", "patterns": [r"penicillin"]},
    {"key": "epinephrine", "display": "Epinephrine", "patterns": [r"epinephrine", r"adrenaline"]},
    {"key": "dexamethasone", "display": "Dexamethasone", "patterns": [r"dexamethasone"]},
]


ICD_PATTERN = re.compile(
    r"\b((?:[A-Z]\d{2}|LC\d{2})(?:[.．]\d{1,5})?(?:[xX×�]+\d{1,5})?)\b",
    flags=re.IGNORECASE,
)


def _field(record: CaseRecord, name: str) -> str:
    return (record.fields or {}).get(name, "")


def normalize_icd_code(code: str) -> str:
    normalized = (
        code.upper()
        .replace("．", ".")
        .replace("×", "X")
        .replace("�", "X")
        .strip(" .;,()[]")
    )
    return re.sub(r"X+", "X", normalized)


def extract_diagnosis_codes(text: str) -> List[str]:
    return sorted({normalize_icd_code(match) for match in ICD_PATTERN.findall(text or "")})


def extract_tooth_diagnosis_pairs(text: str) -> List[Tuple[str, str]]:
    """Extract conservative ``(adult FDI tooth, ICD code)`` associations.

    MMDental diagnoses commonly use blocks such as ``*16,*14 Acute pulpitis
    (K04.001)``.  For each ICD occurrence we inspect only the local text since
    the previous diagnosis/visit boundary.  Codes without an explicit local
    tooth remain patient-level diagnoses and are deliberately not paired.
    """
    text = text or ""
    pairs = set()
    previous_code_end = 0
    for match in ICD_PATTERN.finditer(text):
        local = text[previous_code_end:match.start()]
        boundary = max(
            local.rfind("[VISIT]"),
            local.rfind("[visit]"),
            local.rfind(")"),
            local.rfind(";"),
        )
        if boundary >= 0:
            local = local[boundary + 1:]
        code = normalize_icd_code(match.group(1))
        for tooth in extract_tooth_notations(local):
            if tooth in ADULT_FDI_TEETH:
                pairs.add((tooth, code))
        previous_code_end = match.end()
    return sorted(
        pairs,
        key=lambda item: (ADULT_FDI_TEETH.index(item[0]), item[1]),
    )


def extract_tooth_notations(text: str) -> List[str]:
    """Extract FDI notations while filtering obvious time/dose false positives."""
    text = text or ""
    found = set()
    patterns = [
        r"\*\s*(" + FDI_TOKEN_PATTERN + r")(?!\d)",
        r"#\s*(" + FDI_TOKEN_PATTERN + r")(?!\d)",
        r"\b(?:tooth|teeth|FDI)\s*(?:number|no\.?|#)?\s*(" + FDI_TOKEN_PATTERN + r")(?!\d)",
        r"(?<![\d.])(" + FDI_TOKEN_PATTERN + r")(?=\s*(?:[,;/]|and\b|tooth\b|teeth\b|crown\b|root\b|caries\b|decay\b|dental\b|check[- ]?up\b|exam|pit\b|fissure\b|implant\b|impacted\b|missing\b|defect\b|CBCT\b|gingiv|percussion\b|fracture\b|pulp|periodont|are\s+left|remain|present))",
        r"(?:^|[.;:]|\[VISIT\])\s*(" + FDI_TOKEN_PATTERN + r")(?=\s*(?:[,;/]|\.?\s*(?:There\b|The\b|CBCT\b|crown\b|root\b|caries\b|decay\b|dental\b|check[- ]?up\b|exam|pit\b|fissure\b|implant\b|impacted\b|missing\b|defect\b|gingiv|percussion\b|fracture\b|pulp|periodont|are\s+left|remain|present)))",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            tooth = match.group(1)
            context_after = text[match.end():match.end() + 16].lower()
            context_before = text[max(0, match.start() - 10):match.start()].lower()
            if re.match(r"\s*(?:hour|minute|week|month|year|ml|mm|%)", context_after):
                continue
            if re.search(r"(?:age|aged)\s*$", context_before):
                continue
            if tooth in FDI_TEETH:
                found.add(tooth)
    # Parse multi-tooth lists only when the surrounding text is clearly dental.
    sequence_pattern = (
        r"(?<![\d.])((?:" + FDI_TOKEN_PATTERN + r")"
        r"(?:(?:\s*[,./]\s*|\s+)(?:" + FDI_TOKEN_PATTERN + r")){1,})(?!\d)"
    )
    dental_context = re.compile(
        r"black\s+caries|caries|decay|dental|check[- ]?up|exam|pit|fissure|residual\s+root|impacted|missing|defect|crown|root|pulp|periodont|gingiv|CBCT|loose|percussion|fixed\s+bridge|repair|recommend|treatment|are\s+left|remain|present",
        flags=re.IGNORECASE,
    )
    for match in re.finditer(sequence_pattern, text):
        before = text[max(0, match.start() - 20):match.start()]
        after = text[match.end():match.end() + 40]
        if not re.search(r"(?:tooth|teeth|FDI)\s*$", before, flags=re.IGNORECASE) and not dental_context.search(after):
            continue
        for tooth in re.findall(FDI_TOKEN_PATTERN, match.group(1)):
            if tooth in FDI_TEETH:
                found.add(tooth)
    # Expand compact same-quadrant ranges such as "31-38" or "51–55".
    range_pattern = (
        r"(?<![\d.])(" + FDI_TOKEN_PATTERN + r")\s*[-–—]\s*(" + FDI_TOKEN_PATTERN + r")(?!\d)"
    )
    for match in re.finditer(range_pattern, text):
        start, end = match.group(1), match.group(2)
        before = text[max(0, match.start() - 20):match.start()]
        after = text[match.end():match.end() + 50]
        if re.match(r"\s*(?:Ncm|mg\b|g\b|ml\b|mm\b|cm\b|hour|day|week)", after, flags=re.IGNORECASE):
            continue
        follows_another_tooth = re.match(
            r"\s*(?:" + FDI_TOKEN_PATTERN + r")(?:\s*[-–—,./]|\s+)", after
        )
        has_dental_context = bool(
            dental_context.search(after)
            or re.search(r"(?:tooth|teeth|FDI)\s*$", before, flags=re.IGNORECASE)
            or re.match(r"\s*(?:[,.;]|tooth\b|teeth\b|$)", after, flags=re.IGNORECASE)
        )
        if not follows_another_tooth and not has_dental_context:
            continue
        if start[0] == end[0] and int(start[1]) <= int(end[1]):
            for position in range(int(start[1]), int(end[1]) + 1):
                tooth = "{}{}".format(start[0], position)
                if tooth in FDI_TEETH:
                    found.add(tooth)
    return sorted(found, key=FDI_TEETH.index)


def _match_definitions(text: str, definitions: Sequence[Dict[str, Any]]) -> List[str]:
    text = text or ""
    output = []
    for definition in definitions:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in definition["patterns"]):
            output.append(definition["key"])
    return output


def extract_actions(text: str) -> List[str]:
    return _match_definitions(text, ACTION_DEFINITIONS)


def extract_medications(text: str) -> List[str]:
    text = text or ""
    output = []
    negative_pattern = re.compile(
        r"allerg|den(?:y|ies|ied)|no\s+(?:drug\s+)?history|skin test (?:was )?positive",
        flags=re.IGNORECASE,
    )
    for definition in MEDICATION_DEFINITIONS:
        accepted = False
        for pattern in definition["patterns"]:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                context = text[max(0, match.start() - 50):match.end() + 50]
                if not negative_pattern.search(context):
                    accepted = True
                    break
            if accepted:
                break
        if accepted:
            output.append(definition["key"])
    return output


def _definition_keys(definitions: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(item["key"]) for item in definitions]


def _definition_display(definitions: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    return {str(item["key"]): str(item["display"]) for item in definitions}


def _multihot(values: Iterable[str], vocabulary: Sequence[str]) -> np.ndarray:
    index = {value: idx for idx, value in enumerate(vocabulary)}
    output = np.zeros(len(vocabulary), dtype=np.float32)
    for value in values:
        if value in index:
            output[index[value]] = 1.0
    return output


def _derive_diagnosis_names(records: Sequence[CaseRecord], codes: Sequence[str]) -> Dict[str, str]:
    candidates: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        diagnosis = _field(record, "Diagnosis")
        if not diagnosis:
            continue
        for match in ICD_PATTERN.finditer(diagnosis):
            code = normalize_icd_code(match.group(1))
            # Diagnosis cells often concatenate several teeth/codes without a delimiter.
            # Use only the local phrase immediately preceding each code.
            prefix = diagnosis[max(0, match.start() - 120):match.start()]
            last_boundary = max(prefix.rfind(")"), prefix.rfind(";"), prefix.rfind("[VISIT]"))
            if last_boundary >= 0:
                prefix = prefix[last_boundary + 1:]
            tooth_chunks = re.split(r"\*\s*" + FDI_TOKEN_PATTERN + r"(?:\s*[,/]\s*)?", prefix)
            cleaned = tooth_chunks[-1]
            cleaned = ICD_PATTERN.sub("", cleaned)
            cleaned = re.sub(
                r"\b(?:tooth|teeth)\s*" + FDI_TOKEN_PATTERN + r"\b",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(r"\[?VISIT\]?", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"[()*,:]+", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" -.")
            cleaned = re.sub(r"^(?:" + FDI_TOKEN_PATTERN + r"\s*)+", "", cleaned).strip()
            cleaned = re.sub(r"^\d+[A-Z]?\s+", "", cleaned).strip()
            if 2 <= len(cleaned) <= 80 and re.search(r"[A-Za-z]", cleaned):
                candidates[code][cleaned] += 1
    output = {}
    for code in codes:
        if candidates.get(code):
            output[code] = candidates[code].most_common(1)[0][0]
        else:
            output[code] = code
    return output


@dataclass
class LabelSchema:
    diagnosis_codes: List[str]
    diagnosis_names: Dict[str, str]
    tooth_labels: List[str]
    action_labels: List[str]
    medication_labels: List[str]
    action_display: Dict[str, str]
    medication_display: Dict[str, str]
    age_mean: float
    age_std: float
    text_dim: int
    source_splits: List[str]
    source_case_ids: List[str]
    vectorizer: Optional[Any] = None
    text_reducer: Optional[Any] = None
    artifact_dir: Optional[Path] = None

    @classmethod
    def fit(
        cls,
        records: Sequence[CaseRecord],
        text_dim: int = 64,
        min_diagnosis_frequency: int = 1,
        max_text_features: int = 4096,
    ) -> "LabelSchema":
        if len(records) < 3:
            raise ValueError("At least three patient records are required to fit the schema")

        code_counter: Counter = Counter()
        for record in records:
            code_counter.update(extract_diagnosis_codes(_field(record, "Diagnosis")))
        diagnosis_codes = sorted(
            [code for code, count in code_counter.items() if count >= min_diagnosis_frequency]
        )
        if not diagnosis_codes:
            raise ValueError("No diagnosis codes were extracted from the selected records")

        ages = np.asarray([record.age for record in records if not math.isnan(record.age)], dtype=np.float32)
        age_mean = float(ages.mean()) if len(ages) else 0.0
        age_std = float(ages.std()) if len(ages) and float(ages.std()) > 1e-6 else 1.0
        # Stateless hashing keeps the text target coordinate system identical
        # across folds without fitting on fold-validation reports.
        actual_text_dim = int(text_dim)
        if actual_text_dim < 8:
            raise ValueError("text_dim must be at least 8")
        vectorizer = HashingVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            n_features=actual_text_dim,
            alternate_sign=False,
            norm="l2",
        )

        return cls(
            diagnosis_codes=diagnosis_codes,
            diagnosis_names=_derive_diagnosis_names(records, diagnosis_codes),
            tooth_labels=list(FDI_TEETH),
            action_labels=_definition_keys(ACTION_DEFINITIONS),
            medication_labels=_definition_keys(MEDICATION_DEFINITIONS),
            action_display=_definition_display(ACTION_DEFINITIONS),
            medication_display=_definition_display(MEDICATION_DEFINITIONS),
            age_mean=age_mean,
            age_std=age_std,
            text_dim=actual_text_dim,
            source_splits=sorted({record.split for record in records}),
            source_case_ids=sorted("{}:{}".format(record.split, record.case_id) for record in records),
            vectorizer=vectorizer,
            text_reducer=None,
        )

    def signature(self) -> str:
        payload = {
            "diagnosis_codes": self.diagnosis_codes,
            "diagnosis_names": self.diagnosis_names,
            "tooth_labels": self.tooth_labels,
            "action_labels": self.action_labels,
            "medication_labels": self.medication_labels,
            "text_dim": self.text_dim,
            "text_backend": "hashing" if self.text_reducer is None else "tfidf_svd",
            "age_mean": round(self.age_mean, 6),
            "age_std": round(self.age_std, 6),
            "source_splits": self.source_splits,
            "source_case_ids": self.source_case_ids,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def transform_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("LabelSchema text models are not loaded")
        matrix = self.vectorizer.transform([text or "no documented clinical target" for text in texts])
        if self.text_reducer is None:
            embedding = matrix.toarray().astype(np.float32)
        else:
            embedding = self.text_reducer.transform(matrix).astype(np.float32)
        return normalize(embedding, norm="l2").astype(np.float32)

    def encode_record(self, record: CaseRecord) -> Dict[str, np.ndarray]:
        tooth_text = " ".join(
            _field(record, column)
            for column in ["Oral Check", "Diagnosis", "Treatment plan", "Handle"]
        )
        diagnosis_text = _field(record, "Diagnosis")
        action_text = " ".join(
            _field(record, column)
            for column in ["Treatment plan", "Handle", "Doctor advices"]
        )
        medication_text = " ".join(
            _field(record, column)
            for column in ["Handle", "Doctor advices"]
        )
        sex_value = 1 if "female" in record.sex else 0 if "male" in record.sex else -1
        age_value = 0.0 if math.isnan(record.age) else (record.age - self.age_mean) / self.age_std
        text_embedding = self.transform_texts([record.target_text])[0]
        tooth_values = extract_tooth_notations(tooth_text)
        diagnosis_values = extract_diagnosis_codes(diagnosis_text)
        action_values = extract_actions(action_text)
        medication_values = extract_medications(medication_text)
        tooth_diagnosis_pairs = extract_tooth_diagnosis_pairs(diagnosis_text)
        tooth_diagnosis = np.zeros(
            (len(ADULT_FDI_TEETH), len(self.diagnosis_codes)),
            dtype=np.float32,
        )
        tooth_index = {value: index for index, value in enumerate(ADULT_FDI_TEETH)}
        diagnosis_index = {value: index for index, value in enumerate(self.diagnosis_codes)}
        for tooth, code in tooth_diagnosis_pairs:
            if code in diagnosis_index:
                tooth_diagnosis[tooth_index[tooth], diagnosis_index[code]] = 1.0

        return {
            "teeth": _multihot(tooth_values, self.tooth_labels),
            "diagnosis": _multihot(diagnosis_values, self.diagnosis_codes),
            "tooth_diagnosis": tooth_diagnosis,
            "actions": _multihot(action_values, self.action_labels),
            "medications": _multihot(medication_values, self.medication_labels),
            "text_embedding": text_embedding.astype(np.float32),
            "sex": np.asarray(sex_value, dtype=np.int64),
            "age": np.asarray(age_value, dtype=np.float32),
            "teeth_mask": np.asarray(float(bool(tooth_values)), dtype=np.float32),
            "diagnosis_mask": np.asarray(float(bool(diagnosis_values)), dtype=np.float32),
            "tooth_diagnosis_mask": np.asarray(
                float(bool(tooth_diagnosis_pairs)), dtype=np.float32
            ),
            "actions_mask": np.asarray(float(bool(action_values)), dtype=np.float32),
            "medications_mask": np.asarray(float(bool(medication_text.strip())), dtype=np.float32),
            "text_embedding_mask": np.asarray(float(bool(record.target_text.strip())), dtype=np.float32),
            "sex_mask": np.asarray(float(sex_value >= 0), dtype=np.float32),
            "age_mask": np.asarray(float(not math.isnan(record.age)), dtype=np.float32),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis_codes": self.diagnosis_codes,
            "diagnosis_names": self.diagnosis_names,
            "tooth_labels": self.tooth_labels,
            "action_labels": self.action_labels,
            "medication_labels": self.medication_labels,
            "action_display": self.action_display,
            "medication_display": self.medication_display,
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "text_dim": self.text_dim,
            "source_splits": self.source_splits,
            "source_case_ids": self.source_case_ids,
            "text_backend": "hashing" if self.text_reducer is None else "tfidf_svd",
            "signature": self.signature(),
        }

    def save(self, artifact_dir: Path, records: Sequence[CaseRecord]) -> None:
        if self.vectorizer is None:
            raise RuntimeError("Cannot save an unfitted schema")
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with (artifact_dir / "schema.json").open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
        joblib.dump(self.vectorizer, artifact_dir / "tfidf.joblib")
        if self.text_reducer is not None:
            joblib.dump(self.text_reducer, artifact_dir / "text_svd.joblib")
        elif (artifact_dir / "text_svd.joblib").is_file():
            (artifact_dir / "text_svd.joblib").unlink()
        embeddings = self.transform_texts([record.target_text for record in records])
        np.save(artifact_dir / "retrieval_embeddings.npy", embeddings, allow_pickle=False)
        with (artifact_dir / "retrieval_records.json").open("w", encoding="utf-8") as handle:
            json.dump([record.to_dict() for record in records], handle, ensure_ascii=False, indent=2)
        self.artifact_dir = artifact_dir.resolve()

    @classmethod
    def load(cls, artifact_dir: Path) -> "LabelSchema":
        artifact_dir = Path(artifact_dir)
        with (artifact_dir / "schema.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        schema = cls(
            diagnosis_codes=list(payload["diagnosis_codes"]),
            diagnosis_names=dict(payload["diagnosis_names"]),
            tooth_labels=list(payload["tooth_labels"]),
            action_labels=list(payload["action_labels"]),
            medication_labels=list(payload["medication_labels"]),
            action_display=dict(payload["action_display"]),
            medication_display=dict(payload["medication_display"]),
            age_mean=float(payload["age_mean"]),
            age_std=float(payload["age_std"]),
            text_dim=int(payload["text_dim"]),
            source_splits=list(payload.get("source_splits", [])),
            source_case_ids=list(payload.get("source_case_ids", [])),
            vectorizer=joblib.load(artifact_dir / "tfidf.joblib"),
            text_reducer=(
                joblib.load(artifact_dir / "text_svd.joblib")
                if payload.get("text_backend", "tfidf_svd") == "tfidf_svd"
                and (artifact_dir / "text_svd.joblib").is_file()
                else None
            ),
            artifact_dir=artifact_dir.resolve(),
        )
        expected = payload.get("signature")
        if expected and expected != schema.signature():
            raise ValueError("Schema signature mismatch in {}".format(artifact_dir))
        return schema

    def load_retrieval_bank(self) -> Tuple[List[CaseRecord], np.ndarray]:
        if self.artifact_dir is None:
            raise RuntimeError("Schema artifact directory is unknown")
        with (self.artifact_dir / "retrieval_records.json").open("r", encoding="utf-8") as handle:
            records = [CaseRecord.from_dict(item) for item in json.load(handle)]
        embeddings = np.load(self.artifact_dir / "retrieval_embeddings.npy", allow_pickle=False)
        return records, embeddings.astype(np.float32)
