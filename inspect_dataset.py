"""Print a patient-level audit of the local MMDental challenge data."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import nibabel as nib

from mmdental.data import load_split_records
from mmdental.labels import extract_actions, extract_diagnosis_codes, extract_medications, extract_tooth_notations
from mmdental.paths import default_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_records = []
    for split in ["Train-Labeled", "Train-Unlabeled", "Validation"]:
        records = load_split_records(args.data_root, split)
        all_records.extend(records)
        visit_counts = [record.num_visits for record in records if record.num_visits]
        print(
            "{}: cases={}, visits={}, visits_per_case={}..{}".format(
                split,
                len(records),
                sum(visit_counts),
                min(visit_counts) if visit_counts else "n/a",
                max(visit_counts) if visit_counts else "n/a",
            )
        )

    sample = all_records[0]
    image = nib.load(sample.image_path)
    print("NIfTI sample: shape={}, spacing={}, dtype={}".format(image.shape, image.header.get_zooms(), image.get_data_dtype()))

    diagnosis = Counter()
    teeth = Counter()
    actions = Counter()
    medications = Counter()
    for record in all_records:
        fields = record.fields or {}
        diagnosis.update(extract_diagnosis_codes(fields.get("Diagnosis", "")))
        teeth.update(extract_tooth_notations(record.target_text))
        actions.update(extract_actions(record.target_text))
        medications.update(extract_medications(record.target_text))
    print("Diagnosis codes: {}".format(dict(diagnosis.most_common())))
    print("FDI teeth: {}".format(dict(teeth.most_common())))
    print("Treatment actions: {}".format(dict(actions.most_common())))
    print("Medications: {}".format(dict(medications.most_common())))


if __name__ == "__main__":
    main()
