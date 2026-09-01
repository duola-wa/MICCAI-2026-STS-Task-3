"""Regression tests for the official Codabench submission package."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from mmdental.submission import (
    OFFICIAL_SUBMISSION_FIELDS,
    build_official_payload,
    write_official_submission,
)


class SubmissionTests(unittest.TestCase):
    def prediction(self, case_id: str = "100"):
        return {
            "case_id": case_id,
            "fields": {name: "{} text".format(name) for name in OFFICIAL_SUBMISSION_FIELDS},
        }

    def test_payload_has_exact_official_fields(self) -> None:
        payload = build_official_payload([self.prediction()])
        self.assertEqual(list(payload), ["100"])
        self.assertEqual(tuple(payload["100"]), OFFICIAL_SUBMISSION_FIELDS)

    def test_empty_official_field_is_rejected(self) -> None:
        prediction = self.prediction()
        prediction["fields"]["Main appeal"] = ""
        with self.assertRaises(ValueError):
            build_official_payload([prediction])

    def test_zip_contains_only_predictions_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_path, zip_path = write_official_submission(
                [self.prediction()], Path(directory)
            )
            self.assertTrue(json_path.is_file())
            with ZipFile(zip_path, "r") as archive:
                self.assertEqual(archive.namelist(), ["predictions.json"])
                payload = json.loads(archive.read("predictions.json"))
            self.assertIn("100", payload)


if __name__ == "__main__":
    unittest.main()
