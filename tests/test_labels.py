"""Regression tests for noisy dental entity formats observed in the real CSV."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmdental.labels import (
    extract_actions,
    extract_diagnosis_codes,
    extract_medications,
    extract_tooth_diagnosis_pairs,
    extract_tooth_notations,
)


class LabelParsingTests(unittest.TestCase):
    def test_extended_and_local_diagnosis_codes(self) -> None:
        text = "K02.900x001 K08.300x002 K02.400��001 LC01 LC10 DDE65"
        self.assertEqual(
            extract_diagnosis_codes(text),
            ["K02.400X001", "K02.900X001", "K08.300X002", "LC01", "LC10"],
        )

    def test_adult_and_primary_ranges(self) -> None:
        text = "31-38 41-48 Chronic periodontitis. 51-55 teeth"
        expected = [
            "31", "32", "33", "34", "35", "36", "37", "38",
            "41", "42", "43", "44", "45", "46", "47", "48",
            "51", "52", "53", "54", "55",
        ]
        self.assertEqual(extract_tooth_notations(text), expected)

    def test_noisy_tooth_lists(self) -> None:
        self.assertEqual(
            extract_tooth_notations("27 37 46 Black caries can be seen"),
            ["27", "37", "46"],
        )
        self.assertEqual(
            extract_tooth_notations("33.43.44.45 Residual root"),
            ["33", "43", "44", "45"],
        )
        self.assertEqual(
            extract_tooth_notations("65 Caries. 54,64 Dental check-up"),
            ["54", "64", "65"],
        )

    def test_numeric_dose_false_positives(self) -> None:
        text = "torque 15Ncm/25Ncm/35Ncm; implant 3.75*10mm; dose 0.25g; 12.11 iodophors"
        self.assertEqual(extract_tooth_notations(text), [])

    def test_allergy_is_not_medication(self) -> None:
        self.assertEqual(extract_medications("history of penicillin allergy"), [])
        self.assertEqual(extract_medications("Cephalosporin skin test was positive"), [])
        self.assertEqual(extract_medications("Local anesthesia with articaine"), ["articaine"])

    def test_common_action_abbreviations(self) -> None:
        self.assertIn("root_canal_treatment", extract_actions("*47 RCT"))
        self.assertIn("follow_up", extract_actions("Return to the clinic after one week"))
        self.assertIn("tooth_extraction", extract_actions("Recommended removal of the impacted tooth"))

    def test_tooth_diagnosis_pairs_are_local(self) -> None:
        text = (
            "*16,*14,*24 Acute pulpitis (K04.001) "
            "*18,*28,*38 Impacted tooth (K01.100) "
            "*17,*15,*25,*26,*36,*37 Dental caries (K02.900)"
        )
        pairs = extract_tooth_diagnosis_pairs(text)
        self.assertIn(("14", "K04.001"), pairs)
        self.assertIn(("38", "K01.100"), pairs)
        self.assertIn(("36", "K02.900"), pairs)
        self.assertNotIn(("14", "K01.100"), pairs)

    def test_global_diagnosis_is_not_forced_to_previous_tooth(self) -> None:
        text = "*36 Missing teeth (K00.002). [VISIT] Malformed teeth (K07.302)"
        self.assertEqual(extract_tooth_diagnosis_pairs(text), [("36", "K00.002")])


if __name__ == "__main__":
    unittest.main()
