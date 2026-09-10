"""
Batch-evaluate the OCR + extraction pipeline against all synthetic sample
records, comparing extracted values to ground truth, broken down by scan
degradation severity (light/medium/heavy) so you can quote real accuracy
numbers in your demo instead of guessing.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))

from preprocess import preprocess_with_fallback
from ocr_engine import run_ocr
from extract import extract_fields, overall_confidence

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/sample_records")
FIELDS = ["owner_name", "khasra_no", "khata_no", "area_hectare", "village",
          "tehsil", "district", "classification", "mutation_no"]


def main():
    gt = json.load(open(os.path.join(DATA_DIR, "ground_truth.json")))

    per_severity = {}
    total_field_matches = 0
    total_fields = 0
    low_conf_flagged_correctly = 0
    low_conf_flagged_total = 0

    for fname, truth in sorted(gt.items()):
        img_path = os.path.join(DATA_DIR, fname)
        severity = truth["_severity"]
        per_severity.setdefault(severity, {"n": 0, "field_matches": 0, "field_total": 0})

        clean, used_recovery = preprocess_with_fallback(img_path)
        words = run_ocr(clean)
        fields = extract_fields(words)

        per_severity[severity]["n"] += 1
        for k in FIELDS:
            extracted = fields[k].value.strip().lower()
            expected = str(truth[k]).strip().lower()
            match = extracted == expected
            per_severity[severity]["field_total"] += 1
            total_fields += 1
            if match:
                per_severity[severity]["field_matches"] += 1
                total_field_matches += 1

            # does low confidence correctly predict a wrong/needs-review field?
            if fields[k].needs_review:
                low_conf_flagged_total += 1
                if not match:
                    low_conf_flagged_correctly += 1

    print(f"{'Severity':10s} {'#Records':10s} {'Field Accuracy':15s}")
    for sev, stats in per_severity.items():
        acc = 100 * stats["field_matches"] / stats["field_total"]
        print(f"{sev:10s} {stats['n']:<10d} {acc:5.1f}%")

    print()
    overall_acc = 100 * total_field_matches / total_fields
    print(f"Overall field-level accuracy: {overall_acc:.1f}% ({total_field_matches}/{total_fields})")
    if low_conf_flagged_total:
        precision = 100 * low_conf_flagged_correctly / low_conf_flagged_total
        print(f"Low-confidence flags that were genuinely wrong: {precision:.1f}% "
              f"({low_conf_flagged_correctly}/{low_conf_flagged_total}) "
              f"-> shows the confidence score usefully routes errors to officer review")


if __name__ == "__main__":
    main()
