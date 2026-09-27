#!/usr/bin/env python3
"""Reproduce the handoff's full-cohort pipeline/reader consistency checks."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc import batch, dicom_io


def comparable(frame):
    return frame.drop(columns=["time_of_processing"]).set_index("path_to_study").sort_index().fillna("")


def mismatches(left, right):
    a, b = comparable(left), comparable(right)
    if not a.index.equals(b.index) or not a.columns.equals(b.columns):
        raise ValueError("report schemas or paths differ")
    return int(a.ne(b).any(axis=1).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "Data/НД_для_обучения/Исследования")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/rotation_human_profile/pipeline_verification.json")
    args = ap.parse_args()
    if not dicom_io.HAVE_PYDICOM:
        raise SystemExit("pydicom is required; fallback alone is insufficient")
    with tempfile.TemporaryDirectory(prefix="bones-day6-") as tmp:
        tmp = Path(tmp)
        reports, diagnostics = [], []
        for name, pydicom in (("first", True), ("repeat", True), ("builtin", False)):
            with patch.object(dicom_io, "HAVE_PYDICOM", pydicom):
                if batch.run(args.root, tmp / f"{name}.csv", tmp / f"{name}_diag.csv", quiet=True):
                    raise RuntimeError(f"pipeline failed: {name}")
            reports.append(pd.read_csv(tmp / f"{name}.csv"))
            diagnostics.append(pd.read_csv(tmp / f"{name}_diag.csv"))
        first, repeat, builtin = reports
        diag = diagnostics[0]
        old = pd.read_csv(ROOT / "reports/batch_diagnostics_train.csv")
        # Колонки, добавленные после сохранения диагностики (projection, 24.09),
        # сверять не с чем: сравниваются общие.
        old = old[[c for c in first.columns if c in old.columns]]
        first_cmp = first[old.columns]
        rules = {}
        for filename, pairs in {
            "spine_rules.csv": [("pred_pos", "SPINE_POSITIONING"), ("pred_tilt", "SPINE_AXIS_TILT")],
            "femur_rules.csv": [("pred_roi", "FEMUR_ROI_COVERAGE")],
            # С 23 сентября: правило ротации, границы из scripts/17_fit_rotation_rule.py.
            "rotation_rule/decisions.csv": [("pred", "FEMUR_POSITIONING_ROTATION")],
        }.items():
            reference = pd.read_csv(ROOT / "reports" / filename)
            merged = reference.merge(diag[["path_to_study", "violation_type"]],
                                      left_on="rel_path", right_on="path_to_study",
                                      validate="one_to_one", how="left", indicator=True)
            if not merged._merge.eq("both").all():
                raise ValueError(f"missing rule comparison paths: {filename}")
            errors = 0
            for column, label in pairs:
                detected = merged.violation_type.fillna("").map(lambda v: label in v.split(";"))
                # -1 в decisions.csv — отказ правила: нарушения нет, как в конвейере.
                errors += int(detected.ne(merged[column].clip(lower=0).astype(bool)).sum())
            rules[filename] = {"matched_images": len(merged), "prediction_mismatches": errors}
        result = {
            "files": len(first), "unique_study_images": len(diag[["study", "md5"]].drop_duplicates()),
            "status": first.processing_status.value_counts().to_dict(),
            "reader": diag.reader.value_counts().to_dict(),
            "repeat_mismatches_except_time": mismatches(first, repeat),
            "builtin_reader_mismatches_except_time": mismatches(first, builtin),
            "day5_prediction_mismatches": mismatches(first_cmp, old),
            "rule_comparisons": rules,
            "rows_with_read_warnings": int(diag.warnings.notna().sum()),
            "versions": {p: importlib.metadata.version(p) for p in
                         ("pydicom", "numpy", "pandas", "opencv-python-headless", "pytest")},
            "input_sha256": {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                             for p in ("reports/spine_rules.csv", "reports/femur_rules.csv",
                                       "reports/rotation_rule/decisions.csv",
                                       "models/rotation_profile_v1.json",
                                       "reports/batch_diagnostics_train.csv")},
            "scope": "Core inference dependencies only; full training/service requirements and Docker not verified. CSV content comparison excludes processing time; byte-identical XLSX not tested.",
        }
        passed = (len(first) == 499 and first.processing_status.eq("Success").all()
                  and diag.reader.eq("pydicom").all()
                  and diagnostics[2].reader.eq("minidicom").all()
                  and all(result[k] == 0 for k in ("repeat_mismatches_except_time",
                          "builtin_reader_mismatches_except_time", "day5_prediction_mismatches"))
                  and all(v["prediction_mismatches"] == 0 for v in rules.values()))
        result["passed"] = bool(passed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
