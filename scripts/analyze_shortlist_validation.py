#!/usr/bin/env python3
"""Analyze saved compact-feature predictions with paired group bootstrap."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

from stylometry.shortlist_statistics import analyze_results
from stylometry.shortlist_report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/02_stylometry/shortlist")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/shortlist_analysis",
                        help="Analysis copy; the published --results folder remains an input")
    parser.add_argument("--protocol", type=Path, default=None)
    args = parser.parse_args()
    if args.results.resolve() != args.output.resolve():
        shutil.copytree(args.results, args.output, dirs_exist_ok=True)
    summary = analyze_results(args.output, args.protocol)
    report = write_report(args.output, args.protocol)
    print(json.dumps({"primary_metrics": summary["primary_metrics"],
                      "report": str(report),
                      "elapsed_seconds": summary["elapsed_seconds"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
