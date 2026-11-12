#!/usr/bin/env python3
"""Verify the frozen panel and record source-only input handling before inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from humanizers.study_protocol import prepare_study_protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--panel', choices=['smoke', 'pilot', 'study'], default='study')
    a = p.parse_args()
    print(json.dumps(prepare_study_protocol(a.cohort, a.output, panel=a.panel), indent=2))


if __name__ == '__main__':
    main()
