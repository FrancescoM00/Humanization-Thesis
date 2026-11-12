#!/usr/bin/env python3
"""Evaluate the dedicated hedging panel at its frozen historical threshold."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache'))
from semantics.hedging_analysis import analyze,write_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs',type=Path,required=True)
    parser.add_argument('--scores',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--review',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summary=analyze(args.pairs,args.scores,args.output,args.protocol,args.review)
    report=write_report(args.output)
    print(json.dumps({'primary_results':summary['primary_results'],'report':str(report),
                      'independent_human_validation_available':summary['independent_human_validation_available']},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
