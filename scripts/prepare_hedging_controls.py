#!/usr/bin/env python3
"""Prepare the frozen native-caution panel without NLI or generative models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from semantics.hedging_controls import prepare_panel

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'supplements/hedging/results/protocol.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/hedging')
    args=parser.parse_args()
    print(json.dumps(prepare_panel(args.protocol,args.output,ROOT),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
