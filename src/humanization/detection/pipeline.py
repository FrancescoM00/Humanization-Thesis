"""Valuta detector aperti con calibrazione sui soli umani e ASR condizionato.

I punteggi dei detector sono medie pesate per token su blocchi completi e non
sovrapposti.  Ogni detector usa una propria soglia congelata, il voto di maggioranza
fra tre modelli combina decisioni binarie.
"""
from __future__ import annotations
from collections import defaultdict
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time
import numpy as np
import pandas as pd

def run(cohort_path, rewrites_path, config_path, output, device='cpu', analysis_only=False, evaluate=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cohort, rewrites = (read_jsonl(cohort_path), read_jsonl(rewrites_path))
    config = json.loads(Path(config_path).read_text())
    records = text_records(cohort, rewrites)
    metadata = []
    score_path = output / 'detector_scores.csv'
    if analysis_only:
        previous_path = output / 'detection_run.json'
        if not previous_path.exists():
            raise ValueError('Analysis-only requires the inference provenance in detection_run.json')
        previous = json.loads(previous_path.read_text())
        for key, path in [('cohort_sha256', cohort_path), ('rewrites_sha256', rewrites_path), ('config_sha256', config_path)]:
            if previous.get(key) != hashlib.sha256(Path(path).read_bytes()).hexdigest():
                raise ValueError('Analysis-only inputs/config differ from the recorded inference run')
        scores = pd.read_csv(score_path, dtype={'source_id': str})
    else:
        frames = []
        for detector in config['detectors']:
            if not detector.get('enabled', True):
                continue
            frame, details = score_detector(records, detector, output, device=device)
            frames.append(frame)
            metadata.append(details)
            frame.to_csv(output / ('scores_' + re.sub('[^a-zA-Z0-9_-]', '_', detector['name']) + '.csv'), index=False)
        scores = pd.concat(frames, ignore_index=True)
        scores.to_csv(score_path, index=False)
    if not evaluate:
        (output / 'scoring_run.json').write_text(json.dumps({'models_executed':not analysis_only,'detectors':metadata,'n_scores':len(scores)},indent=2)+'\n')
        return scores
    thresholds, summary, macro = summarize_detection(cohort, rewrites, scores, config)
    (output / 'thresholds.json').write_text(json.dumps(thresholds, indent=2) + '\n')
    summary.to_csv(output / 'detection_summary.csv', index=False)
    macro.to_csv(output / 'detection_macro_summary.csv', index=False)
    manifest = {'cohort_sha256': hashlib.sha256(Path(cohort_path).read_bytes()).hexdigest(), 'rewrites_sha256': hashlib.sha256(Path(rewrites_path).read_bytes()).hexdigest(), 'config_sha256': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(), 'analysis_only': analysis_only, 'detectors': metadata, 'n_text_records_per_detector': len(records), 'asr_denominator': 'test sources detected AI before rewriting; failed generations remain failures', 'calibration': 'human-only, source-group-disjoint from test; score > threshold', 'majority_vote': 'two of three calibrated binary decisions, only when exactly three detectors are used', 'asr_intervals': 'source-group bootstrap, fixed thresholds and checkpoints; calibration/model uncertainty excluded', 'bootstrap': config.get('bootstrap', 500), 'macro_mean': 'equal detector weight over defined ASRs; majority vote is excluded', 'naturalness_measured': False, 'python': sys.version.split()[0]}
    if analysis_only and (output / 'detection_run.json').exists():
        previous = json.loads((output / 'detection_run.json').read_text())
        manifest['detectors'] = previous.get('detectors', [])
    (output / 'detection_run.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(summary[summary.domain == 'overall'].to_string(index=False), flush=True)
    return summary

from humanization.datasets.records import read_jsonl, digest, validate_inputs, text_records, _score_key
from humanization.evaluation.detection import calibrate_threshold, _asr_interval, summarize_detection
from .roberta import validate_ai_label, _require_device, score_detector
