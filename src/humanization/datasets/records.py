"""Valuta detector aperti con calibrazione sui soli umani e ASR condizionato.

Gli score dei detector sono medie pesate per token di blocchi completi non
sovrapposti, sono punteggi del modello. Ogni detector ha una propria soglia fissata; 
il voto di maggioranza dei tre modelli combina decisioni binarie, mai score grezzi. Le generazioni fallite
rimangono nei denominatori dell'ASR.
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


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def validate_inputs(cohort, rewrites):
    ids = [str(row['id']) for row in cohort]
    if len(ids) != len(set(ids)) or not ids or any((not value.strip() for value in ids)):
        raise ValueError('Cohort IDs must be nonempty and unique')
    if any((row.get('split') not in {'calibration', 'test'} for row in cohort)):
        raise ValueError('Cohort split must be calibration or test')
    by_id = {str(row['id']): row for row in cohort}
    groups = {split: {row['source_group_id'] for row in cohort if row['split'] == split} for split in ('calibration', 'test')}
    if not all(groups.values()) or groups['calibration'] & groups['test']:
        raise ValueError('Calibration/test must be nonempty and disjoint by source_group_id')
    for row in cohort:
        if not isinstance(row.get('human'), str) or not row['human'].strip():
            raise ValueError('Every source needs nonempty human text')
        if row['split'] == 'test' and (not isinstance(row.get('machine'), str) or not row['machine'].strip()):
            raise ValueError('Every test source needs nonempty machine text')
    systems, seen = (defaultdict(set), set())
    test_ids = {str(row['id']) for row in cohort if row['split'] == 'test'}
    for row in rewrites:
        source = str(row['source_id'])
        system = row['humanizer']
        key = (system, source)
        if source not in test_ids or not system or key in seen:
            raise ValueError('Rewrites require one result per system/test source, without duplicates')
        if row.get('original') != by_id[source]['machine'] or row.get('source_group_id') != by_id[source]['source_group_id']:
            raise ValueError('Rewrite original/group does not match its frozen cohort source')
        if not row.get('status'):
            raise ValueError('Every rewrite result requires an explicit status')
        if row['status'] == 'ok' and (not isinstance(row.get('modified'), str) or not row['modified'].strip()):
            raise ValueError('An ok rewrite must contain nonempty modified text')
        seen.add(key)
        systems[system].add(source)
    if not systems or any((sources != test_ids for sources in systems.values())):
        raise ValueError('Generation file is incomplete: finish all test sources for each included system')
    return by_id

def text_records(cohort, rewrites):
    """Include umani di calibrazione, controlli di test appaiati e riscritture riuscite."""
    sources = validate_inputs(cohort, rewrites)
    rows = []
    for source in cohort:
        common = {key: source.get(key, '') for key in ('source_group_id', 'domain', 'generator', 'split')}
        common['source_id'] = str(source['id'])
        roles = [('calibration_human', source['human'])] if source['split'] == 'calibration' else [('human', source['human']), ('machine', source['machine'])]
        for role, text in roles:
            rows.append(dict(common, role=role, humanizer='', rewrite_id='', text=text))
    for rewrite in rewrites:
        if rewrite['status'] != 'ok':
            continue
        source = sources[str(rewrite['source_id'])]
        rows.append({'source_id': str(source['id']), 'source_group_id': source['source_group_id'], 'domain': source['domain'], 'generator': source.get('generator', ''), 'split': 'test', 'role': 'rewrite', 'humanizer': rewrite['humanizer'], 'rewrite_id': rewrite['id'], 'text': rewrite['modified']})
    return rows

def _score_key(row):
    return (str(row['source_id']), row['role'], row.get('humanizer') or '')
