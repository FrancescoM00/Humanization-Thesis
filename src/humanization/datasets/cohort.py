"""Freeze source-disjoint humanizer cohorts before observing detector scores."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_samples(samples):
    by_pair = defaultdict(dict)
    for row in samples:
        label = int(row['label'])
        if label not in (0, 1) or label in by_pair[row['pair_id']]:
            raise ValueError('Expected exactly one human and one machine per source pair')
        by_pair[row['pair_id']][label] = row
    rows = []
    for pair, members in sorted(by_pair.items()):
        if set(members) != {0, 1}:
            raise ValueError(f'Incomplete pair: {pair}')
        h, m = members[0], members[1]
        if h['group_id'] != m['group_id'] or h['domain'] != m['domain']:
            raise ValueError(f'Inconsistent source group: {pair}')
        rows.append({'id': pair, 'source_id': pair, 'source_group_id': m['group_id'],
                     'domain': m['domain'], 'generator': m['generator'],
                     'human': h['text'], 'machine': m['text'],
                     'matched_n_words': int(m['matched_n_words'])})
    return rows


def select_cohorts(rows, *, per_cell=2, calibration_per_domain=100,
                   reserve_per_cell=6, seed=42):
    """Nested test panels; calibration excludes the entire reserved study panel.

    Ranking depends on seed and identifiers only. Feature values, detector
    scores and humanizer outputs never enter the selection.
    """
    if not 1 <= per_cell <= reserve_per_cell or calibration_per_domain < 1:
        raise ValueError('Require 1 <= per_cell <= reserve_per_cell and positive calibration size')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source IDs')
    rank = lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()
    cells = defaultdict(list)
    for row in rows:
        cells[row['domain'], row['generator']].append(row)
    domains = sorted({r['domain'] for r in rows})
    generators = sorted({r['generator'] for r in rows})
    if set(cells) != {(d, g) for d in domains for g in generators}:
        raise ValueError('The source cohort must cover every domain/generator cell')
    reserved, pilot, used = [], [], set()
    for key in sorted(cells):
        chosen = []
        for row in sorted(cells[key], key=rank):
            if row['source_group_id'] not in used:
                chosen.append(row)
                used.add(row['source_group_id'])
                if len(chosen) == reserve_per_cell:
                    break
        if len(chosen) < reserve_per_cell:
            raise ValueError(f'Insufficient disjoint sources for study cell {key}')
        reserved += chosen
        pilot += chosen[:per_cell]
    calibration = []
    for domain in domains:
        chosen = []
        for row in sorted((r for r in rows if r['domain'] == domain), key=rank):
            if row['source_group_id'] not in used:
                chosen.append(row)
                used.add(row['source_group_id'])
                if len(chosen) == calibration_per_domain:
                    break
        if len(chosen) < calibration_per_domain:
            raise ValueError(f'Insufficient calibration humans in {domain}')
        calibration += chosen
    smoke = [next(r for r in pilot if r['domain'] == d) for d in domains]
    panels = {}
    for name, tests in [('smoke', smoke), ('pilot', pilot), ('study', reserved)]:
        panel = [dict(r, split='calibration') for r in calibration]
        panel += [dict(r, split='test') for r in tests]
        assert not ({r['source_group_id'] for r in calibration}
                    & {r['source_group_id'] for r in tests})
        panels[name] = panel
    return panels


def reconstruct_samples(raw_dir, metadata_csv, historical_manifest):
    """Rebuild the existing short passages, not the original matching search.

    The published feature table contains the selected source rows and exact
    prefix lengths. Its text IDs verify the reconstructed strings. This avoids
    uploading texts or the cloud virtual environment to the repository.
    """
    from humanization.datasets.mgtbench import DOMAINS, text_hash, process_spaces, trim_incomplete_tail
    from humanization.stylometry.features import load_nlp
    from humanization.datasets.length_matching import sentence_prefixes

    raw_dir = Path(raw_dir)
    manifest = json.loads(Path(historical_manifest).read_text())
    for item in manifest['source_files'].values():
        if fingerprint(raw_dir / item['file']) != item['sha256']:
            raise ValueError(f"Raw CSV differs from the experiment: {item['file']}")
    with Path(metadata_csv).open(newline='') as handle:
        metadata = list(csv.DictReader(handle))
    frames = {}
    for domain, short in DOMAINS.items():
        with (raw_dir / f'{short}_LLMs.csv').open(newline='', encoding='utf-8-sig') as handle:
            frames[domain] = list(csv.DictReader(handle))
    nlp = load_nlp()
    samples = []
    keys = ['id', 'pair_id', 'group_id', 'domain', 'generator', 'label',
            'source_row', 'source_id', 'matched_n_words']
    for item in metadata:
        row = {key: item[key] for key in keys}
        raw = frames[row['domain']][int(row['source_row'])][row['generator']]
        count = int(row['matched_n_words'])
        # Reproduce both normalization passes of the original matcher. Its
        # caller cleans the full text before sentence_prefixes cleans it again.
        cleaned = trim_incomplete_tail(process_spaces(raw))
        prefixes = sentence_prefixes(cleaned, nlp.make_doc, count, count)
        if count not in prefixes:
            raise ValueError(f"Cannot reconstruct the historical prefix: {row['id']}")
        row['text'] = prefixes[count]
        expected = f"{row['generator']}:{text_hash(row['text'])[:24]}"
        if expected != row['id']:
            raise ValueError(f"Reconstructed text hash differs: {row['id']}")
        samples.append(row)
    return samples


def save_cohorts(panels, output, provenance):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(provenance, selection_uses_detector_scores=False,
                    calibration_excludes_reserved_study_sources=True,
                    estimand='90–180-word sentence-ended initial passages, not complete documents',
                    panels={})
    for name, rows in panels.items():
        body = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows)
        path = output / f'{name}.jsonl'
        if path.exists() and path.read_text() != body:
            raise ValueError(f'Refusing to replace a different frozen cohort: {path}')
        path.write_text(body)
        manifest['panels'][name] = {
            'file': path.name, 'sha256': fingerprint(path),
            'calibration_humans': sum(r['split'] == 'calibration' for r in rows),
            'test_sources': sum(r['split'] == 'test' for r in rows),
            'test_cells': dict(Counter(f"{r['domain']}/{r['generator']}"
                                      for r in rows if r['split'] == 'test')),
        }
    (output / 'cohort_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
