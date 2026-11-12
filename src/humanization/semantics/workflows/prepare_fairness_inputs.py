"""Assembla i pannelli congelati del confronto equo e inizializza cache riutilizzabili senza inferenza."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from humanization.paths import project_root
ROOT = project_root()

def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def preserve(path, rows):
    text = ''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f'Existing frozen inputs differ: {path}; do not replace a scored panel silently')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text)

def _argument_error(message):
    raise ValueError(message)

def run(args):
    data = ROOT / 'data/processed/semantic_fairness'
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    old = read(ROOT / 'data/processed/semantic_extension/pairs.jsonl')
    initial = read(data / 'distributed_pairs.jsonl')
    expanded = read(data / 'expanded_positive_pairs.jsonl') + read(data / 'expanded_distributed_pairs.jsonl')
    combined = old + initial + expanded
    if len({x['id'] for x in combined}) != len(combined):
        raise ValueError('Frozen panels contain duplicate pair IDs')
    oldgroups = {x['source_group_id'] for x in old if x['dataset'] == 'opai_positive'}
    if oldgroups & {x['source_group_id'] for x in expanded}:
        raise ValueError('Expanded sources overlap the original OpAI sources')
    for path, rows in [(data / 'pairs.jsonl', combined), (data / 'new_pairs.jsonl', initial + expanded), (data / 'original_panel/pairs.jsonl', initial), (data / 'expanded_panel/pairs.jsonl', expanded)]:
        preserve(path, rows)
    for name in ['nli_probability_cache.jsonl', 'nli_pytorch_reference.jsonl']:
        target = data / name
        if not target.exists():
            shutil.copyfile(ROOT / 'data/processed/semantic_extension' / name, target)
    alignment_paths = [ROOT / 'data/processed/semantic_extension/alignment.json', data / 'original_panel/alignment.json', data / 'expanded_panel/alignment.json']
    if all((path.is_file() for path in alignment_paths)):
        alignment = {}
        for path in alignment_paths:
            for key, value in json.loads(path.read_text()).items():
                if key in alignment:
                    raise ValueError('Overlapping IDs in alignment components')
                alignment[key] = value
        if set(alignment) != {p['id'] for p in combined}:
            raise ValueError('Alignment does not cover the frozen input panels')
        target = data / 'alignment.json'
        if target.exists() and json.loads(target.read_text()) != alignment:
            raise ValueError('Existing alignment differs from frozen alignment components')
        if not target.exists():
            target.write_text(json.dumps(alignment, ensure_ascii=False))
    else:
        print('Run the embedding stages for original_panel and expanded_panel, then rerun this script to merge alignment.')
    metadata = {'n_pairs': len(combined), 'n_reused_pairs': len(old), 'n_original_distributed_pairs': len(initial), 'n_expanded_pairs': len(expanded), 'n_new_pairs': len(initial + expanded), 'pairs_sha256': hashlib.sha256((data / 'pairs.jsonl').read_bytes()).hexdigest(), 'original_proxy_labels_preserved': True, 'expanded_sources_disjoint_from_old_99': True}
    (output / 'preparation_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    provenance = ROOT / 'results/published/03_semantics/extension/models_provenance.json'
    if provenance.is_file() and (not (output / provenance.name).exists()):
        shutil.copyfile(provenance, output / provenance.name)
    print(json.dumps(metadata, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'output': ROOT / 'results/new_runs/03_semantics/fairness'}
    values.update(overrides)
    return SimpleNamespace(**values)
