"""Congela l'esprimento delle cautele dopo la revisione dell'assistente senza score e prima dell'inferenza."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from humanization.semantics.hedging_controls import source_pairs
from humanization.semantics.perturbations import _stable_hex

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def write_jsonl(path, rows):
    path.write_text(''.join((json.dumps(row, ensure_ascii=False) + '\n' for row in rows)))

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def finalize(output, protocol_path, review_paths):
    output = output.resolve()
    if (output / 'review_freeze.json').exists() or (output / 'pre_review').exists():
        raise FileExistsError('This panel was already finalized or archived; do not overwrite it')
    if any(output.rglob('nli*')):
        raise RuntimeError('Refusing source selection after NLI artifacts exist')
    protocol = json.loads(protocol_path.read_text())
    seed = int(protocol['seed'])
    initial = read_jsonl(output / 'sources.jsonl')
    reserve = read_jsonl(output / 'candidate_reserve.jsonl')
    initial_pairs = [p for source in initial for p in source_pairs(source, seed)]
    if {p['id'] for p in initial_pairs} != {p['id'] for p in read_jsonl(output / 'pairs.jsonl')}:
        raise ValueError('Prepared source identities do not match the existing pair file')
    reviews = {}
    for path in review_paths:
        for row in csv.DictReader(path.open(newline='', encoding='utf-8')):
            if row['id'] in reviews:
                raise ValueError('Duplicate reviewed pair: ' + row['id'])
            if row.get('reviewer_source') != 'assistant' or row.get('validity') not in {'valid', 'qualified', 'invalid'}:
                raise ValueError('Every variant requires an explicit assistant validity judgment')
            reviews[row['id']] = row
    expected = {p['id'] for source in initial + reserve for p in source_pairs(source, seed)}
    if set(reviews) != expected:
        raise ValueError('Reviews must cover exactly all selected and reserve variants')

    def invalid(source):
        return [reviews[p['id']] for p in source_pairs(source, seed) if reviews[p['id']]['validity'] == 'invalid']
    timestamp = datetime.now(timezone.utc).isoformat()
    available = [source for source in reserve if not invalid(source)]
    selected, replacements = ([], [])
    for source in initial:
        rejected = invalid(source)
        if not rejected:
            selected.append(source)
            continue
        same_domain = [r for r in available if r['domain'] == source['domain']]
        if not available:
            raise ValueError('No reviewed reserve remains; no files have been changed')
        replacement = (same_domain or available)[0]
        available.remove(replacement)
        selected.append(replacement)
        replacements.append({'old_source_id': source['source_id'], 'new_source_id': replacement['source_id'], 'old_domain': source['domain'], 'new_domain': replacement['domain'], 'same_domain': bool(same_domain), 'rejected_pair_ids': '|'.join((r['id'] for r in rejected)), 'reason': ' | '.join((r['notes'] for r in rejected)), 'replacement_before_inference_utc': timestamp})
    if len({source['source_group_id'] for source in selected}) != len(initial):
        raise ValueError('Final source groups must remain independent')
    pairs = [p for source in selected for p in source_pairs(source, seed)]
    if sum((p['intended_label'] for p in pairs)) < 100:
        raise ValueError('Fewer than 100 intended alterations')
    archive = output / 'pre_review'
    archive.mkdir()
    names = ['pairs.jsonl', 'sources.jsonl', 'review_pairs.csv', 'human_annotation.csv', 'human_annotation_key.csv', 'preparation_manifest.json', 'candidate_reserve.jsonl', 'protocol.json']
    for name in names:
        shutil.copy2(output / name, archive / name)
    for path in review_paths:
        shutil.copy2(path, archive / path.name)
    write_csv(archive / 'all_assistant_reviews.csv', list(reviews.values()))
    write_jsonl(output / 'sources.jsonl', selected)
    write_jsonl(output / 'pairs.jsonl', pairs)
    write_csv(output / 'assistant_review.csv', [reviews[p['id']] for p in pairs])
    review_rows = []
    for p in pairs:
        review_rows.append({**{k: p[k] for k in ['id', 'source_id', 'source_group_id', 'condition', 'domain', 'generator', 'original', 'modified', 'target_original', 'target_modified']}, **{k: reviews[p['id']][k] for k in ['validity', 'reviewer_source', 'notes']}})
    write_csv(output / 'review_pairs.csv', review_rows)
    blind = sorted(pairs, key=lambda p: _stable_hex(seed, 'human-blind-order', p['id']))
    write_csv(output / 'human_annotation.csv', [{'review_id': f'H{i:03d}', 'original': p['original'], 'modified': p['modified'], 'meaning_preserved': '', 'change_type': '', 'notes': ''} for i, p in enumerate(blind, 1)])
    write_csv(output / 'human_annotation_key.csv', [{'review_id': f'H{i:03d}', **{k: p[k] for k in ['id', 'source_group_id', 'condition', 'intended_label']}} for i, p in enumerate(blind, 1)])
    if replacements:
        write_csv(output / 'replacements.csv', replacements)
    manifest = json.loads((archive / 'preparation_manifest.json').read_text())
    manifest.update({'review_applied_at_utc': timestamp, 'n_sources': len(selected), 'n_pairs': len(pairs), 'n_intended_alterations': sum((p['intended_label'] for p in pairs)), 'n_presumed_preserved': sum((not p['intended_label'] for p in pairs)), 'sources_per_domain': dict(Counter((r['domain'] for r in selected))), 'sources_per_generator': dict(Counter((r['generator'] for r in selected))), 'sources_per_family': dict(Counter((r['family'] for r in selected))), 'n_replaced_sources': len(replacements), 'n_unused_reviewed_eligible_reserves': len(available), 'assistant_review_counts': dict(Counter((reviews[p['id']]['validity'] for p in pairs))), 'initial_pairs_sha256': sha256(archive / 'pairs.jsonl'), 'pairs_sha256': sha256(output / 'pairs.jsonl'), 'finalization_code_sha256': sha256(Path(__file__)), 'human_gold_labels_available': False})
    (output / 'preparation_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    final_names = ['sources.jsonl', 'pairs.jsonl', 'assistant_review.csv', 'review_pairs.csv', 'human_annotation.csv', 'human_annotation_key.csv', 'preparation_manifest.json']
    freeze = {'created_at_utc': timestamp, 'before_new_nli_inference': True, 'selection_uses_scores': False, 'reviewer_source': 'assistant', 'human_gold_labels_available': False, 'replacement_rule': 'reject any invalid variant; first same-domain eligible reserve, else first other-domain; qualified retained', 'n_replaced_sources': len(replacements), 'protocol_sha256': sha256(protocol_path), 'input_sha256': {str(p.relative_to(output)): sha256(p) for p in sorted(archive.iterdir())}, 'final_sha256': {name: sha256(output / name) for name in final_names}}
    if replacements:
        freeze['final_sha256']['replacements.csv'] = sha256(output / 'replacements.csv')
    (output / 'review_freeze.json').write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + '\n')
    return manifest

def _argument_error(message):
    raise ValueError(message)

def run(args):
    print(json.dumps(finalize(args.output, args.protocol, args.review), ensure_ascii=False, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'output': ROOT / 'results/new_runs/hedging', 'protocol': ROOT / 'configs/protocols/hedging.json', 'review': None}
    values.update(overrides)
    return SimpleNamespace(**values)
