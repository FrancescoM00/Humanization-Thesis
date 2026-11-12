"""Confronta premesse NLI a unità singole, finestre contigue e documento intero."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from humanization import resources as resource
import sys
import time
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
from humanization.semantics.contextual_nli import LocalNLIPredictor, MODES, PROTOCOL, build_plan, score_plan

def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def focused_pilot(root=ROOT):
    """Pannello fisso di prove esplicative, distinto esplicitamente da un test casuale di accuratezza."""
    dipper_dir = root / 'data/processed/pilot_42/dipper'
    pairs = jsonl(dipper_dir / 'semantic/pairs.jsonl')
    alignment = json.loads((dipper_dir / 'semantic/alignment.json').read_text())
    baseline = {r['id']: r for r in csv_rows(dipper_dir / 'semantic/scores/nli.csv')}
    qwen_dir = root / 'data/processed/pilot_42/qwen'
    qwen = jsonl(qwen_dir / 'semantic/pairs.jsonl')
    alignment.update(json.loads((qwen_dir / 'semantic/alignment.json').read_text()))
    baseline.update({r['id']: r for r in csv_rows(qwen_dir / 'semantic/scores/nli.csv')})
    lookup = {(r['source_id'], r['humanizer']): r for r in [*pairs, *qwen]}
    sources = [('WritingPrompts:285:80b9ea16794068e3', 'narrative_fusion_15_to_5_assistant_review_not_gold'), ('Essay:198:ee27e99f1bc3dc32', 'diagnostic_list_fusion_and_omissions_mixed'), ('Reuters:417:b07c03bd035b9249', 'legal_challenges_omission'), ('Essay:476:30c42c3c6951daac', 'largely_conservative_case_assistant_review_not_gold')]
    selected = []
    selection = []

    def add(row, reason, origin):
        selected.append(row)
        selection.append({'id': row['id'], 'source_id': row.get('source_id', ''), 'humanizer': row.get('humanizer', ''), 'selection_reason': reason, 'origin': origin})
    for source, reason in sources:
        for system in ('dipper', 'qwen_humanize'):
            add(lookup[source, system], reason, 'fixed rewrite pilot')
    for source in ('Essay:198:ee27e99f1bc3dc32', 'Reuters:417:b07c03bd035b9249', 'WritingPrompts:562:785eead74dc70cd2'):
        add(lookup[source, 'identity'], 'identity_one_per_domain', 'fixed rewrite pilot')
    dipper_by_id = {r['id']: r for r in pairs}
    for row_id, reason in [('humanizer_1262d303f86a672c96ff1467', 'observed_number_bound_change_100_to_30'), ('humanizer_82c81ae99ce2c8599bc313e4', 'observed_currency_change_euros_to_francs')]:
        add(dipper_by_id[row_id], reason, 'fixed rewrite pilot')
    clean = jsonl(root / 'data/processed/semantic_clean/pairs.jsonl')
    for condition in ('negation', 'entity_substitution'):
        candidates = [r for r in clean if r['perturbation'] == condition and r['status'] == 'applied' and r['source_id'].startswith(('Chat', 'Claude', 'Dolly', 'GPT', 'Stable')) and (condition != 'entity_substitution' or r.get('metadata', {}).get('entity_type') in {'DATE', 'MONEY', 'PERCENT'})]
        chosen = min(candidates, key=lambda r: hashlib.sha256(('nli-context-focus-v1:' + r['id']).encode()).hexdigest())
        add(chosen, 'hash_selected_' + condition + '_controlled_candidate_not_gold', 'semantic clean')
        identity = next((r for r in clean if r['perturbation'] == 'identity' and r['source_id'] == chosen['source_id']))
        add(identity, 'same_source_identity_for_' + condition, 'semantic clean')
    extension = jsonl(root / 'data/processed/semantic_extension/pairs.jsonl')
    for operation in ('opai_polish', 'opai_paraphrase'):
        candidates = [r for r in extension if r['perturbation'] == operation]
        chosen = min(candidates, key=lambda r: hashlib.sha256(('nli-context-focus-v1:' + r['id']).encode()).hexdigest())
        add(chosen, 'hash_selected_' + operation + '_presumed_conservative_not_gold', 'semantic extension')
    for candidate in [root / 'data/processed/semantic_clean/alignment.json', root / 'data/processed/semantic_clean/alignment_recomputed.json', root / 'data/processed/semantic_extension/alignment.json']:
        if candidate.exists():
            data = json.loads(candidate.read_text())
            for row in selected:
                if row['id'] in data and all((data[row['id']].get(k + '_sha256') == hashlib.sha256(row[k].encode()).hexdigest() for k in ('original', 'modified'))):
                    alignment[row['id']] = data[row['id']]
    for candidate in [root / 'results/published/03_semantics/entity_clean/nli.csv', root / 'results/published/03_semantics/extension/nli.csv']:
        if candidate.exists():
            baseline.update({r['id']: r for r in csv_rows(candidate)})
    assert len(selected) == 19 and len({r['id'] for r in selected}) == 19
    return (selected, {r['id']: alignment[r['id']] for r in selected if r['id'] in alignment}, selection, baseline)

def _argument_error(message):
    raise ValueError(message)

def run(args):
    if args.focused_pilot:
        if args.pairs or args.alignment:
            _argument_error('Use either --focused-pilot or explicit --pairs/--alignment')
        pairs, alignment, selection, baseline = focused_pilot()
    else:
        if not args.pairs:
            _argument_error('--pairs is required without --focused-pilot')
        pairs = jsonl(args.pairs)
        alignment = json.loads(args.alignment.read_text()) if args.alignment else {}
        selection = [{'id': r['id'], 'selection_reason': 'all provided pairs'} for r in pairs]
        baseline = {}
    args.output.mkdir(parents=True, exist_ok=True)
    body = ''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in pairs))
    for path, text in [(args.output / 'pairs.jsonl', body), (args.output / 'alignment.json', json.dumps(alignment, ensure_ascii=False)), (args.output / 'selection.json', json.dumps(selection, indent=2, ensure_ascii=False) + '\n')]:
        if path.exists() and path.read_text() != text:
            raise ValueError('Refusing to replace different frozen inputs: ' + str(path))
        path.write_text(text)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    length_cache = {}
    print('Planning exact tokenizer pairs; repeated lengths are reused.', flush=True)
    plan = []
    for index, pair in enumerate(pairs, 1):
        plan.extend(build_plan(pair, alignment, tokenizer, max_pair_tokens=args.max_pair_tokens, modes=args.modes, length_cache=length_cache))
        if index % 25 == 0 or index == len(pairs):
            print(f'Planning pair {index}/{len(pairs)}; exact lengths cached: {len(length_cache)}', flush=True)
    length_cache.clear()
    unique = {(p, row['hypothesis']) for row in plan for p in row['premises']}
    info = {'protocol': PROTOCOL, 'status': 'planned', 'n_pairs': len(pairs), 'n_comparisons': len(unique), 'modes': args.modes, 'device': args.device, 'precision': 'float32', 'max_pair_tokens': args.max_pair_tokens, 'hypothesis_units': 'exact existing alignment units; otherwise sentence/chunks <=160 NLI tokens', 'premise_units': 'native contiguous sentence/chunks; window2/window3 include shorter windows', 'document_mode': 'exact full text only when actual tokenizer pair fits; incomplete coverage is not averaged', 'selection': 'focused explanatory challenge panel, not a random accuracy or AUC sample' if args.focused_pilot else 'all supplied pairs', 'pairs_sha256': hashlib.sha256(body.encode()).hexdigest(), 'code_sha256': hashlib.sha256((ROOT / 'src/humanization/semantics/contextual_nli.py').read_bytes()).hexdigest()}
    (args.output / 'plan.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2), flush=True)
    if args.plan_only:
        return
    start = time.monotonic()

    def progress(done, total):
        if done % 100 == 0 or done == total:
            print(f'Contextual NLI evidence {done}/{total}; {time.monotonic() - start:.1f}s', flush=True)
    predictor = LocalNLIPredictor(args.model, tokenizer, device=args.device, batch_size=args.batch_size, threads=args.threads, max_pair_tokens=args.max_pair_tokens, cache_path=args.output / 'contextual_probability_cache.jsonl', progress=progress)
    summaries, units = score_plan(plan, predictor, predictor.labels, low_support_threshold=args.low_support_threshold)
    import pandas as pd
    frame = pd.DataFrame(summaries)
    frame = frame.merge(pd.DataFrame(selection), on='id', validate='many_to_one')
    frame.to_csv(args.output / 'contextual_nli.csv', index=False)
    (args.output / 'contextual_nli_units.jsonl').write_text(''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in units)))
    if baseline:
        baseline_rows = []
        for pair in pairs:
            old = baseline.get(pair['id'])
            if old:
                for direction in ('forward', 'reverse'):
                    prefix = 'nli_' + direction + '_'
                    baseline_rows.append({'id': pair['id'], 'direction': direction, 'mode': 'historical_top3', **{key: float(old[prefix + key]) for key in ('mean', 'q10', 'min', 'low_support_fraction')}})
        pd.DataFrame(baseline_rows).to_csv(args.output / 'historical_top3.csv', index=False)
    info.update(status='executed', elapsed_seconds=time.monotonic() - start, new_evidence_scored=predictor.n_new, cache_entries=len(predictor.cache), peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, batch_size=args.batch_size, threads=args.threads)
    (args.output / 'contextual_nli_run.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2), flush=True)

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'pairs': None, 'alignment': None, 'focused_pilot': False, 'model': ROOT / 'data/models/nli', 'output': ROOT / 'results/new_runs/08_nli_calibration/context_pilot', 'device': 'cpu', 'batch_size': 2, 'threads': 2, 'max_pair_tokens': 384, 'low_support_threshold': 0.5, 'modes': list(MODES), 'plan_only': False}
    values.update(overrides)
    return SimpleNamespace(**values)
