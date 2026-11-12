"""Prepare calibration-only NLI pairs, then select a mode from saved scores.

Neural inference is performed by the existing score_contextual_nli.py command.
This driver never scores or selects from the previously inspected test split.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
MODES = ('single_all', 'window2', 'window3', 'document')
AGGREGATORS = ('mean', 'q10', 'min', 'low_support_fraction')
PROTOCOL = 'contextual_nli_calibration_selection_v1'

def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def csv_text(rows, columns):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()

def frozen_policy():
    """These operational sanity cutoffs are fixed before scoring new modes."""
    return dict(protocol=PROTOCOL, modes=list(MODES), aggregators=list(AGGREGATORS), selected_split='calibration', label_provenance='operational proxies, not human gold', score_orientation='higher means presumed preservation', bidirectional_score='min(forward, reverse); fraction uses 1-max(forward, reverse)', low_support_threshold=0.5, primary_requirements=dict(complete_labelled_coverage=1.0, complete_identity_coverage=1.0, identity_mean_min=0.95, identity_q10_min=0.9), selection='maximum pooled calibration AUC among eligible candidates', tie_tolerance=1e-12, tie_break_modes=list(MODES), tie_break_aggregators=list(AGGREGATORS), exclusions='test pairs and their metric scores are not used for selection', interpretation='development selection; independent human validation remains pending')

def _write_frozen(output, bodies):
    output = Path(output)
    for name, body in bodies.items():
        path = output / name
        if path.exists() and path.read_bytes() != body.encode():
            raise ValueError(f'Refusing to change frozen input {path}; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    for name, body in bodies.items():
        (output / name).write_bytes(body.encode())

def prepare_panel(pairs_path, alignment_path, output):
    rows = read_jsonl(pairs_path)
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Input pair IDs must be unique')
    negatives = [r for r in rows if r.get('dataset') == 'opai_matched_negative' and r.get('evaluation_split') == 'calibration']
    if not negatives:
        raise ValueError('No matched calibration negatives')
    positive_ids = {r.get('metadata', {}).get('matched_positive_id') for r in negatives}
    if None in positive_ids or not positive_ids <= by_id.keys():
        raise ValueError('Matched negatives need existing positive IDs')
    positives = [by_id[key] for key in sorted(positive_ids)]
    if any((r.get('dataset') != 'opai_positive' or r.get('evaluation_split') != 'calibration' for r in positives)):
        raise ValueError('A matched positive is not in the calibration split')
    selected = sorted([*positives, *negatives], key=lambda r: r['id'])
    if any((r.get('status') != 'applied' for r in selected)):
        raise ValueError('Calibration pairs must have applied status')
    groups = {r['source_group_id'] for r in selected}
    test_groups = {r['source_group_id'] for r in rows if r.get('evaluation_split') == 'test'}
    if groups & test_groups:
        raise ValueError('Calibration and test source groups overlap')
    for row in negatives:
        positive = by_id[row['metadata']['matched_positive_id']]
        if any((row[key] != positive[key] for key in ('original', 'source_group_id'))):
            raise ValueError('Matched corruption and positive have different originals or sources')
    alignment_source = json.loads(Path(alignment_path).read_text())
    alignment, labels = ({}, [])
    for row in selected:
        if row['id'] not in alignment_source:
            raise ValueError(f"Missing frozen alignment for {row['id']}")
        item = alignment_source[row['id']]
        if any((item.get(key + '_sha256') != sha(row[key]) for key in ('original', 'modified'))):
            raise ValueError('Alignment text hash mismatch')
        alignment[row['id']] = item
        positive = row['id'] in positive_ids
        labels.append(dict(id=row['id'], source_group_id=row['source_group_id'], source_id=row['source_id'], kind='presumed_positive' if positive else 'intentional_alteration', preserved=1 if positive else 0, condition=row['perturbation'], matched_positive_id='' if positive else row['metadata']['matched_positive_id'], evaluation_split='calibration'))
    originals = {}
    for row in positives:
        originals.setdefault((row['source_group_id'], row['original']), row)
    identities = []
    for (group, original), row in sorted(originals.items()):
        row_id = 'nli_calibration_identity:' + sha(json.dumps([group, original], ensure_ascii=False))[:24]
        identity = dict(id=row_id, pair_id=row_id, source_id=row['source_id'], source_group_id=group, original=original, modified=original, domain=row.get('domain', ''), generator=row.get('generator', ''), dataset='nli_calibration_identity', perturbation='identity', evaluation_split='calibration', status='applied', metadata=dict(semantic_interpretation='exact_identity', derived_from=row['id']))
        identities.append(identity)
        chunks = alignment[row['id']]['original_chunks']
        alignment[row_id] = dict(original_sha256=sha(original), modified_sha256=sha(original), original_chunks=chunks, modified_chunks=chunks, chunk_token_budget=alignment[row['id']].get('chunk_token_budget', 160))
        labels.append(dict(id=row_id, source_group_id=group, source_id=row['source_id'], kind='identity', preserved='', condition='identity', matched_positive_id='', evaluation_split='calibration'))
    all_pairs = [*selected, *identities]
    policy = frozen_policy()
    bodies = {'pairs.jsonl': ''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in all_pairs)), 'alignment.json': json.dumps(alignment, ensure_ascii=False), 'labels.csv': csv_text(labels, list(labels[0])), 'policy.json': json.dumps(policy, ensure_ascii=False, indent=2) + '\n'}
    excluded_test_negatives = [r for r in rows if r.get('dataset') == 'opai_matched_negative' and r.get('evaluation_split') == 'test']
    excluded_test_positive_ids = {r.get('metadata', {}).get('matched_positive_id') for r in excluded_test_negatives}
    manifest = dict(protocol=PROTOCOL, preparation='calibration only; no neural inference', n_positive=len(positives), n_negative=len(negatives), n_identity=len(identities), n_labelled=len(selected), n_pairs=len(all_pairs), n_source_groups=len(groups), excluded_test_rows=sum((r.get('evaluation_split') == 'test' for r in rows)), excluded_matched_test_positive=len(excluded_test_positive_ids - {None}), excluded_matched_test_negative=len(excluded_test_negatives), test_source_overlap=0, labels_are_human_gold=False, source_files={'pairs_sha256': file_sha(pairs_path), 'alignment_sha256': file_sha(alignment_path)}, files_sha256={name: sha(body) for name, body in bodies.items()})
    bodies['preparation_manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    _write_frozen(output, bodies)
    return manifest

def _complete(value):
    if value not in (True, False, 'True', 'False', 'true', 'false', '1', '0'):
        raise ValueError(f'Invalid coverage flag {value!r}')
    return value in (True, 'True', 'true', '1')

def combine_directions(forward, reverse, aggregator):
    for row in (forward, reverse):
        complete = _complete(row['complete'])
        if 'coverage_fraction' in row and complete != (float(row['coverage_fraction']) == 1.0):
            raise ValueError('Coverage fraction disagrees with completeness flag')
        if 'n_hypotheses' in row and 'n_eligible_hypotheses' in row:
            if int(row['n_hypotheses']) < 1 or complete != (int(row['n_hypotheses']) == int(row['n_eligible_hypotheses'])):
                raise ValueError('Hypothesis coverage disagrees with completeness flag')
    if not _complete(forward['complete']) or not _complete(reverse['complete']):
        return math.nan
    a, b = (float(forward[aggregator]), float(reverse[aggregator]))
    if not all((math.isfinite(x) and -1e-06 <= x <= 1 + 1e-06 for x in (a, b))):
        raise ValueError('Complete NLI aggregates must be finite probabilities/fractions')
    return 1 - max(a, b) if aggregator == 'low_support_fraction' else min(a, b)

def select_candidate(candidates, policy):
    eligible = [r for r in candidates if r['eligible_primary']]
    if not eligible:
        return None
    best = max((r['calibration_auc'] for r in eligible))
    tied = [r for r in eligible if best - r['calibration_auc'] <= policy['tie_tolerance']]
    return min(tied, key=lambda r: (policy['tie_break_modes'].index(r['mode']), policy['tie_break_aggregators'].index(r['aggregator'])))

def _auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))

def _bootstrap_auc(labels, scores, groups, *, bootstrap, seed):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    if not bootstrap:
        return (math.nan, math.nan)
    names = sorted(set(groups))
    index = {name: i for i, name in enumerate(names)}
    group_index = np.array([index[name] for name in groups])
    y = np.array(labels)
    rng, values = (np.random.default_rng(seed), [])
    for _ in range(bootstrap):
        weights = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))[group_index]
        if weights[y == 0].sum() and weights[y == 1].sum():
            values.append(roc_auc_score(y, scores, sample_weight=weights))
    return tuple((float(x) for x in np.quantile(values, [0.025, 0.975]))) if values else (math.nan, math.nan)

def analyze_scores(input_dir, scores_path, output, *, bootstrap=2000, seed=42):
    import numpy as np
    input_dir, output = (Path(input_dir), Path(output))
    preparation = json.loads((input_dir / 'preparation_manifest.json').read_text())
    for name, expected in preparation['files_sha256'].items():
        if file_sha(input_dir / name) != expected:
            raise ValueError(f'Prepared input changed after freezing: {name}')
    policy = json.loads((input_dir / 'policy.json').read_text())
    if policy != frozen_policy():
        raise ValueError('Prepared policy differs from this selection implementation')
    with (input_dir / 'labels.csv').open(newline='') as handle:
        labels = list(csv.DictReader(handle))
    if not labels or any((r['evaluation_split'] != 'calibration' for r in labels)):
        raise ValueError('Only frozen calibration labels can be analyzed')
    by_id = {r['id']: r for r in labels}
    if len(by_id) != len(labels):
        raise ValueError('Frozen labels contain duplicate IDs')
    with Path(scores_path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(r['id'], r['direction'], r['mode']): r for r in rows}
    expected = {(key, direction, mode) for key in by_id for direction in ('forward', 'reverse') for mode in MODES}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError('Scores must cover each frozen ID, both directions and all four modes exactly once')
    if bootstrap < 0:
        raise ValueError('bootstrap must be nonnegative')
    inputs = dict(protocol=PROTOCOL, policy_sha256=file_sha(input_dir / 'policy.json'), labels_sha256=file_sha(input_dir / 'labels.csv'), scores_sha256=file_sha(scores_path), bootstrap=bootstrap, seed=seed)
    guard = output / 'analysis_inputs.json'
    if guard.exists() and json.loads(guard.read_text()) != inputs:
        raise ValueError('Analysis inputs changed; use a new output directory')
    labelled = [r for r in labels if r['kind'] != 'identity']
    identities = [r for r in labels if r['kind'] == 'identity']
    if not identities or {r['preserved'] for r in labelled} != {'0', '1'}:
        raise ValueError('Calibration needs positives, negatives and separate identities')
    candidates, condition_rows, score_rows = ([], [], [])
    for mode in MODES:
        for aggregator in AGGREGATORS:
            values = {key: combine_directions(lookup[key, 'forward', mode], lookup[key, 'reverse', mode], aggregator) for key in by_id}
            for row in labels:
                score_rows.append(dict(**row, mode=mode, aggregator=aggregator, preservation_score=values[row['id']], complete=math.isfinite(values[row['id']])))
            labelled_complete = all((math.isfinite(values[r['id']]) for r in labelled))
            identity_complete = all((math.isfinite(values[r['id']]) for r in identities))
            iv = np.array([values[r['id']] for r in identities])
            im = float(iv.mean()) if identity_complete else math.nan
            iq = float(np.quantile(iv, 0.1)) if identity_complete else math.nan
            auc = _auc([int(r['preserved']) for r in labelled], [values[r['id']] for r in labelled]) if labelled_complete else math.nan
            eligible = labelled_complete and identity_complete and (im >= policy['primary_requirements']['identity_mean_min']) and (iq >= policy['primary_requirements']['identity_q10_min'])
            low, high = _bootstrap_auc([int(r['preserved']) for r in labelled], [values[r['id']] for r in labelled], [r['source_group_id'] for r in labelled], bootstrap=bootstrap, seed=seed) if labelled_complete else (math.nan, math.nan)
            reasons = []
            if not labelled_complete:
                reasons.append('incomplete_labelled_coverage')
            if not identity_complete:
                reasons.append('incomplete_identity_coverage')
            if identity_complete and im < policy['primary_requirements']['identity_mean_min']:
                reasons.append('identity_mean_below_cutoff')
            if identity_complete and iq < policy['primary_requirements']['identity_q10_min']:
                reasons.append('identity_q10_below_cutoff')
            candidates.append(dict(mode=mode, aggregator=aggregator, n_labelled=len(labelled), n_labelled_complete=sum((math.isfinite(values[r['id']]) for r in labelled)), n_identity=len(identities), n_identity_complete=sum((math.isfinite(values[r['id']]) for r in identities)), identity_mean=im, identity_q10=iq, identity_median=float(np.median(iv)) if identity_complete else math.nan, identity_min=float(iv.min()) if identity_complete else math.nan, calibration_auc=auc, calibration_auc_ci95_low=low, calibration_auc_ci95_high=high, eligible_primary=eligible, exclusion_reasons=';'.join(reasons)))
            for condition in sorted({r['condition'] for r in labelled if r['preserved'] == '0'}):
                negative = [r for r in labelled if r['preserved'] == '0' and r['condition'] == condition]
                positive_ids = {r['matched_positive_id'] for r in negative}
                sub = [r for r in labelled if r['id'] in positive_ids] + negative
                if any((key not in by_id or by_id[key]['preserved'] != '1' for key in positive_ids)):
                    raise ValueError('Condition analysis needs its matched calibration positives')
                complete = all((math.isfinite(values[r['id']]) for r in sub))
                condition_rows.append(dict(mode=mode, aggregator=aggregator, condition=condition, n_positive=len(positive_ids), n_negative=len(negative), complete=complete, calibration_auc=_auc([int(r['preserved']) for r in sub], [values[r['id']] for r in sub]) if complete else math.nan))
    selected = select_candidate(candidates, policy)
    selection = dict(protocol=PROTOCOL, status='selected_on_calibration' if selected else 'no_eligible_candidate', selected=selected, policy=policy, n_positive=preparation['n_positive'], n_negative=preparation['n_negative'], n_identity=preparation['n_identity'], n_source_groups=preparation['n_source_groups'], test_used_for_selection=False, human_validation_completed=False, note='Operational development choice on presumed labels; not gold accuracy or independent validation')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'analysis_inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    for name, data in [('candidate_summary.csv', candidates), ('auc_by_condition.csv', condition_rows), ('calibration_scores.csv', score_rows)]:
        (output / name).write_text(csv_text(data, list(data[0])))

    def clean_json(value):
        if isinstance(value, float) and (not math.isfinite(value)):
            return None
        if isinstance(value, dict):
            return {key: clean_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean_json(item) for item in value]
        return value
    selection_clean = clean_json(selection)
    (output / 'selection.json').write_text(json.dumps(selection_clean, ensure_ascii=False, indent=2) + '\n')
    table = ['| Modalità | Aggregatore | AUC calibrazione | Media identità | q10 identità | Primaria ammissibile |', '| --- | --- | ---: | ---: | ---: | --- |']
    fmt = lambda x: f'{x:.4f}' if math.isfinite(x) else 'mancante'
    for row in candidates:
        table.append(f"| {row['mode']} | {row['aggregator']} | {fmt(row['calibration_auc'])} | {fmt(row['identity_mean'])} | {fmt(row['identity_q10'])} | {('sì' if row['eligible_primary'] else 'no')} |")
    verdict = f"Scelta operativa: **{selected['mode']} + {selected['aggregator']}**, AUC di calibrazione {selected['calibration_auc']:.4f}." if selected else '**Nessun candidato soddisfa tutti i requisiti: nessuna modalità primaria viene forzata.**'
    table_text = '\n'.join(table)
    report = f"# Scelta NLI sul solo pannello di calibrazione\n\n{verdict}\n\nSono incluse {preparation['n_positive']} coppie di conservazione presunta, {preparation['n_negative']} alterazioni intenzionali e {preparation['n_identity']} identità separate, su {preparation['n_source_groups']} gruppi di fonte. I positivi OpAI e le perturbazioni sono proxy, non etichette umane gold. Le {preparation['excluded_matched_test_positive']} coppie positive e {preparation['excluded_matched_test_negative']} alterate del test già esaminato non sono usate per selezionare questa configurazione.\n\nLa policy viene salvata prima delle nuove inferenze. Si richiedono copertura completa in entrambe le direzioni per tutte le coppie e tutte le identità, media degli score identità almeno 0,95 e decimo percentile almeno 0,90. Sono criteri operativi di coerenza, non soglie universali di validità NLI. Lo score è il minimo delle due direzioni dell'aggregatore; per la quota non supportata è uno meno la quota massima. Si massimizza l'AUC di calibrazione; parità entro 10^-12 preferisce la premessa più corta, poi mean/q10/min/quota. Se nessuno passa, non si sceglie un vincitore.\n\n{table_text}\n\nL'AUC non viene calcolata eliminando le coppie prive di copertura. Gli intervalli nel [CSV](candidate_summary.csv) ricampionano i gruppi di fonte ({bootstrap} repliche, seed {seed}); descrivono l'incertezza di sviluppo e non correggono l'ottimismo della selezione. Le [AUC per alterazione](auc_by_condition.csv) usano soltanto i positivi collegati a quella condizione. Le identità verificano coerenza, non dimostrano accuratezza sulle parafrasi o sulle fusioni di frasi.\n\nQuesta è una scelta di sviluppo successiva allo studio sulle 126 fonti, non una preregistrazione retroattiva. Le modalità secondarie restano disponibili; non si trasferiscono automaticamente AUC o soglie ai veri riscrittori. La verifica con le annotazioni umane reference ancora da raccogliere resta necessaria e non entra nella scelta qui effettuata.\n"
    (output / 'report.md').write_text(report)
    return selection_clean

def _argument_error(message):
    raise ValueError(message)

def run(args):
    result = prepare_panel(args.pairs, args.alignment, args.output) if args.command == 'prepare' else analyze_scores(args.input, args.scores, args.output, bootstrap=args.bootstrap, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'pairs': None, 'alignment': None, 'output': None, 'input': None, 'scores': None, 'bootstrap': 2000, 'seed': 42}
    values.update(overrides)
    return SimpleNamespace(**values)
