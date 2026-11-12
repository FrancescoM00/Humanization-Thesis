"""Confronta le politiche sulle entità nella stessa coorte e rifiuta modifiche estranee.

Di norma legge coppie e score già calcolati senza modificare gli input.
La modalità esplicita --freeze-unchanged-alignment riusa invece il recupero
precedente sulle coppie identiche prima della pianificazione NLI, mantenendo
un nuovo recupero per le entità. Nessuna modalità esegue inferenze dei modelli
o modifica risultati precedenti. I contatori di esecuzione non sono metriche
semantiche.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from humanization.paths import project_root
ROOT = project_root()
COUNTERS = {'embedding_unique_chunks_encoded_cumulative', 'bert_unique_chunks_encoded_cumulative', 'bert_unique_chunk_pairs_scored_cumulative', 'nli_unique_pairs_scored_cumulative', 'nli_new_pairs_scored'}
METRICS = {'local_nli_forward_mean': 'Entailment locale originale → alterato', 'local_nli_reverse_mean': 'Entailment locale alterato → originale', 'local_nli_bidirectional_mean': 'Minimo delle due medie NLI locali', 'nli_forward_min': 'Supporto minimo documento, forward', 'nli_reverse_min': 'Supporto minimo documento, reverse', 'nli_bidirectional_min': 'Supporto minimo documento, bidirezionale', 'embedding_cosine': 'Coseno documento', 'local_embedding_cosine': 'Coseno frase modificata', 'bert_chunk_f1': 'BERTScore F1 documento (protocollo chunk, non riscalato)', 'local_bert_f1': 'BERTScore F1 frase modificata (non riscalato)'}

def read_pairs(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    mapping = {p['id']: p for p in records}
    if len(mapping) != len(records):
        raise ValueError(f'Duplicate pair IDs: {path}')
    return mapping

def unchanged_pairs(old, new):

    def may_change(pair):
        return pair.get('dataset') == 'controlled' and pair['perturbation'] == 'entity_substitution' or (pair.get('dataset') == 'local_control' and pair['perturbation'] == 'identity_for_entity_substitution')
    a = {k: p for k, p in old.items() if not may_change(p)}
    b = {k: p for k, p in new.items() if not may_change(p)}
    if set(a) != set(b):
        raise ValueError(f'Non-entity IDs differ: missing={len(set(a) - set(b))}, added={len(set(b) - set(a))}. Use the same cohort, seed and --opai-file as the original run.')
    differences = [key for key in a if a[key] != b[key]]
    if differences:
        first = differences[0]
        fields = [k for k in set(a[first]) | set(b[first]) if a[first].get(k) != b[first].get(k)]
        raise ValueError(f'Non-entity inputs changed in {len(differences)} rows; first={first}, fields={fields}')
    return sorted(a)

def freeze_unchanged_alignment(args, old, new, ids):
    """Congela un fattore controllato prima del NLI, senza selezionare score dei modelli."""
    if sorted(ids) != unchanged_pairs(old, new):
        raise ValueError('Freeze IDs must be exactly the validated identical non-entity pairs')
    blockers = [args.clean_data / 'nli_jobs.jsonl']
    blockers += [args.clean_output / name for name in ['nli.csv', 'nli_run.json', 'nli_plan.json', 'nli_progress.json', 'nli_precompute_run.json', 'nli_precompute_progress.json']]
    existing = [str(path) for path in blockers if path.exists()]
    if existing:
        raise ValueError(f'Freeze must precede NLI planning and inference; existing artifacts: {existing}')
    manifest_path = args.clean_output / 'unchanged_alignment_reuse.json'
    backup_path = args.clean_data / 'alignment_recomputed.json'
    if manifest_path.exists() or backup_path.exists():
        raise FileExistsError('An alignment freeze or its fresh snapshot already exists; refusing an implicit replacement')
    a_run = json.loads((args.baseline_output / 'embedding_run.json').read_text())
    b_run = json.loads((args.clean_output / 'embedding_run.json').read_text())
    if not a_run.get('model_directory') or not b_run.get('model_directory'):
        raise ValueError('Embedding model directories are missing from run provenance')
    model = Path(a_run['model_directory']).resolve()
    if model != Path(b_run['model_directory']).resolve():
        raise ValueError('Embedding checkpoint directories differ; cannot reuse retrieval')
    old_code_hash, code_hash = (a_run.get('metrics_code_sha256'), b_run.get('metrics_code_sha256'))
    current_code_hash = hashlib.sha256((ROOT / 'src/humanization/semantics/metrics.py').read_bytes()).hexdigest()
    if not old_code_hash or not code_hash or code_hash != current_code_hash:
        raise ValueError('Embedding metrics_code_sha256 is missing or does not match the current metrics file')
    nli_run_path = args.baseline_output / 'nli_run.json'
    historical_nli_hash = json.loads(nli_run_path.read_text()).get('metrics_code_sha256') if nli_run_path.exists() else None
    same_embedding_code = old_code_hash == code_hash
    if not same_embedding_code and historical_nli_hash != code_hash:
        raise ValueError('Embedding metrics_code_sha256 differs and historical NLI code does not match current code')
    embedding_checks = validate_scores(pd.read_csv(args.baseline_output / 'embedding.csv'), pd.read_csv(args.clean_output / 'embedding.csv'), ids, args.atol, args.rtol)
    baseline = json.loads((args.baseline_data / 'alignment.json').read_text())
    path = args.clean_data / 'alignment.json'
    alignment = json.loads(path.read_text())
    if set(baseline) != set(old) or set(alignment) != set(new):
        raise ValueError('Alignment does not exactly cover the corresponding input pairs')
    checks = ['original_sha256', 'modified_sha256', 'original_chunks', 'modified_chunks', 'chunk_token_budget', 'retrieval_top_k']
    differences = []
    unchanged_settings = set()
    for key in ids:
        a, b = (baseline[key], alignment[key])
        for field in checks:
            if field not in a or field not in b or a[field] != b[field]:
                raise ValueError(f'Cannot reuse {key}: {field} differs or is absent')
        if a['chunk_token_budget'] < 1 or a['retrieval_top_k'] < 1:
            raise ValueError(f'Invalid chunk/retrieval settings in {key}')
        unchanged_settings.add((a['chunk_token_budget'], a['retrieval_top_k']))
        for side in ['original', 'modified']:
            expected = hashlib.sha256(new[key][side].encode('utf-8')).hexdigest()
            if a[side + '_sha256'] != expected:
                raise ValueError(f'Input hash and cached {side} hash disagree: {key}')
        for direction, evidence in [('forward', 'original_chunks'), ('reverse', 'modified_chunks')]:
            ac, bc = (a[direction + '_candidates'], b[direction + '_candidates'])
            if len(ac) != len(bc):
                raise ValueError(f'Query counts differ in {key}, {direction}')
            for query, (av, bv) in enumerate(zip(ac, bc)):
                if av != bv:
                    old_texts = {a[evidence][i] for i in av}
                    new_texts = {b[evidence][i] for i in bv}
                    differences.append({'id': key, 'direction': direction, 'query_index': query, 'historical_candidate_indices': av, 'recomputed_candidate_indices': bv, 'candidate_text_set_differs': old_texts != new_texts})
    for key in ids:
        alignment[key] = baseline[key]
    manifest = {'n_unchanged_pairs_reused': len(ids), 'n_fresh_entity_and_local_control_pairs': len(new) - len(ids), 'input_pairs_exactly_identical': True, 'cached_hashes_and_chunk_texts_identical': True, 'embedding_model_directory': str(model), 'historical_embedding_metrics_code_sha256': old_code_hash, 'recomputed_embedding_metrics_code_sha256': code_hash, 'current_metrics_code_sha256': current_code_hash, 'historical_nli_metrics_code_sha256': historical_nli_hash, 'historical_embedding_code_hash_matches': same_embedding_code, 'code_provenance_basis': 'Matching historical and recomputed embedding module hashes.' if same_embedding_code else 'Historical embedding module hash differs and its source snapshot is unavailable; historical NLI module hash matches the current/recomputed module. Reuse the valid historical NLI evidence plan on identical pairs, exact chunks/settings and empirically equivalent embeddings; no historical embedding AST equivalence is claimed.', 'embedding_score_comparison_atol': args.atol, 'embedding_score_comparison_rtol': args.rtol, 'embedding_metric_max_abs_differences': dict(zip(embedding_checks.metric, embedding_checks.max_abs_difference)), 'embedding_metrics_equivalent_within_tolerance': True, 'embedding_checkpoint_directory_matches': True, 'chunk_budget_and_retrieval_top_k_match': True, 'observed_budget_top_k': [list(v) for v in sorted(unchanged_settings)], 'candidate_index_arrays_differ_before_freeze': len(differences), 'pairs_with_different_candidate_arrays': len({x['id'] for x in differences}), 'candidate_text_sets_differ_before_freeze': sum((x['candidate_text_set_differs'] for x in differences)), 'candidate_differences': differences, 'fresh_alignment_snapshot': str(backup_path), 'embedding_cosine_scores_recomputed_and_not_replaced': True, 'nli_not_planned_or_executed_at_freeze': True, 'nli_scores_used_for_evidence_selection': False, 'reason': 'Hold the retrieval factor fixed on identical controls; near ties can reorder after floating-point recomputation.'}
    serialized = json.dumps(alignment, ensure_ascii=False, indent=2) + '\n'
    shutil.copyfile(path, backup_path)
    temporary = path.with_name(path.name + '.freeze.tmp')
    temporary.write_text(serialized)
    temporary.replace(path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest

def validate_scores(old, new, ids, atol, rtol):
    if old.id.duplicated().any() or new.id.duplicated().any():
        raise ValueError('Score table contains duplicate IDs')
    a, b = (old.set_index('id'), new.set_index('id'))
    if not set(ids).issubset(a.index) or not set(ids).issubset(b.index):
        raise ValueError('Scores do not cover all unchanged input pairs')
    prefixes = ('lexical_', 'local_lexical_', 'embedding_', 'local_embedding_', 'bert_', 'local_bert_', 'nli_', 'local_nli_')
    old_columns = {c for c in a if c.startswith(prefixes)} - COUNTERS
    new_columns = {c for c in b if c.startswith(prefixes)} - COUNTERS
    if old_columns != new_columns:
        raise ValueError(f'Semantic score schemas differ: old-only={old_columns - new_columns}, new-only={new_columns - old_columns}')
    rows = []
    for column in sorted(old_columns):
        av, bv = (a.loc[ids, column], b.loc[ids, column])
        if pd.api.types.is_numeric_dtype(av) and pd.api.types.is_numeric_dtype(bv):
            equal = np.isclose(av.to_numpy(float), bv.to_numpy(float), atol=atol, rtol=rtol, equal_nan=True)
            valid = av.notna() & bv.notna()
            maximum = float((av[valid].astype(float) - bv[valid].astype(float)).abs().max()) if valid.any() else None
        else:
            equal = ((av == bv) | av.isna() & bv.isna()).to_numpy()
            maximum = None
        if not equal.all():
            wrong = list(np.asarray(ids)[~equal])
            raise ValueError(f'Unrelated metric changed: {column}, {len(wrong)} rows, first={wrong[0]}, max_abs_difference={maximum}')
        rows.append({'metric': column, 'n_unchanged_pairs': len(ids), 'max_abs_difference': maximum, 'within_tolerance': True})
    return pd.DataFrame(rows)

def entities(pairs, scores):
    metadata = []
    for pair in pairs.values():
        if pair['dataset'] == 'controlled' and pair['perturbation'] == 'entity_substitution':
            p = pair.get('metadata', {})
            metadata.append({'id': pair['id'], 'source_id': pair['source_id'], 'source_group_id': pair.get('source_group_id', pair['source_id']), 'entity_type': p.get('entity_type', 'unknown'), 'replacement_key': p.get('replacement_dictionary_key', 'unknown'), 'source_entity': p.get('source_entity', ''), 'replacement_entity': p.get('replacement_entity', ''), 'surface_occurrences_replaced': p.get('surface_occurrences_replaced'), 'surname_occurrences_replaced': p.get('surname_occurrences_replaced', 0)})
    if not metadata:
        raise ValueError('No applied entity substitutions are available')
    frame = pd.DataFrame(metadata)
    if frame.source_id.duplicated().any():
        raise ValueError('Expected at most one entity substitution per source and policy')
    metric_columns = [c for c in METRICS if c in scores]
    required = set(METRICS) - {'local_nli_bidirectional_mean', 'nli_bidirectional_min'}
    if not required.issubset(metric_columns):
        raise ValueError(f'Missing requested entity metrics: {sorted(required - set(metric_columns))}')
    frame = frame.merge(scores[['id', *metric_columns]], on='id', how='left', validate='one_to_one')
    frame['nli_bidirectional_min'] = np.minimum(frame.nli_forward_min, frame.nli_reverse_min)
    frame['local_nli_bidirectional_mean'] = np.minimum(frame.local_nli_forward_mean, frame.local_nli_reverse_mean)
    return frame

def summaries(frame, policy, scope):
    rows = []
    for metric in METRICS:
        values = frame.dropna(subset=[metric]).groupby('source_group_id')[metric].mean()
        if len(values):
            rows.append({'scope': scope, 'policy': policy, 'metric': metric, 'n_pairs': int(frame[metric].notna().sum()), 'n_sources': len(values), 'mean': values.mean(), 'median': values.median(), 'q10': values.quantile(0.1), 'q90': values.quantile(0.9)})
    return rows

def paired_changes(a, b, bootstrap, seed):
    paired = a.merge(b, on='source_id', suffixes=('_v1', '_clean'), validate='one_to_one')
    if not (paired.source_group_id_v1 == paired.source_group_id_clean).all():
        raise ValueError('Source group changed in entity cohort')
    rng = np.random.default_rng(seed)
    rows = []
    for metric in METRICS:
        delta = paired[metric + '_clean'] - paired[metric + '_v1']
        values = delta.groupby(paired.source_group_id_v1).mean().dropna().to_numpy()
        if len(values):
            ci = np.quantile(rng.choice(values, size=(bootstrap, len(values)), replace=True).mean(axis=1), [0.025, 0.975])
            rows.append({'metric': metric, 'n_pairs': int(delta.notna().sum()), 'n_sources': len(values), 'mean_clean_minus_v1': values.mean(), 'ci_low': ci[0], 'ci_high': ci[1]})
    return (paired, pd.DataFrame(rows))

def table(frame, columns=None):
    if columns:
        frame = frame[columns]

    def cell(value):
        return f'{value:.6f}' if isinstance(value, (float, np.floating)) else str(value)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |', *['| ' + ' | '.join((cell(x) for x in row)) + ' |' for row in frame.itertuples(index=False, name=None)]])

def _argument_error(message):
    raise ValueError(message)

def run(args):
    if args.bootstrap < 1 or min(args.atol, args.rtol) < 0:
        _argument_error('bootstrap must be positive and tolerances nonnegative')
    old = read_pairs(args.baseline_data / 'pairs.jsonl')
    new = read_pairs(args.clean_data / 'pairs.jsonl')
    ids = unchanged_pairs(old, new)
    if args.freeze_unchanged_alignment:
        freeze_unchanged_alignment(args, old, new, ids)
        return
    old_scores = pd.read_csv(args.baseline_output / 'scores.csv')
    new_scores = pd.read_csv(args.clean_output / 'scores.csv')
    if set(old_scores.id) != set(old) or set(new_scores.id) != set(new):
        raise ValueError("Score rows must exactly cover each run's pair file")
    invariant = validate_scores(old_scores, new_scores, ids, args.atol, args.rtol)
    a, b = (entities(old, old_scores), entities(new, new_scores))
    common = sorted(set(a.source_id) & set(b.source_id))
    if not common:
        raise ValueError('No source supports both entity policies')
    aa, bb = (a[a.source_id.isin(common)], b[b.source_id.isin(common)])
    rows = summaries(a, 'v1', 'all_applied') + summaries(b, 'clean', 'all_applied')
    rows += summaries(aa, 'v1', 'common_sources') + summaries(bb, 'clean', 'common_sources')
    summary = pd.DataFrame(rows)
    paired, deltas = paired_changes(aa, bb, args.bootstrap, args.seed)
    distribution = []
    for policy, frame in [('v1', a), ('clean', b)]:
        for (kind, key), part in frame.groupby(['entity_type', 'replacement_key']):
            distribution.append({'policy': policy, 'entity_type': kind, 'replacement_key': key, 'n_pairs': len(part)})
    coverage = []
    skip_details = []
    for policy, directory, pairs, frame in [('v1', args.baseline_data, old, a), ('clean', args.clean_data, new, b)]:
        path = directory / 'skipped.jsonl'
        skipped = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
        source_ids = {p['source_id'] for p in pairs.values() if p['dataset'] == 'controlled' and p['perturbation'] == 'identity'}
        eligible = set(frame.source_id)
        if not eligible.issubset(source_ids):
            raise ValueError('Entity policy selected sources outside its identity cohort')
        entity_skips = [p for p in skipped if p['perturbation'] == 'entity_substitution']
        skip_sources = {p['source_id'] for p in entity_skips}
        if skip_sources != source_ids - eligible:
            raise ValueError('Entity skipped rows do not account for all inapplicable sources')
        coverage.append({'policy': policy, 'n_cohort_sources': len(source_ids), 'n_applied': len(frame), 'n_skipped': len(entity_skips), 'coverage': len(frame) / len(source_ids), 'n_sources_common_to_both': len(common)})
        for reason, count in Counter((p.get('metadata', {}).get('reason', 'unspecified') for p in entity_skips)).items():
            skip_details.append({'policy': policy, 'reason': reason, 'count': count})
    if {p['source_id'] for p in old.values() if p['dataset'] == 'controlled'} != {p['source_id'] for p in new.values() if p['dataset'] == 'controlled'}:
        raise ValueError('The source cohort changed; only entity eligibility may change')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    invariant.to_csv(output / 'nonentity_metric_invariance.csv', index=False)
    summary.to_csv(output / 'entity_policy_summary.csv', index=False)
    deltas.to_csv(output / 'entity_policy_paired_deltas.csv', index=False)
    paired.to_csv(output / 'entity_policy_paired_scores.csv', index=False)
    pd.DataFrame(distribution).to_csv(output / 'entity_policy_type_distribution.csv', index=False)
    pd.DataFrame(coverage).to_csv(output / 'entity_policy_coverage.csv', index=False)
    pd.DataFrame(skip_details, columns=['policy', 'reason', 'count']).to_csv(output / 'entity_policy_skip_reasons.csv', index=False)
    report = ['# Confronto fra sostituzioni di entità v1 e clean', '', 'Le sole condizioni autorizzate a cambiare sono entity_substitution e il relativo controllo identità locale. La coorte e tutte le altre coppie, incluse fixture e OpAI, sono confrontate per ID e contenuto integrale.', '', f'**Invarianza verificata:** {len(ids)} coppie non-entity esattamente identiche; {len(invariant)} colonne di metriche entro atol={args.atol:g}, rtol={args.rtol:g}. I contatori cumulativi di lavoro non sono metriche semantiche e sono esclusi.', '', '## Copertura sulla stessa coorte', '', table(pd.DataFrame(coverage)), '', 'I casi non applicabili della politica clean sono saltati esplicitamente, senza rimpiazzare le fonti. Il confronto principale seguente usa solo le fonti applicabili a entrambe le politiche; le medie all_applied nel CSV sono descrittive e possono riflettere selezione diversa.', '', '## Metriche sulle stesse fonti', '']
    reuse_path = args.clean_output / 'unchanged_alignment_reuse.json'
    if reuse_path.exists():
        reuse = json.loads(reuse_path.read_text())
        report[-2:-2] = ['## Retrieval mantenuto fisso sui controlli', '', f"Prima di pianificare l'NLI, {reuse['n_unchanged_pairs_reused']} coppie integralmente identiche hanno riusato il retrieval storico, dopo verifica di checkpoint, provenienza del codice, equivalenza numerica degli embedding, hash degli input, chunk, budget e top-k. Coseno e BERTScore sono stati ricalcolati; nessun punteggio NLI è stato usato per scegliere le evidenze. Le nuove sostituzioni clean e i loro controlli locali mantengono retrieval appena calcolato.", '', f"Prima del congelamento differivano {reuse['candidate_index_arrays_differ_before_freeze']} array candidati in {reuse['pairs_with_different_candidate_arrays']} coppie; I testi delle evidenze differivano in {reuse['candidate_text_sets_differ_before_freeze']} casi. Questo evita di cambiare un fattore estraneo alla policy a causa di pareggi numerici. Dettagli in unchanged_alignment_reuse.json; allineamento ricalcolato conservato separatamente.", '', 'Gli hash storici e attuali del modulo embedding coincidono.' if reuse['historical_embedding_code_hash_matches'] else "Il vecchio hash del modulo embedding è diverso e quel sorgente non è archiviato: non si sostiene un'equivalenza AST del codice storico. Il modulo usato dall'NLI storico coincide con quello attuale; si riusa il suo piano di evidenze su input e chunk identici, con coseni ricalcolati equivalenti entro la tolleranza dichiarata.", '']
    display = summary[summary.scope == 'common_sources'].pivot(index='metric', columns='policy', values='mean').reset_index()
    display = display.merge(deltas, on='metric', validate='one_to_one')
    report += [table(display, ['metric', 'v1', 'clean', 'mean_clean_minus_v1', 'ci_low', 'ci_high', 'n_sources']), '', 'Gli IC sono bootstrap appaiati per gruppo sorgente. Un calo del supporto non dimostra da solo che la perturbazione sia semanticamente valida: nessun punteggio NLI viene usato come gold label. Le categorie di entità e la frase bersaglio possono cambiare fra politiche anche sulla stessa fonte.', '', 'Il minimo bidirezionale è min(nli_forward_min, nli_reverse_min). La probabilità locale principale è local_nli_forward_mean; sono riportati anche reverse e minimo delle due direzioni. Coseno e BERTScore sono distinti fra documento e frase alterata. BERTScore qui usa lo stesso protocollo non riscalato del run storico, per un confronto omogeneo.', '', '## Distribuzione delle categorie', '', table(pd.DataFrame(distribution)), '', 'La politica clean limita le categorie e può modificare più menzioni coreferenti. Non certifica la coreferenza o la verità dei fatti; le corruzioni restano condizioni sperimentali.', '', '## Riproduzione', '', '```bash', 'python experiments/02_semantic_validation.py --mode reproduce --analysis entity_clean', '```', '']
    (output / 'entity_policy_comparison.md').write_text('\n'.join(report).rstrip() + '\n')
    manifest = {'unchanged_nonentity_input_pairs': len(ids), 'unchanged_metric_columns': len(invariant), 'semantic_metrics_unchanged_within_tolerance': True, 'atol': args.atol, 'rtol': args.rtol, 'runtime_counters_excluded': sorted(COUNTERS), 'coverage': coverage, 'comparison_uses_common_sources': True, 'bootstrap': args.bootstrap, 'labels_are_not_gold_semantic_annotations': True}
    (output / 'entity_policy_comparison_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'baseline_data': ROOT / 'data/processed/semantic_baseline', 'baseline_output': ROOT / 'results/published/03_semantics/baseline', 'clean_data': ROOT / 'data/processed/semantic_clean', 'clean_output': ROOT / 'results/published/03_semantics/entity_clean', 'output': ROOT / 'results/new_runs/03_semantics/entity_clean_comparison', 'atol': 1e-06, 'rtol': 1e-06, 'bootstrap': 1000, 'seed': 42, 'freeze_unchanged_alignment': False}
    values.update(overrides)
    return SimpleNamespace(**values)
