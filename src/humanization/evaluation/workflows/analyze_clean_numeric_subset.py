"""Describe the existing DATE/MONEY/PERCENT clean pairs; no model inference."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from humanization.paths import project_root
ROOT = project_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from humanization.evaluation.workflows.compare_entity_policy import METRICS, entities, paired_changes, read_pairs, summaries, table
FAMILY = {'DATE', 'MONEY', 'PERCENT'}
LABELS = {'local_nli_forward_mean': 'Entailment locale, originale → modificato', 'nli_bidirectional_min': 'Supporto minimo documento, bidirezionale', 'embedding_cosine': 'Coseno documento', 'local_embedding_cosine': 'Coseno frase', 'bert_chunk_f1': 'BERTScore F1 documento', 'local_bert_f1': 'BERTScore F1 frase'}

def score(row, metric):
    if metric == 'nli_bidirectional_min':
        return min(row['nli_forward_min'], row['nli_reverse_min'])
    return row[metric]

def _argument_error(message):
    raise ValueError(message)

def run(args):
    if args.bootstrap < 1:
        _argument_error('--bootstrap must be positive')
    pairs = read_pairs(args.data_dir / 'pairs.jsonl')
    old_pairs = read_pairs(args.baseline_data / 'pairs.jsonl')
    new_scores = pd.read_csv(args.results / 'scores.csv')
    old_scores = pd.read_csv(args.baseline_results / 'scores.csv')
    all_clean = entities(pairs, new_scores)
    selected = all_clean[all_clean.entity_type.isin(FAMILY)].copy()
    if len(all_clean) != 145 or len(selected) != 34:
        raise ValueError('This frozen sensitivity analysis expects 34 numeric pairs among 145 clean entities')
    old = entities(old_pairs, old_scores)
    old = old[old.source_id.isin(selected.source_id)].copy()
    if len(old) != len(selected):
        raise ValueError('The baseline does not cover all selected sources')
    paired, deltas = paired_changes(old, selected, args.bootstrap, args.seed)
    summary = pd.DataFrame(summaries(old, 'v1', 'numeric_clean_sources') + summaries(selected, 'clean', 'numeric_clean_sources'))
    display = summary.pivot(index='metric', columns='policy', values='mean').reset_index().merge(deltas, on='metric')
    category = []
    for kind, part in selected.groupby('entity_type'):
        for row in summaries(part, 'clean', kind):
            category.append({'entity_type': kind, **row})
    category = pd.DataFrame(category)
    identities = {p['source_id']: p for p in pairs.values() if p['dataset'] == 'controlled' and p['perturbation'] == 'identity'}
    local = {p['metadata']['control_for']: p for p in pairs.values() if p['dataset'] == 'local_control'}
    indexed = new_scores.set_index('id')
    references, cases = ([], [])
    for row in selected.to_dict('records'):
        p = pairs[row['id']]
        document, sentence = (identities[row['source_id']], local[row['id']])
        if document['original'] != p['original'] or document['modified'] != p['original'] or sentence['original'] != p['target_original'] or (sentence['modified'] != p['target_original']):
            raise ValueError('Identity control does not match the selected document/target')
        for metric in LABELS:
            control = sentence if metric.startswith('local_') else document
            references.append({'source_id': row['source_id'], 'source_group_id': row['source_group_id'], 'metric': metric, 'identity': score(indexed.loc[control['id']], metric), 'altered': row[metric]})
        cases.append({**row, 'target_original': p['target_original'], 'target_modified': p['target_modified']})
    references = pd.DataFrame(references)
    reference_summary = []
    for metric, part in references.groupby('metric', sort=False):
        values = part.groupby('source_group_id')[['identity', 'altered']].mean()
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f'Missing/nonfinite identity or altered score: {metric}')
        labels = np.r_[np.ones(len(values)), np.zeros(len(values))]
        quality = np.r_[values.identity, values.altered]
        reference_summary.append({'metric': metric, 'n_sources': len(values), 'identity_mean': values.identity.mean(), 'altered_mean': values.altered.mean(), 'identity_min': values.identity.min(), 'identity_max': values.identity.max(), 'altered_min': values.altered.min(), 'altered_max': values.altered.max(), 'identity_vs_altered_auc': roc_auc_score(labels, quality)})
    reference_summary = pd.DataFrame(reference_summary)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in [('summary', summary), ('paired_deltas', deltas), ('paired_scores', paired), ('by_category', category), ('identity_references', references), ('identity_summary', reference_summary), ('cases', pd.DataFrame(cases))]:
        frame.to_csv(args.output / (name + '.csv'), index=False)
    primary = display[display.metric.isin(LABELS)].copy()
    primary['misura'] = primary.metric.map(LABELS)
    identity_display = reference_summary.copy()
    identity_display['misura'] = identity_display.metric.map(LABELS)
    counts = dict(Counter(selected.entity_type))
    same_type = int((paired.entity_type_v1 == paired.entity_type_clean).sum())
    by_type = category.pivot(index=['entity_type', 'n_pairs'], columns='metric', values='mean').reset_index()
    local_ref = reference_summary.set_index('metric').loc['local_nli_forward_mean']
    report = ['# Controllo di robustezza: date, importi e percentuali', '', '**L’NLI continua ad assegnare basso supporto alle alterazioni anche in questa famiglia più omogenea.** Il risultato è descrittivo su corruzioni controllate; non certifica ogni perturbazione né la superiorità sulle parafrasi corrette.', '', '## Selezione e metodo', '', f"Selezione fissata dai metadati clean, senza usare gli score: **34 coppie e {selected.source_group_id.nunique()} fonti** fra le 145 entity applicate; {counts['DATE']} DATE, {counts['MONEY']} MONEY, {counts['PERCENT']} PERCENT. CARDINAL, ORDINAL, PERSON e GPE sono esclusi. Nessuna coppia delle 34 è stata scartata dopo la lettura o la valutazione.", '', 'Sono riusati esclusivamente JSONL e CSV già disponibili: **zero nuove inferenze e zero modifiche ai dati**. Checkpoint e metriche sono quelli del run clean: DeBERTa-v3 base float32, MiniLM e BERTScore RoBERTa-base con il protocollo non riscalato del confronto storico. Il minimo bidirezionale è il minore fra minimo forward e reverse per documento.', '', f'La v1 è confrontata sulle stesse 34 fonti. Solo **{same_type}/34** hanno lo stesso tipo di entità nelle due politiche: la selezione DATE/MONEY/PERCENT riguarda la clean. Il confronto non isola un cambiamento dello stesso numero, della stessa frase o della stessa entità. IC 95%: bootstrap appaiato per fonte, {args.bootstrap} ricampionamenti, seed {args.seed}; nessuna nuova soglia è calibrata.', '', '## Stessa analisi del campione di 145, sulle 34 fonti', '', table(primary, ['misura', 'v1', 'clean', 'mean_clean_minus_v1', 'ci_low', 'ci_high']), '', 'Il minimo NLI resta basso in entrambe le politiche. Le differenze fra v1 e clean non vanno interpretate come un miglioramento dell’accuratezza semantica: cambiano target e composizione delle sostituzioni. Gli IC delle differenze di entailment locale forward e minimo documentale includono zero: non emerge un vantaggio netto della clean rispetto alla v1 su queste 34 fonti. Le dieci misure complete, incluse le direzioni inverse, sono nei CSV summary e paired_deltas.', '', '## Riferimento positivo: testo e frase identici', '', 'Ogni alterazione è accoppiata al controllo identità dell’intero originale per le misure documentali e al controllo identità della stessa frase bersaglio per le misure locali. Le identità sono state inferite realmente, non impostate a 1.', '', table(identity_display, ['misura', 'identity_mean', 'altered_mean', 'altered_min', 'altered_max', 'identity_vs_altered_auc']), '', f'Entailment locale delle alterazioni: media **{local_ref.altered_mean:.6f}**, intervallo osservato [{local_ref.altered_min:.6f}; {local_ref.altered_max:.6f}]. Sulle frasi identiche la media è **{local_ref.identity_mean:.6f}** e il minimo {local_ref.identity_min:.6f}: il calo non dipende solo dalla difficoltà intrinseca delle frasi.', '', 'Le identità sono controlli facili: anche coseno e BERTScore separano perfettamente queste identità dalle alterazioni. L’AUC contro identità, riportata soltanto come diagnostica secondaria, non dimostra un vantaggio discriminativo dell’NLI. Per quello servono riscritture corrette e abbastanza diverse: il confronto equo precedente resta il riferimento.', '', '## Risultato per famiglia', '', table(by_type, ['entity_type', 'n_pairs', 'local_nli_forward_mean', 'nli_bidirectional_min', 'embedding_cosine', 'local_embedding_cosine', 'bert_chunk_f1', 'local_bert_f1']), '', 'I tre gruppi confermano il basso entailment locale. Le numerosità 18/6/10 sono piccole e non giustificano una graduatoria di affidabilità fra date, importi e percentuali.', '', '## Direzionalità, lettura dei testi e limiti', '', 'Un caso percentuale trasforma **“more than 10 percent” → “37%”**: il modificato è più specifico e può implicare l’originale. L’entailment locale forward è 0,014795, il reverse 0,994922. Il reverse alto è semanticamente plausibile, non un fallimento automatico né prova di contraddizione. Le sostituzioni possono togliere qualificatori o cambiare valuta, oltre al valore numerico.', '', 'La lettura delle 34 frasi da parte dell’assistente rileva anche **1940’s → 1998’s**, con ambiguità di decennio/forma. Non sono 34 contraddizioni gold: questa famiglia evita molte ambiguità PERSON/GPE ma mantiene limiti di contesto. Molte modifiche sono ampie; tutti i sei importi arrivano a somme da 26/47/83 milioni. Il controllo non misura l’affidabilità su piccoli scarti aritmetici, errori numerici realistici o humanizer nuovi.', '', '**Conclusione per la tesi:** il basso supporto NLI e l’alta similarità persistono sul sottoinsieme DATE/MONEY/PERCENT; il risultato non è sostenuto soltanto dalle sostituzioni di persone o luoghi più problematiche. È un controllo di robustezza favorevole sulle perturbazioni considerate, con validità semantica e generalizzazione da verificare separatamente.', '', '## Riproduzione e file', '', '```bash', '.venv/bin/python src/humanization/evaluation/workflows/analyze_clean_numeric_subset.py', '```', '', '`cases.csv` contiene tutti i 34 ID, tipi, valori sostituiti, frasi prima/dopo e punteggi. `paired_scores.csv` permette il confronto v1/clean per fonte; `summary.csv`, `paired_deltas.csv` e `by_category.csv` riportano statistiche e IC. I riferimenti identità sono in `identity_references.csv` e `identity_summary.csv`.', '']
    (args.output / 'report.md').write_text('\n'.join(report).rstrip() + '\n', encoding='utf-8')
    print(f'34 pairs ({counts}), {len(paired)} matched sources; report: {args.output}/report.md')

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'data_dir': ROOT / 'data/semantic_clean', 'baseline_data': ROOT / 'data/semantic_full', 'results': ROOT / 'results/published/03_semantics/entity_clean', 'baseline_results': ROOT / 'results/published/03_semantics/baseline', 'output': ROOT / 'results/new_runs/03_semantics/entity_clean/numeric34', 'bootstrap': 1000, 'seed': 42}
    values.update(overrides)
    return SimpleNamespace(**values)
