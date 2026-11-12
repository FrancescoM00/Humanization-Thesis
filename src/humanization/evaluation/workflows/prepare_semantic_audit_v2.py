"""Prepara schede di annotazione indipendenti con gruppi di riferimento e diagnostica separati.

Le fonti di riferimento sono campionate senza usare gli score. La diagnostica
include più casi con NLI basso o disaccordo coseno/NLI: non è un test della
popolazione. Tutte le riscritture di una fonte selezionata restano unite e i
due gruppi sono disgiunti per fonte. Qui non si eseguono inferenze o annotazioni.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
PROTOCOL = 'semantic_audit_reference_diagnostic_v2'
FORM_COLUMNS = ['review_id', 'original', 'modified', 'input_type_prose_instructions_mixed', 'semantic_preserved_yes_partial_no', 'error_type', 'notes']

def rank(seed, namespace, value):
    return hashlib.sha256(f'{seed}:{namespace}:{value}'.encode()).hexdigest()

def read_pairs(path):
    """Preferisce l’input semantico effettivo negli ZIP storici con riscritture non aggiornate."""
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            candidates = [name for name in ('semantic/pairs.jsonl', 'rewriting_cuda0/rewrites.jsonl', 'rewriting/rewrites.jsonl') if name in archive.namelist()]
            if not candidates:
                raise ValueError(f'No recognized pairs/rewrites member in {path}')
            body = archive.read(candidates[0]).decode('utf-8')
    else:
        body = path.read_text(encoding='utf-8')
    return [json.loads(line) for line in body.splitlines() if line.strip()]

def read_scores(paths):
    scores = {}
    for path in paths:
        with Path(path).open(newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                if not row.get('id'):
                    raise ValueError(f'Score row has no id: {path}')
                previous = scores.setdefault(row['id'], {})
                for key, value in row.items():
                    if value == '' or key not in previous or previous[key] == '':
                        previous.setdefault(key, value)
                        if previous[key] == '':
                            previous[key] = value
                    elif previous[key] != value:
                        try:
                            equal = math.isclose(float(previous[key]), float(value), rel_tol=0, abs_tol=1e-12)
                        except ValueError:
                            equal = False
                        if not equal:
                            raise ValueError(f"Conflicting scores for {row['id']}: {key}")
    return scores

def finite_score(row, names, *, lower, upper):
    for name in names:
        if row.get(name, '') != '':
            value = float(row[name])
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Invalid {name} for {row.get('id')}: {value}")
            return value
    raise ValueError(f"Missing {names} for {row.get('id')}")

def candidates_from_inputs(pair_rows, score_rows):
    candidates, excluded = ({}, Counter())
    for pair in pair_rows:
        system = pair.get('humanizer', '')
        if not system:
            raise ValueError('Expected explicit humanizer in pair input')
        if system == 'identity' or system.startswith('null_'):
            excluded['identity_or_null'] += 1
            continue
        if pair.get('status', 'ok') != 'ok':
            excluded['generation_not_ok'] += 1
            continue
        required = ['id', 'source_id', 'source_group_id', 'domain', 'generator', 'humanizer', 'original', 'modified']
        if any((not isinstance(pair.get(k), str) or not pair[k].strip() for k in required)):
            raise ValueError('Pairs require nonempty IDs, source metadata and texts')
        if pair['id'] not in score_rows:
            raise ValueError(f"Missing scores for {pair['id']}; no silent exclusion")
        score = score_rows[pair['id']]
        for key in ['source_id', 'source_group_id', 'domain', 'generator', 'humanizer']:
            if score.get(key, '') not in ('', pair[key]):
                raise ValueError(f"Score metadata differ for {pair['id']}: {key}")
        nli_names = ['nli_worst_direction_min', 'nli_bidirectional_min']
        if not any((score.get(k, '') != '' for k in nli_names)):
            nli = min((finite_score(score, [f'nli_{d}_min'], lower=0, upper=1) for d in ['forward', 'reverse']))
        else:
            nli = finite_score(score, nli_names, lower=0, upper=1)
        cosine = finite_score(score, ['embedding_cosine'], lower=-1, upper=1)
        row = {key: pair[key] for key in required}
        row.update(nli_min=nli, embedding_cosine=cosine)
        if row['id'] in candidates and candidates[row['id']] != row:
            raise ValueError(f"Conflicting duplicate annotation pair: {row['id']}")
        candidates[row['id']] = row
    if not candidates:
        raise ValueError('No eligible successful humanizer rewrites')
    return (sorted(candidates.values(), key=lambda r: r['id']), dict(excluded))

def select_candidates(candidates, *, reference_per_cell=1, diagnostic_size=100, seed=42, low_nli=0.1, high_cosine=0.9):
    if reference_per_cell < 1 or diagnostic_size < 0:
        raise ValueError('Positive reference_per_cell and nonnegative diagnostic_size required')
    if not 0 <= low_nli <= 1 or not -1 <= high_cosine <= 1:
        raise ValueError('Invalid diagnostic thresholds')
    groups, cells = (defaultdict(list), defaultdict(list))
    for row in candidates:
        groups[row['source_group_id']].append(row)
    for group, rows in groups.items():
        strata = {(r['domain'], r['generator']) for r in rows}
        if len(strata) != 1:
            raise ValueError(f'Source group crosses domain/generator strata: {group}')
        if len({r['humanizer'] for r in rows}) != len(rows):
            raise ValueError(f'More than one rewrite per humanizer/source group: {group}')
        cells[next(iter(strata))].append(group)
    reference, probabilities, cell_summary = (set(), {}, [])
    for cell, members in sorted(cells.items()):
        ordered = sorted(members, key=lambda g: rank(seed, 'reference-source', g))
        take = min(reference_per_cell, len(ordered))
        reference.update(ordered[:take])
        probabilities.update({g: take / len(ordered) for g in ordered[:take]})
        cell_summary.append(dict(domain=cell[0], generator=cell[1], available_sources=len(ordered), reference_sources=take, inclusion_probability=take / len(ordered)))

    def own_stratum(row):
        if row['nli_min'] <= low_nli:
            return 'high_cosine_low_nli' if row['embedding_cosine'] >= high_cosine else 'low_nli'
        return 'other'
    source_strata = {}
    buckets = defaultdict(list)
    for group, rows in groups.items():
        if group in reference:
            continue
        flags = {own_stratum(row) for row in rows}
        stratum = 'high_cosine_low_nli' if 'high_cosine_low_nli' in flags else 'low_nli' if 'low_nli' in flags else 'other'
        source_strata[group] = stratum
        buckets[stratum].append(group)
    for stratum, members in buckets.items():
        by_cell = defaultdict(list)
        for group in members:
            first = groups[group][0]
            by_cell[first['domain'], first['generator']].append(group)
        for cell in by_cell:
            by_cell[cell].sort(key=lambda g: (min((r['nli_min'] for r in groups[g])) if stratum != 'other' else 0, -max((r['embedding_cosine'] for r in groups[g])) if stratum != 'other' else 0, rank(seed, 'diagnostic-source', g)))
        order = sorted(by_cell, key=lambda c: rank(seed, 'diagnostic-cell', repr(c)))
        interleaved = []
        while any(by_cell.values()):
            for cell in order:
                if by_cell[cell]:
                    interleaved.append(by_cell[cell].pop(0))
        buckets[stratum] = deque(interleaved)
    schedule = ['high_cosine_low_nli'] * 3 + ['low_nli'] * 2 + ['other']
    diagnostic, count = (set(), 0)
    while any(buckets.values()) and count < diagnostic_size:
        progressed = False
        for stratum in schedule:
            queue = buckets[stratum]
            while queue:
                group = queue.popleft()
                cost = len(groups[group])
                if cost <= diagnostic_size - count:
                    diagnostic.add(group)
                    count += cost
                    progressed = True
                    break
            if count == diagnostic_size:
                break
        if not progressed:
            break
    selected = []
    for row in candidates:
        group = row['source_group_id']
        if group not in reference | diagnostic:
            continue
        arm = 'reference' if group in reference else 'diagnostic'
        selected.append(dict(row, selection_arm=arm, diagnostic_source_stratum=source_strata.get(group, ''), diagnostic_pair_stratum=own_stratum(row) if arm == 'diagnostic' else '', reference_source_inclusion_probability=probabilities.get(group, ''), reference_source_weight=1 / probabilities[group] if group in reference else ''))
    assert reference.isdisjoint(diagnostic)
    summary = dict(available_pairs=len(candidates), available_sources=len(groups), reference_pairs=sum((len(groups[g]) for g in reference)), reference_sources=len(reference), requested_diagnostic_pairs=diagnostic_size, available_diagnostic_pairs=sum((len(rows) for g, rows in groups.items() if g not in reference)), selected_diagnostic_pairs=count, diagnostic_sources=len(diagnostic), diagnostic_target_met=count == diagnostic_size, total_selected_unique_pairs=len(selected), total_selected_unique_sources=len(reference | diagnostic), selected_is_full_census=len(selected) == len(candidates), cell_summary=cell_summary, diagnostic_pair_strata=dict(Counter((r['diagnostic_pair_stratum'] for r in selected if r['selection_arm'] == 'diagnostic'))))
    if not summary['diagnostic_target_met']:
        summary['warning'] = 'Diagnostic target not reached: finite remaining capacity and complete-source grouping; no duplicated or fabricated pairs.'
    return (selected, summary)
INSTRUCTIONS = "# Revisione semantica v2\n\nLe due schede contengono le stesse coppie, con ordine diverso e giudizi vuoti.\nConsegnare una sola copia a ciascun valutatore; i lettori lavorano indipendentemente,\nsenza consultare chiave, score, report, commenti dell'assistente o giudizi altrui.\nDue annotatori della stessa coppia non creano due fonti indipendenti.\n\nIn `input_type_prose_instructions_mixed` usare prose/instructions/mixed.\nIn `semantic_preserved_yes_partial_no` usare yes (informazioni e forza conservate),\npartial (quadro principale conservato ma informazioni rilevanti alterate, omesse\no aggiunte), no (contenuto ampiamente alterato o cambio di compito). Fusione e\ndivisione delle frasi sono ammesse se il contenuto rimane. Non giudicare soltanto\nla lunghezza; rimuovere ripetizioni può conservare l'informazione unica.\n\nIn `error_type` indicare omission, unsupported_addition, number_value, number_bound,\nentity, negation, hedging_or_claim_strength, task_shift o other, anche più d'uno.\nNelle note citare il punto preciso e l'eventuale ambiguità. Si valuta fedeltà al\npassaggio, non verità esterna o naturalezza. Non copiare il verdetto da NLI.\n\nConservare ID e testi esatti. Le etichette vuote non sono risultati. La chiave\ntenuta separatamente identifica il campione reference, scelto senza score, e\nquello diagnostic, arricchito per casi problematici: non usarli come un unico\ntest casuale. La chiave permette anche di riconoscere coppie già annotate nelle\nprecedenti schede e di evitare il doppio conteggio di giudizi dello stesso lettore.\n"

def csv_text(rows, columns):
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()

def prepare_audit(pair_paths, score_paths, output, *, reference_per_cell=1, diagnostic_size=100, seed=42, low_nli=0.1, high_cosine=0.9, require_full_diagnostic=False):
    scores = read_scores(score_paths)
    pair_rows = [row for path in pair_paths for row in read_pairs(path)]
    candidates, excluded = candidates_from_inputs(pair_rows, scores)
    selected, summary = select_candidates(candidates, reference_per_cell=reference_per_cell, diagnostic_size=diagnostic_size, seed=seed, low_nli=low_nli, high_cosine=high_cosine)
    if require_full_diagnostic and (not summary['diagnostic_target_met']):
        raise ValueError(summary['warning'])
    selected.sort(key=lambda r: rank(seed, 'opaque-review-id', r['id']))
    forms, key = ([], [])
    for index, row in enumerate(selected, 1):
        review_id = f'V2{index:04d}'
        form = {name: '' for name in FORM_COLUMNS}
        form.update(review_id=review_id, original=row['original'], modified=row['modified'])
        forms.append(form)
        key.append(dict(review_id=review_id, **{k: v for k, v in row.items() if k not in {'original', 'modified'}}))
    files = {'ISTRUZIONI.md': INSTRUCTIONS, 'selection_key.csv': csv_text(key, list(key[0]))}
    for annotator in ['A', 'B']:
        ordered = sorted(forms, key=lambda r: rank(seed, f'annotator-{annotator}-order', r['review_id']))
        files[f'semantic_audit_annotator_{annotator}.csv'] = csv_text(ordered, FORM_COLUMNS)
    manifest = dict(protocol=PROTOCOL, seed=seed, parameters=dict(reference_per_cell=reference_per_cell, diagnostic_size=diagnostic_size, low_nli=low_nli, high_cosine=high_cosine), input_files=[dict(path=str(Path(p)), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in [*pair_paths, *score_paths]], excluded_input_rows=excluded, **summary, reference_unit='source group within domain/generator; all successful humanizers retained', reference_selection='seeded hash of source identifiers, independent of metric scores', diagnostic_selection='score strata with weighted 3:2:1 schedule; complete source groups; cell interleaving', identity_and_null_controls_included=False, annotators=2, labels_present=False, gold_labels_created=False, diagnostic_population_accuracy_allowed=False, reference_probability_scope='successful scored humanizer outputs in the supplied frame', annotator_judgments_are_not_additional_independent_sources=True)
    files['sampling_manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    output = Path(output)
    for name, body in files.items():
        path = output / name
        if path.exists() and path.read_text() != body:
            raise ValueError(f'Refusing to overwrite a different frozen audit: {path}')
    output.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (output / name).write_text(body)
    with zipfile.ZipFile(output / 'annotation_forms.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ['semantic_audit_annotator_A.csv', 'semantic_audit_annotator_B.csv', 'ISTRUZIONI.md']:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, files[name].encode())
    return manifest

def _argument_error(message):
    raise ValueError(message)

def run(args):
    manifest = prepare_audit(args.input_pairs, args.scores, args.output, reference_per_cell=args.reference_per_cell, diagnostic_size=args.diagnostic_size, seed=args.seed, low_nli=args.low_nli, high_cosine=args.high_cosine, require_full_diagnostic=args.require_full_diagnostic)
    print(json.dumps({k: manifest[k] for k in ['reference_pairs', 'selected_diagnostic_pairs', 'requested_diagnostic_pairs', 'total_selected_unique_pairs', 'total_selected_unique_sources', 'diagnostic_target_met']}, indent=2))
    if 'warning' in manifest:
        print(manifest['warning'])

def default_arguments(**overrides):
    from types import SimpleNamespace
    values = {'input_pairs': None, 'scores': None, 'output': None, 'diagnostic_size': 100, 'reference_per_cell': 1, 'seed': 42, 'low_nli': 0.1, 'high_cosine': 0.9, 'require_full_diagnostic': False}
    values.update(overrides)
    return SimpleNamespace(**values)
