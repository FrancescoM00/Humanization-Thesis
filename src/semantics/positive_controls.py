"""Adjacent OpAI positive controls and controlled errors in the same edits.

Operation names specify intended meaning preservation, not independently
verified gold semantic labels. No editing API or new generated text is used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from .opai import (
    OFFICIAL_CODE_COMMIT, OFFICIAL_DATASET, OFFICIAL_REPOSITORY,
    _balanced_selection, _normalize,
)

POSITIVE_OPERATIONS = ((0, 1, 'polish'), (1, 2, 'paraphrase'), (2, 3, 'style'))


def _conservative_entity(entity):
    """Reduce obvious small-model NER span errors without semantic scoring."""
    text = entity.text.strip()
    if not text or '\n' in text or '\r' in text or len(text) > 100:
        return False
    if any(token.is_sent_start for token in entity[1:]):
        return False
    if re.search(r'[.!?;:<>\[\]{}]', text):
        return False
    words = text.split()
    if entity.label_ == 'PERSON':
        return (2 <= len(words) <= 5 and all(word.isalpha() for word in words)
                and not text.isupper() and all(word[0].isupper() for word in words))
    if entity.label_ in {'GPE', 'LOC'}:
        # Precision-oriented finite geography list: no NLI-based filtering.
        known = {'united states', 'united kingdom', 'england', 'scotland', 'ireland',
                 'wales', 'france', 'germany', 'italy', 'spain', 'portugal', 'greece',
                 'sweden', 'norway', 'denmark', 'finland', 'poland', 'ukraine', 'russia',
                 'china', 'japan', 'india', 'pakistan', 'bangladesh', 'canada', 'mexico',
                 'brazil', 'argentina', 'australia', 'new zealand', 'south africa',
                 'nigeria', 'egypt', 'kenya', 'iran', 'iraq', 'israel', 'palestine',
                 'turkey', 'syria', 'saudi arabia', 'south korea', 'north korea',
                 'indonesia', 'malaysia', 'singapore', 'thailand', 'vietnam', 'taiwan',
                 'hong kong', 'europe', 'asia', 'africa', 'north america', 'south america',
                 'london', 'paris', 'berlin', 'rome', 'tokyo', 'beijing', 'shanghai',
                 'new york', 'washington', 'los angeles', 'chicago', 'boston', 'moscow',
                 'mumbai', 'delhi', 'sydney', 'melbourne', 'toronto', 'montreal',
                 'california', 'texas', 'florida', 'ohio', 'pennsylvania', 'virginia'}
        return text.casefold().removeprefix('the ') in known
    if entity.label_ == 'ORG':
        if text in {'IBM', 'NASA', 'UN', 'WHO', 'IMF', 'NATO', 'FBI', 'CIA', 'OECD', 'UNESCO', 'EU', 'MIT'}:
            return True
        markers = {'university', 'institute', 'inc', 'corporation', 'bank', 'council',
                   'agency', 'association', 'company', 'group', 'party', 'committee',
                   'organization', 'organisation', 'ministry', 'union', 'foundation',
                   'department', 'court', 'commission'}
        return 2 <= len(words) <= 12 and bool({word.casefold() for word in words} & markers)
    if entity.label_ == 'DATE':
        return not any(token.tag_ == 'MD' for token in entity)
    return entity.label_ == 'CARDINAL'


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_local_opai_trajectories(opai_dir=Path('data/opai'), max_version=5):
    """Read the already verified local Parquet release, without network work.

    Return ``(trajectories, provenance)``. Trajectory keys are
    ``(document_id, domain, generator)`` and version keys are integers. The
    saved downloader provenance pins the 16 official shard checksums.
    """
    import pyarrow.parquet as pq

    opai_dir = Path(opai_dir)
    source = json.loads((opai_dir / 'records.provenance.json').read_text())
    if source['dataset'] != OFFICIAL_DATASET:
        raise ValueError('Local source provenance does not identify official OpAI-Bench')
    trajectories, rows, retained, duplicates = {}, 0, 0, 0
    columns = ['record_id', 'id', 'document_hash_id', 'domain', 'generator',
               'edit_operation', 'version', 'version_index', 'text', 'sentences', 'sentence_ids']
    source_files = []
    for entry in source['source_files']:
        path = opai_dir / 'official_parquet' / Path(entry['repository_path']).name
        if not path.is_file() or _sha256_file(path) != entry['sha256']:
            raise ValueError(f'Local official OpAI Parquet is absent or changed: {path}')
        source_files.append({**entry, 'local_file': str(path.resolve())})
        parquet = pq.ParquetFile(path)
        available_columns = [name for name in columns if name in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(batch_size=256, columns=available_columns):
            for raw in batch.to_pylist():
                rows += 1
                row = _normalize(raw)
                if row['version'] > max_version:
                    continue
                retained += 1
                key = (row['document_id'], row['domain'], row['generator'])
                versions = trajectories.setdefault(key, {})
                if row['version'] in versions:
                    if row['text'] != versions[row['version']]['text']:
                        raise ValueError(f'Ambiguous OpAI duplicate {key}/v{row["version"]}')
                    duplicates += 1
                    continue
                versions[row['version']] = row
    provenance = {
        'method': 'local_official_parquet_reprojection_no_new_download',
        'dataset': OFFICIAL_DATASET, 'dataset_revision': source['dataset_revision'],
        'split': source['split'], 'source_files': source_files,
        'source_rows': rows, 'rows_retained': retained, 'max_version': max_version,
        'exact_duplicate_rows_ignored': duplicates,
        'original_projection_sha256': source['sha256'],
    }
    return trajectories, provenance


def assign_source_splits(pairs, seed=42):
    """Assign half the source groups per domain to calibration, then test.

    Every editor, version and perturbation of a human source keeps one split.
    These experiment splits are distinct from OpAI's original dataset split.
    """
    source_domains = {}
    for pair in pairs:
        group = pair.get('source_group_id') or pair['source_id']
        previous = source_domains.setdefault(group, pair['domain'])
        if previous != pair['domain']:
            raise ValueError('A source group appears in multiple domains')
    strata = defaultdict(list)
    for group, domain in source_domains.items():
        strata[domain].append(group)
    splits = {}
    for domain in sorted(strata):
        groups = sorted(strata[domain])
        random.Random(f'{seed}|{domain}|source-split').shuffle(groups)
        n_calibration = len(groups) // 2
        for index, group in enumerate(groups):
            splits[group] = 'calibration' if index < n_calibration else 'test'
    for pair in pairs:
        group = pair.get('source_group_id') or pair['source_id']
        pair['source_group_id'] = group
        pair['evaluation_split'] = splits[group]
        pair.setdefault('metadata', {}).update(source_group_id=group, evaluation_split=splits[group])
    return splits


def select_positive_controls(trajectories, n=100, seed=42, require_through_version=5):
    """Select common trajectories, balanced across domain×editor, then all edits.

    Sorted candidates and seeded round-robin match the first n of the style
    study's larger selection when that study uses the same seed and loader.
    Selection is independent of semantic model scores and edit applicability.
    """
    required = set(range(require_through_version + 1))
    candidates = [dict(key=key, domain=key[1], generator=key[2])
                  for key, versions in sorted(trajectories.items()) if required.issubset(versions)]
    if len(candidates) < n:
        raise ValueError(f'Only {len(candidates)} complete trajectories, requested {n}')
    chosen = _balanced_selection(candidates, n, random.Random(seed))
    pairs = []
    for candidate in chosen:
        key = candidate['key']
        versions = trajectories[key]
        for before_version, after_version, operation in POSITIVE_OPERATIONS:
            before, after = versions[before_version], versions[after_version]
            if after['operation'] not in {operation, 'style rewrite' if operation == 'style' else operation}:
                raise ValueError(f'OpAI operation/version mismatch: v{after_version} {after["operation"]!r}')
            identity = f'{key[0]}|{key[1]}|{key[2]}|{before_version}|{after_version}'
            pair_id = 'opai:' + hashlib.sha256(identity.encode()).hexdigest()[:24]
            original, modified = before['text'], after['text']
            pairs.append({
                'id': pair_id, 'pair_id': pair_id, 'source_id': key[0],
                'source_group_id': 'opai:' + key[0], 'domain': key[1], 'generator': key[2],
                'dataset': 'opai_positive', 'perturbation': 'opai_' + operation,
                'status': 'applied', 'original': original, 'modified': modified,
                'target_original': None, 'target_modified': None,
                'metadata': {
                    'dataset': OFFICIAL_DATASET, 'comparison': 'adjacent_operation',
                    'from_version': f'v{before_version}', 'to_version': f'v{after_version}',
                    'operation': operation, 'source_record_id': before['record_id'],
                    'target_record_id': after['record_id'], 'text_changed': original != modified,
                    'length_ratio': len(modified.split()) / len(original.split()),
                    'expected_semantic_class': 'preserved', 'evaluation_label': 0,
                    'semantic_label_verified': False, 'positive_control_is_presumed_not_gold': True,
                    'positive_control_basis': 'Official meaning-preserving operation instruction; not manual annotation',
                    'aligned_changed_sentences': [],
                },
            })
    assign_source_splits(pairs, seed)
    manifest = {
        'official_repository': OFFICIAL_REPOSITORY, 'official_code_commit': OFFICIAL_CODE_COMMIT,
        'official_dataset': OFFICIAL_DATASET, 'seed': seed,
        'n_available_complete_trajectories': len(candidates), 'n_selected_trajectories': len(chosen),
        'n_selected_source_documents': len({p['source_id'] for p in pairs}),
        'n_pairs': len(pairs), 'counts_by_operation': dict(Counter(p['perturbation'] for p in pairs)),
        'selected_keys': [list(candidate['key']) for candidate in chosen],
        'trajectory_stratum_counts': dict(Counter(f'{c["domain"]}|{c["generator"]}' for c in chosen)),
        'evaluation_split_pair_counts': dict(Counter(p['evaluation_split'] for p in pairs)),
        'evaluation_split_source_counts': dict(Counter({p['source_group_id']: p['evaluation_split'] for p in pairs}.values())),
        'labels_are_operation_based_proxies_not_gold': True,
        'selection_is_independent_of_metric_scores': True,
        'caveats': ['Polish, Paraphrase and Style Rewrite are presumed preserved controls, not independently annotated truth',
                    'All versions are cumulative; adjacent pairs isolate one editing stage',
                    'All editors/versions of a source share one calibration/test split',
                    'Identity edits remain in primary analysis and are counted'],
        'unchanged_by_operation': dict(Counter(p['perturbation'] for p in pairs if p['original'] == p['modified'])),
    }
    return pairs, manifest


def make_matched_negative_controls(positives, seed=42, nlp=None, n_per_subtle=50):
    """Corrupt each actual positive output while retaining its original anchor.

    A negative compares before-edit text with the *same edited output* plus a
    controlled error. Its paired positive compares that same before-edit text
    with the uncorrupted edited output. Local targets isolate the incremental
    corruption on the post-edit output, not an invented alignment to the input.
    """
    from .perturbations import load_perturbation_nlp, _negation, _entity_substitution, _rng
    from .subtle_perturbations import TRANSFORMS

    nlp = load_perturbation_nlp() if nlp is None else nlp
    negatives, skipped = [], []
    subtle_candidates = {name: [] for name in TRANSFORMS}
    docs = nlp.pipe((pair['modified'] for pair in positives), batch_size=16)
    for positive, doc in zip(positives, docs):
        original_entities = tuple(doc.ents)
        screened_entities = [entity for entity in doc.ents if _conservative_entity(entity)]
        transforms = [('negation', _negation), ('entity_substitution', _entity_substitution)] + list(TRANSFORMS.items())
        for condition, function in transforms:
            # Restrict only entity substitution. Numeric/date and other subtle
            # rules must retain the complete original NER annotation.
            doc.set_ents(screened_entities if condition == 'entity_substitution' else original_entities)
            rng = _rng(seed, positive['id'], doc.text, 'matched_' + condition)
            change = function(doc, rng)
            if change is None or change['modified'] == doc.text:
                skipped.append({'matched_positive_id': positive['id'], 'condition': condition,
                                'source_id': positive['source_id'], 'reason': 'no_applicable_change_on_positive_output'})
                continue
            digest = hashlib.sha256((positive['id'] + '|' + condition + '|' + change['modified']).encode()).hexdigest()[:24]
            pair_id = 'opai_matched:' + digest
            pair = {
                **positive, 'id': pair_id, 'pair_id': pair_id,
                'dataset': 'opai_matched_negative', 'perturbation': 'matched_' + condition,
                'modified': change['modified'], 'target_original': change['target_original'],
                'target_modified': change['target_modified'],
                'metadata': {
                    **positive['metadata'], 'comparison': 'same_edit_with_controlled_corruption',
                    'matched_positive_id': positive['id'], 'comparator_positive_id': positive['id'],
                    'perturbation_on': 'positive_modified_output', 'base_operation': positive['metadata']['operation'],
                    'mutation': change['metadata'], 'seed': seed,
                    'conservative_entity_screening': condition == 'entity_substitution',
                    'original_ner_entity_count': len(original_entities),
                    'screened_ner_entity_count': len(screened_entities),
                    'expected_semantic_class': 'altered', 'evaluation_label': 1,
                    'semantic_label_verified': False, 'positive_control_is_presumed_not_gold': False,
                    'local_target_scope': 'edited_sentence_before_vs_after_incremental_corruption',
                    'length_ratio': len(change['modified'].split()) / len(positive['original'].split()),
                    'text_changed': change['modified'] != positive['original'],
                },
            }
            if condition in subtle_candidates:
                subtle_candidates[condition].append(pair)
            else:
                negatives.append(pair)
    subtle_selection = {}
    for condition, candidates in subtle_candidates.items():
        # Select solely by applicability and the existing domain×editor
        # round-robin. Never inspect an NLI or similarity score.
        chosen = _balanced_selection(candidates, n_per_subtle, random.Random(f'{seed}|{condition}|matched-selection'))
        negatives.extend(chosen)
        subtle_selection[condition] = {
            'eligible': len(candidates), 'requested': n_per_subtle, 'selected': len(chosen),
            'domain_counts': dict(Counter(pair['domain'] for pair in chosen)),
            'generator_counts': dict(Counter(pair['generator'] for pair in chosen)),
        }
    manifest = {
        'n_positive_pairs_attempted': len(positives), 'n_negative_pairs': len(negatives),
        'applied_counts': dict(Counter(pair['perturbation'] for pair in negatives)),
        'skipped_counts': dict(Counter(row['condition'] for row in skipped)),
        'skipped': skipped,
        'subtle_selection': subtle_selection,
        'matched_design': 'before -> edit versus same before -> same edit plus controlled corruption',
        'labels_are_intended_perturbations_not_gold': True,
        'entity_screening': 'Reject newline/sentence/punctuation spans; PERSON 2-5 titlecase alphabetic words; finite GPE/LOC lexicon; ORG markers or finite acronym list; DATE/CARDINAL retain shared typed rules. No NLI filtering.',
    }
    return negatives, manifest
