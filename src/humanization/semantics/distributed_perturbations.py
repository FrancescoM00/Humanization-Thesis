"""Progressive, sentence-local corruptions on the same OpAI edited outputs.

Each anchor has nested K=1/2/3/5 changes. The main panel requires five
applicable distinct sentences before any score is inspected. An applicability
panel with two to four sentences is labelled separately and must not be pooled
as though it had a five-sentence counterpart. Labels are rule intentions, not
human semantic gold.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
import re
from typing import Any

from humanization.datasets.opai import _balanced_selection
from humanization.semantics.metrics import _sentences
from humanization.semantics.perturbations import _negation, _replace_edits, _rng, load_perturbation_nlp
from humanization.semantics.subtle_perturbations import TRANSFORMS

SENTENCE_COUNTS = (1, 2, 3, 5)
FAMILIES = ('distributed_negation', 'distributed_subtle')


def _candidate(sentence, condition, transform, positive, seed):
    # as_doc preserves the already computed POS/dependency/NER annotations.
    # Restricting the input means repeated numbers in other sentences cannot
    # silently turn a requested K-sentence edit into more than K edits.
    local = sentence.as_doc()
    rng = _rng(seed, positive['id'], str(sentence.start_char), condition)
    change = transform(local, rng)
    if change is None or change['modified'] == local.text:
        return None
    edits = []
    for edit in change['metadata']['edits']:
        start, end = edit['start_char'], edit['end_char']
        if not (0 <= start <= end <= len(sentence.text)) or local.text[start:end] != edit['original']:
            raise ValueError('Sentence-local transform produced stale offsets')
        if condition == 'number_change' and re.search(
                r'\b(?:league|division|formula|world\s+war|grade|class)\s+$',
                local.text[max(0, start-30):start], re.I):
            return None  # League Two denotes an identifier, not a quantity.
        edits.append({**edit, 'start_char': sentence.start_char + start,
                      'end_char': sentence.start_char + end})
    if not edits:
        return None
    # Span.as_doc() may append the whitespace after the final token even
    # though Span.text/end_char exclude it. Targets must match exact offsets.
    target_original = sentence.text
    local_edits = [{**edit,'start_char':edit['start_char']-sentence.start_char,
                    'end_char':edit['end_char']-sentence.start_char} for edit in edits]
    target_modified = _replace_edits(target_original, local_edits)
    return {'condition': condition, 'sentence_index': None,
            'start_char': sentence.start_char, 'end_char': sentence.end_char,
            'original': target_original, 'modified': target_modified,
            'edits': edits, 'rule': change['metadata'].get('rule'),
            'semantic_interpretation': change['metadata'].get('semantic_interpretation')}


def sentence_candidates(doc, positive, seed):
    sentences = list(doc.sents)
    canonical = _sentences(doc.text)
    spans, cursor = [], 0
    for text in canonical:
        start = doc.text.find(text, cursor)
        if start < 0:
            raise ValueError('Cannot locate the metric sentence segmentation in the original text')
        spans.append((start, start+len(text), text))
        cursor = start+len(text)
    normalized = Counter(re.sub(r'\s+', ' ', s.text).strip().casefold() for s in sentences)
    by_unit = {family: {} for family in FAMILIES}
    def register(family, change, parser_index):
        if change is None:
            return
        units = [i for i,(start,end,_) in enumerate(spans)
                 if all(start <= edit['start_char'] <= edit['end_char'] <= end
                        for edit in change['edits'])]
        if len(units) != 1:
            return
        unit_index = units[0]
        start, end, unit = spans[unit_index]
        edits = [{**edit,'start_char':edit['start_char']-start,
                  'end_char':edit['end_char']-start} for edit in change['edits']]
        changed_unit = _replace_edits(unit, edits)
        if len(_sentences(changed_unit)) != 1 or changed_unit == unit:
            return
        change.update(sentence_index=unit_index, canonical_sentence_index=unit_index,
                      parser_sentence_index=parser_index)
        by_unit[family].setdefault(unit_index, []).append(change)
    for index, sentence in enumerate(sentences):
        # Duplicate untouched assertions were a problem in the first pilot.
        # Reject exact duplicates and the previously ambiguous existential scope.
        if normalized[re.sub(r'\s+', ' ', sentence.text).strip().casefold()] != 1:
            continue
        if not re.search(r'\bone\s+of\b', sentence.text, re.I):
            change = _candidate(sentence, 'negation', _negation, positive, seed)
            if change is not None:
                # "has not also ..." was an unnecessarily ambiguous target.
                if not re.search(r"\b(?:has|have|had|is|are|was|were)\s+not\s+also\b",
                                 change['modified'], re.I):
                    register('distributed_negation', change, index)
        subtle = []
        for name, transform in TRANSFORMS.items():
            change = _candidate(sentence, name, transform, positive, seed)
            if change is not None:
                subtle.append(change)
        if subtle:
            for change in subtle:
                register('distributed_subtle', change, index)
    candidates = {family:list(by_unit[family].values()) for family in FAMILIES}
    return candidates, len(canonical), len(sentences)


def _ordered_changes(options, positive, family, seed):
    rng = _rng(seed, positive['id'], family, 'sentence_order')
    options = list(options)
    rng.shuffle(options)
    chosen = []
    for choices in options:
        # Choose among available rules without putting a chosen rule type
        # systematically first at K=1. Record composition so it can be checked.
        selected = rng.choice(sorted(choices, key=lambda x: x['condition']))
        chosen.append(selected)
    return chosen


def _pair(positive, changes, family, k, n_sentences, n_parser_sentences, common_panel, seed):
    selected = changes[:k]
    source = positive['modified']
    edits = [edit for change in selected for edit in change['edits']]
    modified = _replace_edits(source, edits)
    assert modified != source and len({x['sentence_index'] for x in selected}) == k
    before_canonical, after_canonical = _sentences(source), _sentences(modified)
    if len(before_canonical) != len(after_canonical):
        raise ValueError('Distributed edit changes the metric sentence segmentation')
    observed = sum(a != b for a,b in zip(before_canonical, after_canonical))
    if observed != k:
        raise ValueError(f'Requested {k} changes but metric segmentation observes {observed}')
    for change in selected:
        local = [{**edit, 'start_char': edit['start_char']-change['start_char'],
                  'end_char': edit['end_char']-change['start_char']} for edit in change['edits']]
        assert _replace_edits(change['original'], local) == change['modified']
    key = json.dumps([positive['id'], family, k, modified], ensure_ascii=False)
    pair_id = 'opai_distributed:' + hashlib.sha256(key.encode()).hexdigest()[:24]
    return {
        **positive, 'id': pair_id, 'pair_id': pair_id,
        'dataset': 'opai_distributed_negative', 'perturbation': f'{family}_k{k}',
        'modified': modified, 'target_original': None, 'target_modified': None,
        'corruption_family': family, 'n_corrupted_sentences': k,
        'metadata': {
            **positive.get('metadata', {}),
            'comparison': 'same_edit_with_distributed_corruption',
            'matched_positive_id': positive['id'],
            'corruption_family': family, 'n_corrupted_sentences': k,
            'maximum_applicable_sentences': len(changes),
            'original_edited_sentence_count': n_sentences,
            'original_parser_sentence_count': n_parser_sentences,
            'metric_sentence_segmentation': 'humanization.semantics.metrics._sentences (spaCy blank English sentencizer)',
            'observed_changed_metric_sentences': observed,
            'metric_sentence_count_preserved': len(before_canonical)==len(after_canonical),
            'corrupted_sentence_fraction': k / n_sentences,
            'common_panel': common_panel,
            'panel': 'common_5_primary' if common_panel else 'applicability_2to4_secondary',
            'nested_sentence_order': [x['sentence_index'] for x in changes],
            'corrupted_sentence_indices': [x['sentence_index'] for x in selected],
            'corruption_subtypes': dict(Counter(x['condition'] for x in selected)),
            'affected_sentences': selected, 'edits': edits, 'seed': seed,
            'perturbation_on': 'positive_modified_output',
            'base_operation': positive.get('metadata', {}).get('operation'),
            'semantic_label_verified': False, 'evaluation_label': 1,
            'expected_semantic_class': 'altered',
            'numeric_consistency_scope': 'within each corrupted sentence only',
            'selection_basis': 'applicability and seeded strata; no metric used',
            'progression': 'same first K sentences and same per-sentence edits for all K',
        },
    }


def generate_distributed_pairs(positives, *, n_per_family=60, seed=20261105,
                               nlp=None, progress_callback=None):
    positives = list(positives)
    nlp = load_perturbation_nlp() if nlp is None else nlp
    screened = {family: [] for family in FAMILIES}
    screening_rows = []
    for index, (positive, doc) in enumerate(zip(positives, nlp.pipe(
            (p['modified'] for p in positives), batch_size=16)), 1):
        candidates, n_sentences, n_parser_sentences = sentence_candidates(doc, positive, seed)
        for family in FAMILIES:
            changes = _ordered_changes(candidates[family], positive, family, seed)
            screening_rows.append({'matched_positive_id': positive['id'],
                                   'source_group_id': positive['source_group_id'],
                                   'domain': positive['domain'], 'generator': positive['generator'],
                                   'evaluation_split': positive['evaluation_split'],
                                   'corruption_family': family,
                                   'n_original_sentences': n_sentences,
                                   'n_original_parser_sentences': n_parser_sentences,
                                   'n_applicable_sentences': len(changes),
                                   'available_subtypes': dict(Counter(x['condition'] for x in changes))})
            if len(changes) >= 2:
                screened[family].append({'positive': positive, 'changes': changes,
                                        'n_sentences': n_sentences,
                                        'n_parser_sentences': n_parser_sentences,
                                        'domain': positive['domain'], 'generator': positive['generator']})
        if progress_callback and (index % 25 == 0 or index == len(positives)):
            progress_callback(index, len(positives))
    pairs, families = [], {}
    for family in FAMILIES:
        common = [x for x in screened[family] if len(x['changes']) >= 5]
        limited = [x for x in screened[family] if 2 <= len(x['changes']) < 5]
        chosen = _balanced_selection(common, n_per_family, random.Random(f'{seed}:{family}:primary'))
        chosen += _balanced_selection(limited, max(0, n_per_family-len(chosen)),
                                      random.Random(f'{seed}:{family}:secondary'))
        family_pairs = []
        for item in chosen:
            positive, changes = item['positive'], item['changes']
            for k in SENTENCE_COUNTS:
                if len(changes) >= k:
                    pair = _pair(positive, changes, family, k, item['n_sentences'],
                                 item['n_parser_sentences'],len(changes)>=5, seed)
                    for key in ['original', 'source_id', 'source_group_id', 'evaluation_split']:
                        assert pair[key] == positive[key]
                    family_pairs.append(pair)
        pairs.extend(family_pairs)
        families[family] = {
            'n_applicable_at_least_2': len(screened[family]),
            'n_applicable_at_least_5': len(common),
            'n_applicable_2_to_4': len(limited), 'n_selected_positive_anchors': len(chosen),
            'n_selected_source_groups': len({x['positive']['source_group_id'] for x in chosen}),
            'n_selected_common_5': sum(len(x['changes'])>=5 for x in chosen),
            'n_selected_limited_2_to_4': sum(len(x['changes'])<5 for x in chosen),
            'n_pairs': len(family_pairs),
            'counts_by_k': dict(Counter(str(x['n_corrupted_sentences']) for x in family_pairs)),
            'counts_by_domain': dict(Counter(x['positive']['domain'] for x in chosen)),
            'counts_by_generator': dict(Counter(x['positive']['generator'] for x in chosen)),
            'counts_by_split_k': dict(Counter(x['evaluation_split']+':k'+str(x['n_corrupted_sentences'])
                                              for x in family_pairs)),
            'counts_by_subtype_k': {str(k):dict(Counter(subtype for pair in family_pairs
                                       if pair['n_corrupted_sentences']==k
                                       for subtype,count in pair['metadata']['corruption_subtypes'].items()
                                       for _ in range(count))) for k in SENTENCE_COUNTS},
        }
    assert len(pairs) <= n_per_family * 2 * len(SENTENCE_COUNTS)
    assert len({p['id'] for p in pairs}) == len(pairs)
    manifest = {'seed': seed, 'requested_anchors_per_family': n_per_family,
                'n_positive_inputs': len(positives), 'n_pairs': len(pairs),
                'requested_sentence_counts': list(SENTENCE_COUNTS), 'families': families,
                'screening': screening_rows, 'labels_are_rule_intentions_not_gold': True,
                'selection_independent_of_model_scores': True,
                'primary_panel_definition': 'same anchors with >=5 eligible distinct sentences, K1/2/3/5 nested',
                'secondary_panel_definition': 'anchors with 2-4 eligible sentences, only available K, analyse separately',
                'subtle_composition_note': 'Rule options vary by sentence. Composition by K is reported; this is not a real-humanizer error benchmark.',
                'consistency_note': 'Each rule operates within its selected sentence. Other mentions elsewhere remain unchanged.',
                'segmentation_validation': 'Every output preserves metric sentence count and changes exactly K canonical metric sentences. Parser and metric counts retained separately.',
                'no_original_inputs_modified': True}
    return pairs, manifest
