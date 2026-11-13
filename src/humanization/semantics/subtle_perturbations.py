
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import random
import re
from typing import Any, Iterable

from humanization.semantics.perturbations import (
    _edit, _replace_edits, _rng, _source_identifier, _stable_hex,
    load_perturbation_nlp, select_base_records,
)

SUBTLE_PERTURBATIONS = (
    "number_change", "quantifier_strengthening", "hedge_removal", "unsupported_detail",
)
_NEGATIVE = {"not", "n't", "never", "neither", "no", "none", "without"}
_WORD_NUMBERS = {
    "two": "three", "three": "four", "four": "five", "five": "six",
    "six": "seven", "seven": "eight", "eight": "nine", "nine": "ten",
    "ten": "eleven", "eleven": "twelve", "twelve": "thirteen",
    "fifteen": "sixteen", "twenty": "twenty-one", "thirty": "thirty-one",
    "fifty": "fifty-one",
}
_MONTHS = {"january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"}
_NUMBER_IDENTIFIERS = {'chapter', 'charter', 'section', 'article', 'figure', 'table',
                       'phase', 'level', 'version', 'type', 'part', 'appendix',
                       'episode', 'series', 'model', 'route', 'highway'}
_EVENT_VERBS = {
    "announce", "arrive", "attend", "celebrate", "complete", "depart", "enter",
    "finish", "launch", "leave", "meet", "open", "organize", "present", "publish",
    "release", "report", "return", "sign", "speak", "visit", "vote", "win",
}
_PHYSICAL_EVENT_VERBS = {'arrive', 'depart', 'enter', 'leave', 'return', 'meet', 'visit'}
_PHYSICAL_SUBJECTS = {'man', 'woman', 'boy', 'girl', 'child', 'person', 'people', 'traveler',
                      'traveller', 'soldier', 'officer', 'driver', 'passenger', 'student',
                      'doctor', 'nurse', 'cat', 'dog', 'horse', 'captain', 'crew', 'team', 'group'}


def _safe_sentence(sentence: Any, *, allow_modals: bool = False) -> bool:
    words = [token for token in sentence if token.is_alpha]
    if not 5 <= len(words) <= 60 or not sentence.text.rstrip().endswith("."):
        return False
    if len([line for line in sentence.text.splitlines() if line.strip()]) > 1:
        return False
    if any(token.text in {'"', '“', '”', '?', '!'} or token.text.casefold() in _NEGATIVE for token in sentence):
        return False
    if not allow_modals and any(token.tag_ == "MD" for token in sentence):
        return False
    return True


def _change(doc: Any, sentence: Any, edits: list[dict[str, Any]], rule: str,
            interpretation: str, **metadata: Any) -> dict[str, Any]:
    local_edits = [dict(edit, start_char=edit['start_char'] - sentence.start_char,
                        end_char=edit['end_char'] - sentence.start_char)
                   for edit in edits if sentence.start_char <= edit['start_char']
                   and edit['end_char'] <= sentence.end_char]
    return {
        'modified': _replace_edits(doc.text, edits),
        'target_original': sentence.text,
        'target_modified': _replace_edits(sentence.text, local_edits),
        'metadata': {'rule': rule, 'edits': edits,
                     'target_start_char': sentence.start_char, 'target_end_char': sentence.end_char,
                     'sentence_index': next(i for i, s in enumerate(doc.sents) if s.start == sentence.start),
                     'semantic_interpretation': interpretation,
                     'semantic_label_verified': False, **metadata},
    }


def _numeric_replacement(token: Any) -> str | None:
    value = token.text.casefold()
    if token.ent_type_ in {'DATE', 'TIME', 'ORDINAL'} or token.pos_ != 'NUM':
        return None
    if token.ent_type_ not in {'CARDINAL', 'PERCENT', 'MONEY', 'QUANTITY'} and token.dep_ != 'nummod':
        return None
    nearby = list(token.doc[max(0, token.i - 2):token.i + 3])
    if any(t.text.casefold() in _MONTHS or t.ent_type_ in {'DATE', 'TIME'} for t in nearby):
        return None
    if any(t.text.casefold().replace('.', '') in {'bce', 'bc', 'ce', 'ad'} for t in nearby):
        return None  # NER talvolta non riconosce anni storici, ad esempio 750 BCE.
    if token.i and token.nbor(-1).lemma_.casefold() in _NUMBER_IDENTIFIERS:
        return None  # Chapter 9 e Charter 77 sono identificatori, non quantità.
    if value in _WORD_NUMBERS:
        # Evita di cambiare solo il moltiplicatore, ad esempio in "three hundred".
        if token.i and token.doc[token.i - 1].pos_ == 'NUM':
            return None
        replacement = _WORD_NUMBERS[value]
    elif re.fullmatch(r'\d+(?:\.\d+)?', value):
        try:
            number = Decimal(value)
        except InvalidOperation:
            return None
        if number <= 1 or (number == int(number) and 1800 <= number <= 2199):
            return None  # Esclude cambiamenti al singolare e anni plausibili anche se NER non li riconosce.
        decimals = len(value.split('.')[1]) if '.' in value else 0
        step = max(Decimal(1) / (10 ** decimals), number * Decimal('.1'))
        new = (number + step).quantize(Decimal(1) / (10 ** decimals))
        if token.ent_type_ == 'PERCENT' and number <= 100 and new > 100:
            new = (number * Decimal('.9')).quantize(Decimal(1) / (10 ** decimals))
        replacement = f'{new:.{decimals}f}'
    else:
        return None
    return replacement.capitalize() if token.text[0].isupper() else replacement


def number_change(doc: Any, rng: random.Random) -> dict[str, Any] | None:
    eligible = {}
    for sentence in doc.sents:
        if not _safe_sentence(sentence, allow_modals=True):
            continue
        for token in sentence:
            replacement = _numeric_replacement(token)
            if replacement:
                eligible.setdefault(token.text.casefold(), (token, replacement))
    candidates = []
    for value, (token, replacement) in eligible.items():
        occurrences = [t for t in doc if t.text.casefold() == value]
        if not occurrences or any(_numeric_replacement(t) is None for t in occurrences):
            continue
        edits = [_edit(t.idx, t.idx + len(t.text), t.text,
                       replacement.capitalize() if t.text[0].isupper() else replacement.lower())
                 for t in occurrences]
        if replacement.casefold() in {t.text.casefold() for t in doc}:
            continue
        candidates.append((token, edits))
    if not candidates:
        return None
    token, edits = rng.choice(candidates)
    return _change(doc, token.sent, edits, 'quantitative_value_change_same_surface_consistent',
                   'changed_quantity_not_gold_contradiction',
                   source_value=token.text, replacement_value=edits[0]['replacement'],
                   surface_occurrences_replaced=len(edits), excludes_dates_ordinals=True)


def quantifier_strengthening(doc: Any, rng: random.Random) -> dict[str, Any] | None:
    candidates = []
    for sentence in doc.sents:
        if not _safe_sentence(sentence):
            continue
        for token in sentence:
            if token.text.casefold() not in {'some', 'many', 'several'} or token.dep_ not in {'det', 'amod', 'nummod'}:
                continue
            noun = token.head
            if noun.pos_ not in {'NOUN', 'PROPN'} or noun.morph.get('Number') != ['Plur']:
                continue
            if noun.dep_ not in {'nsubj', 'nsubjpass', 'dobj', 'obj'} or noun.head != sentence.root:
                continue
            if any(t.text.casefold() in {'if', 'unless', 'whether'} for t in sentence):
                continue
            replacement = 'All' if token.text[0].isupper() else 'all'
            candidates.append((sentence, _edit(token.idx, token.idx + len(token.text), token.text, replacement)))
    if not candidates:
        return None
    sentence, edit = rng.choice(candidates)
    return _change(doc, sentence, [edit], 'plural_matrix_argument_some_many_several_to_all',
                   'stronger_quantification_unsupported_commitment_not_gold_contradiction')


def _affirmative_auxiliary(subject: Any) -> str | None:
    if subject.text.casefold() in {'i', 'you', 'we', 'they'}:
        return 'do'
    if subject.text.casefold() in {'he', 'she', 'it'}:
        return 'does'
    if any(child.dep_ == 'conj' for child in subject.children):
        return 'do'
    number = subject.morph.get('Number')
    return 'does' if number == ['Sing'] else ('do' if number == ['Plur'] else None)


def hedge_removal(doc: Any, rng: random.Random) -> dict[str, Any] | None:
    candidates = []
    for sentence in doc.sents:
        if not _safe_sentence(sentence, allow_modals=True):
            continue
        root = sentence.root
        if root.pos_ not in {'VERB', 'AUX'}:
            continue
        for token in sentence:
            if token.text.casefold() in {'possibly', 'probably', 'perhaps', 'apparently'} and token.dep_ == 'advmod' and token.head == root:
                if token.i <= sentence.start or token.i + 1 >= sentence.end:
                    continue
                if token.nbor(-1).is_punct or token.nbor(1).is_punct:
                    continue
                end = token.idx + len(token.text_with_ws)
                candidates.append((sentence, [_edit(token.idx, end, doc.text[token.idx:end], '')], 'delete_epistemic_adverb'))
            if token.text.casefold() not in {'may', 'might'} or token.tag_ != 'MD' or token.head != root:
                continue
            subjects = [c for c in root.children if c.dep_ in {'nsubj', 'nsubjpass'}]
            if len(subjects) != 1 or any(t.tag_ == 'MD' and t != token for t in sentence):
                continue
            auxiliary = _affirmative_auxiliary(subjects[0])
            if auxiliary is None:
                continue
            if root.lemma_ == 'be' and root.i == token.i + 1:
                if subjects[0].text.casefold() == 'i':
                    replacement = 'am'
                else:
                    replacement = 'is' if auxiliary == 'does' else 'are'
                start, end = token.idx, root.idx + len(root.text)
                candidates.append((sentence, [_edit(start, end, doc.text[start:end], replacement)], 'modal_may_might_be_to_finite_be'))
            elif root.pos_ == 'VERB' and root.tag_ == 'VB' and root.i == token.i + 1:
                if root.lemma_ in {'have', 'do'} or any(c.dep_ in {'aux', 'auxpass'} and c != token for c in root.children):
                    continue
                candidates.append((sentence, [_edit(token.idx, token.idx + len(token.text), token.text, auxiliary)], 'epistemic_modal_to_assertive_do_support'))
    if not candidates:
        return None
    sentence, edits, rule = rng.choice(candidates)
    return _change(doc, sentence, edits, rule,
                   'removed_epistemic_caution_stronger_commitment_not_gold_contradiction')


def unsupported_detail(doc: Any, rng: random.Random) -> dict[str, Any] | None:
    candidates = []
    clocks = [value for value in ['9:17 a.m.', '2:43 p.m.', '11:26 a.m.'] if value not in doc.text]
    if not clocks:
        return None
    for sentence in doc.sents:
        root = sentence.root
        if not _safe_sentence(sentence) or root.lemma_ not in _EVENT_VERBS or root.tag_ != 'VBD':
            continue
        if any(t.ent_type_ in {'DATE', 'TIME'} for t in sentence):
            continue
        subjects = [c for c in root.children if c.dep_ == 'nsubj']
        if len(subjects) != 1:
            continue
        subject = subjects[0]
        subject_tokens = list(subject.subtree)
        if any(t.i >= root.i or t.dep_ in {'relcl', 'ccomp', 'xcomp', 'advcl'} for t in subject_tokens):
            continue
        named_actor = subject.ent_type_ in {'PERSON', 'ORG'}
        physical_actor = root.lemma_ in _PHYSICAL_EVENT_VERBS and (
            subject.text.casefold() in {'i', 'he', 'she', 'we', 'they'}
            or subject.lemma_.casefold() in _PHYSICAL_SUBJECTS)
        if not (named_actor or physical_actor):
            continue
        if any(t.dep_ == 'conj' and t.pos_ in {'VERB', 'AUX'} for t in sentence):
            continue  # Non assegna tempi ambigui a più eventi coordinati.
        last = max(subject_tokens, key=lambda t:t.i)
        end = last.idx + len(last.text)
        if last.is_punct or (end < len(doc.text) and not doc.text[end].isspace()):
            continue
        clock = rng.choice(clocks)
        edit = _edit(end, end, '', ', at ' + clock + ',')
        candidates.append((sentence, edit, clock))
    if not candidates:
        return None
    sentence, edit, clock = rng.choice(candidates)
    return _change(doc, sentence, [edit], 'parenthetical_specific_time_after_actor_of_untimed_past_event',
                   'added_unsupported_specific_detail_not_verified_false_or_contradictory',
                   added_detail=clock, original_event_lemma=sentence.root.lemma_,
                   original_target_had_recognized_date_or_time=False)


TRANSFORMS = {
    'number_change': number_change,
    'quantifier_strengthening': quantifier_strengthening,
    'hedge_removal': hedge_removal,
    'unsupported_detail': unsupported_detail,
}


def generate_subtle_pairs(records: Iterable[dict[str, Any]], *, n_per_type: int = 50,
                          seed: int = 42, preferred_source_ids: set[str] | None = None,
                          nlp: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = select_base_records(records, n=10**9, seed=seed)
    preferred_source_ids = set(preferred_source_ids or ())
    nlp = load_perturbation_nlp() if nlp is None else nlp
    choices = {name: [] for name in SUBTLE_PERTURBATIONS}
    screening = []
    for record, doc in zip(records, nlp.pipe((r['text'] for r in records), batch_size=16)):
        source_id = _source_identifier(record)
        eligible = []
        for name, transform in TRANSFORMS.items():
            change = transform(doc, _rng(seed, source_id, doc.text, name))
            if change is None or change['modified'] == doc.text:
                continue
            choices[name].append((record, change))
            eligible.append(name)
        screening.append({'source_id': source_id, 'domain': record['domain'],
                          'generator': record['generator'], 'eligible': eligible,
                          'in_original_200_cohort': source_id in preferred_source_ids})
    pairs, counts = [], {}
    for name, candidates in choices.items():
        lookup = {_source_identifier(record): (record, change) for record, change in candidates}
        preferred = [record for record, _ in candidates if _source_identifier(record) in preferred_source_ids]
        selected = select_base_records(preferred, n=n_per_type, seed=seed)
        selected_ids = {_source_identifier(record) for record in selected}
        if len(selected) < n_per_type:
            additional = [record for record, _ in candidates if _source_identifier(record) not in selected_ids]
            selected.extend(select_base_records(additional, n=n_per_type-len(selected), seed=seed))
        for record in selected:
            source_id = _source_identifier(record)
            change = lookup[source_id][1]
            pair_id = f'{source_id}:{name}:{_stable_hex(seed, source_id, name, change["modified"])[:16]}'
            pair = {'id': pair_id, 'pair_id': pair_id, 'source_id': source_id,
                    'source_group_id': str(record.get('group_id', record.get('source_group_id', source_id))),
                    'domain': record['domain'], 'generator': record['generator'],
                    'original': record['text'], **change,
                    'dataset': 'controlled_subtle', 'perturbation': name, 'status': 'applied'}
            pair['metadata']['source_selection_tier'] = 'original_200_cohort' if source_id in preferred_source_ids else 'extended_900_machine_candidates'
            pairs.append(pair)
        counts[name] = {'eligible': len(candidates), 'eligible_in_original_200': len(preferred),
                        'selected': len(selected),
                        'selected_in_original_200': sum(_source_identifier(r) in preferred_source_ids for r in selected),
                        'domain_counts': dict(Counter(r['domain'] for r in selected)),
                        'generator_counts': dict(Counter(r['generator'] for r in selected))}
    manifest = {'seed': seed, 'requested_per_type': n_per_type,
                'candidate_count': len(records), 'original_cohort_source_count': len(preferred_source_ids),
                'n_pairs': len(pairs), 'counts_by_perturbation': counts,
                'all_requested_counts_met': all(c['selected'] == n_per_type for c in counts.values()),
                'policy': 'Independent applicability by condition; prefer original200, fill from900; domain/generator balancing within each tier.',
                'semantic_labels_are_rule_intentions_not_gold': True,
                'screening': screening}
    return pairs, manifest
