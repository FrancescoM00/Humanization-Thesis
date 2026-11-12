"""Deterministic, auditable challenges for semantic-preservation metrics.

These transformations supply *experimental conditions*, not ground-truth
semantic labels. An inserted negation is a contradiction candidate, entity
substitution changes a surface referent without resolving coreference, sentence
shuffling can preserve every proposition, and truncation/deletion can remove
information rather than contradict it. ``sanity_fixtures`` supplies a separate,
small manually specified set where the expected NLI directions are explicit.

All offsets refer to Unicode character positions in the original document,
unless a key explicitly says otherwise. No model or resource is downloaded.
"""

from __future__ import annotations

from collections import defaultdict, deque
from hashlib import sha256
import math
import random
import re
from typing import Any, Iterable


PERTURBATIONS = (
    "identity", "negation", "entity_substitution", "sentence_shuffle",
    "truncate", "random_deletion",
)

# Fixed before any metric is examined. Values intentionally include alternatives
# so an original match or existing-document collision can be rejected.
ENTITY_REPLACEMENTS = {
    "PERSON": ("Alex Morgan", "Sam Rivera", "Jordan Chen"),
    "ORG": ("Microsoft", "Toyota", "UNESCO"),
    "ORG:article": ("World Bank", "United Nations", "European Commission"),
    "GPE": ("Canada", "Japan", "Brazil"),
    "GPE:article": ("United Kingdom", "United States", "Netherlands"),
    "LOC": ("Lake Victoria", "Lake Tanganyika", "Lake Geneva"),
    "LOC:article": ("Sahara Desert", "Pacific Ocean", "Atlantic Ocean"),
    "DATE:year": ("1998", "2011", "2032"),
    "DATE:calendar": ("June 15, 1997", "March 8, 2014", "October 12, 2021"),
    "DATE:relative": ("yesterday", "today", "tomorrow"),
    "DATE:weekday": ("Monday", "Thursday", "Saturday"),
    "DATE:month": ("February", "July", "November"),
    "TIME": ("9:15 a.m.", "3:45 p.m.", "11:30 p.m."),
    "MONEY": ("$47 million", "$83 million", "$26 million"),
    "PERCENT": ("37%", "62%", "84%"),
    "CARDINAL:digit": ("7", "14", "29"),
    "CARDINAL:word": ("seven", "fourteen", "twenty-nine"),
    "ORDINAL": ("third", "seventh", "eleventh"),
}

# Clean entity policy (v2), derived from the manual review of 40 perturbations
# (results_semantic_extension/review40.md and the student's own review). Kept as a
# separate condition so that the published v1 results stay reproducible.
# Dropped on purpose: ORG (company/product/disease/acronym/outlet are mixed up by
# NER and the three replacements are not sub-type compatible), LOC (granularity
# mismatch, e.g. Earth -> a lake), TIME (collides with morning/evening wording) and
# DATE:relative (collides with verb tense: "we are seeing tomorrow").
CLEAN_ENTITY_KEYS = frozenset({
    "PERSON", "GPE", "GPE:article", "MONEY", "PERCENT", "CARDINAL:digit",
    "CARDINAL:word", "ORDINAL", "DATE:year", "DATE:calendar", "DATE:weekday",
    "DATE:month",
})
_PERSON_NAME = re.compile(r"[A-Z][A-Za-z'\u2019.-]+(?: [A-Z][A-Za-z'\u2019.-]+)*")
_NON_PERSON_ENT_TYPES = frozenset({"ORG", "GPE", "LOC", "FAC", "PRODUCT", "WORK_OF_ART", "NORP", "EVENT"})

_NEGATIVE_WORDS = frozenset({
    "not", "n't", "never", "no", "neither", "nor", "nothing", "nobody",
    "none", "without", "hardly", "rarely", "scarcely",
})
_NONASSERTIVE_ROOTS = frozenset({
    "think", "believe", "hope", "want", "wish", "imagine", "seem", "appear",
    "suggest", "feel", "say", "claim", "argue", "ask", "tell", "write",
    "wonder", "expect", "suppose", "assume",
})
_CLAUSAL_DEPENDENCIES = frozenset({"ccomp", "xcomp", "advcl", "csubj", "relcl", "acl"})
_AMBIGUOUS_SUBJECT_QUANTIFIERS = frozenset({
    "some", "all", "each", "every", "any", "most", "many", "few", "several",
    "both", "either", "someone", "somebody", "something", "anyone", "anybody",
})


def _stable_hex(seed: int, *parts: Any) -> str:
    value = "\0".join((str(seed), *(str(part) for part in parts)))
    return sha256(value.encode("utf-8")).hexdigest()


def _rng(seed: int, source_id: str, text: str, perturbation: str) -> random.Random:
    return random.Random(int(_stable_hex(seed, source_id, text, perturbation), 16))


def load_perturbation_nlp(model: str = "en_core_web_sm") -> Any:
    """Load all components, including NER; fail rather than auto-download."""
    import spacy

    nlp = spacy.load(model)
    required = {"parser", "ner", "lemmatizer"}
    if not required.issubset(nlp.pipe_names):
        raise ValueError(f"Perturbations require model components: {sorted(required)}")
    return nlp


def select_base_records(
    records: Iterable[dict[str, Any]], n: int = 200, seed: int = 42,
) -> list[dict[str, Any]]:
    """Select unique machine texts, balancing domain first and generator second.

    Sorted SHA-256 ranks produce a sample independent of input order and Python
    hash randomization. Round-robin allocation differs by at most one across
    equally populated domains and, within each domain, equally populated model
    cells. Return fewer than n only if fewer unique eligible texts exist.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    candidates = sorted(
        (record for record in records if str(record.get("label", "")) == "1" and record.get("text", "").strip()),
        key=lambda record: (str(record.get("id", record.get("source_id", ""))), str(record["text"])),
    )
    seen: set[str] = set()
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in candidates:
        text_hash = sha256(record["text"].encode("utf-8")).hexdigest()
        if text_hash in seen:
            continue
        seen.add(text_hash)
        groups[str(record["domain"])][str(record["generator"])].append(record)

    # Work out domain quotas first. Rotating each domain's generator cycle by
    # previously allocated counts also balances the global generator margins
    # when the same generator set is available in every domain.
    domain_capacities = {domain: sum(len(items) for items in generators.values())
                         for domain, generators in sorted(groups.items())}
    domain_targets = {domain: 0 for domain in domain_capacities}
    assigned = 0
    while assigned < n and any(domain_targets[domain] < capacity for domain, capacity in domain_capacities.items()):
        for domain, capacity in domain_capacities.items():
            if assigned < n and domain_targets[domain] < capacity:
                domain_targets[domain] += 1
                assigned += 1

    domain_queues = {}
    generator_cursor = 0
    for domain, generators in sorted(groups.items()):
        generator_names = sorted(generators)
        offset = generator_cursor % len(generator_names)
        generator_names = generator_names[offset:] + generator_names[:offset]
        generator_queues = {
            generator: deque(sorted(generators[generator], key=lambda item: _stable_hex(seed, item.get("id", ""), item["text"])))
            for generator in generator_names
        }
        ordered = []
        while any(generator_queues.values()):
            for queue in generator_queues.values():
                if queue:
                    ordered.append(queue.popleft())
        domain_queues[domain] = deque(ordered)
        generator_cursor += domain_targets[domain]

    selected = []
    while len(selected) < n and any(domain_queues.values()):
        for queue in domain_queues.values():
            if queue and len(selected) < n:
                selected.append(queue.popleft())
    return selected


def _source_identifier(record: dict[str, Any]) -> str:
    text = record["text"]
    return str(record.get("id", record.get("source_id", sha256(text.encode()).hexdigest())))


def select_applicable_records(
    records: Iterable[dict[str, Any]], n: int = 200, seed: int = 42,
    nlp: Any = None, batch_size: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Screen genuine applicability, then sample from the eligible subpopulation.

    Every selected source must admit all five nonidentity conditions using the
    *same* conservative helpers and per-condition RNGs as pair generation. This
    intentionally changes the target population toward texts containing a simple
    affirmative clause and a supported named/numeric entity. The returned
    manifest records that bias, exclusions and per-cell counts. No source is
    fabricated or relaxed to fill a quota; fewer than n eligible sources produce
    a smaller sample and ``target_count_met=False``.

    Screening uses one complete spaCy pass over unique machine documents. Pair
    generation is separate so retained records remain JSON-serializable.
    """
    records = list(records)
    candidates = select_base_records(records, n=len(records), seed=seed)
    nlp = load_perturbation_nlp() if nlp is None else nlp
    eligible = []
    eligibility_rows = []
    excluded_by_condition = {name: 0 for name in PERTURBATIONS if name != "identity"}
    cell_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"candidate": 0, "eligible": 0, "selected": 0})
    )
    for record, doc in zip(candidates, nlp.pipe((item["text"] for item in candidates), batch_size=batch_size)):
        source_id = _source_identifier(record)
        text = record["text"]
        changes = {
            "negation": _negation(doc, _rng(seed, source_id, text, "negation")),
            "entity_substitution": _entity_substitution(doc, _rng(seed, source_id, text, "entity_substitution")),
            "sentence_shuffle": _sentence_shuffle(doc, _rng(seed, source_id, text, "sentence_shuffle")),
            "truncate": _truncate(doc),
            "random_deletion": _random_deletion(doc, _rng(seed, source_id, text, "random_deletion")),
        }
        inapplicable = [name for name, change in changes.items()
                        if change is None or change["modified"] == text]
        for name in inapplicable:
            excluded_by_condition[name] += 1
        cell = cell_counts[str(record["domain"])][str(record["generator"])]
        cell["candidate"] += 1
        if not inapplicable:
            eligible.append(record)
            cell["eligible"] += 1
        eligibility_rows.append({
            "source_id": source_id, "domain": str(record["domain"]),
            "generator": str(record["generator"]), "eligible": not inapplicable,
            "inapplicable_perturbations": inapplicable,
        })
    selected = select_base_records(eligible, n=n, seed=seed)
    for record in selected:
        cell_counts[str(record["domain"])][str(record["generator"])]["selected"] += 1
    manifest = {
        "screening_policy": "all_five_nonidentity_perturbations_genuinely_applicable",
        "seed": seed, "requested_count": n, "selected_count": len(selected),
        "target_count_met": len(selected) == n,
        "input_record_count": len(records),
        "unique_machine_candidate_count": len(candidates),
        "eligible_count": len(eligible), "excluded_count": len(candidates) - len(eligible),
        "excluded_by_condition": excluded_by_condition,
        "counts_by_domain_generator": {domain: dict(generators) for domain, generators in cell_counts.items()},
        "eligibility_rows": eligibility_rows,
        "selection_bias": (
            "Applicability-selected subpopulation: favors documents with a simple affirmative "
            "nonquantified subject clause, a supported named/numeric entity and multiple sentences. "
            "Results do not estimate performance on all machine documents."
        ),
        "semantic_label_verified": False,
    }
    return selected, manifest


def _replace_edits(text: str, edits: list[dict[str, Any]]) -> str:
    """Apply nonoverlapping original-coordinate edits, preserving every other char."""
    edits = sorted(edits, key=lambda edit: edit["start_char"])
    previous_end = 0
    chunks = []
    for edit in edits:
        start, end = edit["start_char"], edit["end_char"]
        if start < previous_end or text[start:end] != edit["original"]:
            raise ValueError("Invalid, overlapping or stale edit coordinates")
        chunks.extend((text[previous_end:start], edit["replacement"]))
        previous_end = end
    chunks.append(text[previous_end:])
    return "".join(chunks)


def _edit(start: int, end: int, original: str, replacement: str, **extra: Any) -> dict[str, Any]:
    return {"start_char": start, "end_char": end, "original": original, "replacement": replacement, **extra}


def _negation(doc: Any, rng: random.Random) -> dict[str, Any] | None:
    candidates = []
    for sentence_index, sentence in enumerate(doc.sents):
        words = [token for token in sentence if token.is_alpha]
        root = sentence.root
        if not 5 <= len(words) <= 45 or not sentence.text.rstrip().endswith("."):
            continue
        if any(token.text.casefold() in _NEGATIVE_WORDS or token.dep_ == "neg" for token in sentence):
            continue
        if root.pos_ not in {"VERB", "AUX"} or root.lemma_.casefold() in _NONASSERTIVE_ROOTS:
            continue
        subjects = [child for child in root.children if child.dep_ in {"nsubj", "nsubjpass"}]
        if not subjects:
            continue
        # "Some people left" and "Some people did not leave" can both be true;
        # existential/universal subject scopes are outside this simple probe.
        if any(token.text.casefold() in _AMBIGUOUS_SUBJECT_QUANTIFIERS
               for subject in subjects for token in subject.subtree):
            continue
        # A parser may merge an unpunctuated headline and the following sentence.
        # Avoid editing that less controlled multi-line target.
        if len([line for line in sentence.text.splitlines() if line.strip()]) > 1:
            continue
        # Avoid questions, quotations, modals, and embedded/coordinated clauses:
        # these make a local syntactic reversal harder to interpret.
        if any(token.text in {'"', "“", "”", "?", "!"} or token.tag_ == "MD" for token in sentence):
            continue
        if any(token.dep_ in _CLAUSAL_DEPENDENCIES or (token.dep_ == "conj" and token.pos_ in {"VERB", "AUX"}) for token in sentence):
            continue

        auxiliaries = sorted(
            (child for child in root.children if child.dep_ in {"aux", "auxpass"} and child.morph.get("VerbForm") == ["Fin"]),
            key=lambda child: child.i,
        )
        if auxiliaries:
            auxiliary = auxiliaries[0]
            if not auxiliary.text.isalpha():
                continue  # Do not guess the expansion of ambiguous contractions.
            edit = _edit(auxiliary.idx, auxiliary.idx + len(auxiliary.text), auxiliary.text, auxiliary.text + " not", token_index=auxiliary.i)
            rule = "finite_auxiliary_not"
        elif root.pos_ == "AUX" and root.lemma_ == "be" and root.morph.get("VerbForm") == ["Fin"]:
            edit = _edit(root.idx, root.idx + len(root.text), root.text, root.text + " not", token_index=root.i)
            rule = "copular_be_not"
        elif root.pos_ == "VERB" and root.tag_ in {"VBD", "VBZ", "VBP"} and root.lemma_.isalpha():
            auxiliary = "did" if root.tag_ == "VBD" else "does" if root.tag_ == "VBZ" else "do"
            replacement = auxiliary + " not " + root.lemma_
            if root.text[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            edit = _edit(root.idx, root.idx + len(root.text), root.text, replacement, token_index=root.i)
            rule = "do_support_" + root.tag_.lower()
        else:
            continue
        candidates.append((sentence_index, sentence, edit, rule))
    if not candidates:
        return None
    sentence_index, sentence, edit, rule = rng.choice(candidates)
    local_edit = dict(edit, start_char=edit["start_char"] - sentence.start_char, end_char=edit["end_char"] - sentence.start_char)
    return {
        "modified": _replace_edits(doc.text, [edit]),
        "target_original": sentence.text,
        "target_modified": _replace_edits(sentence.text, [local_edit]),
        "metadata": {
            "rule": rule, "edits": [edit], "sentence_index": sentence_index,
            "target_start_char": sentence.start_char, "target_end_char": sentence.end_char,
            "eligible_sentence_count": len(candidates), "inserted_negations": 1,
            "semantic_interpretation": "contradiction_candidate_not_verified_truth_label",
        },
    }


def _replacement_key(entity: Any) -> str | None:
    label, text = entity.label_, entity.text.strip()
    lowered = text.casefold()
    external_article = entity.doc[entity.start - 1].text.casefold() if entity.start else None
    if external_article in {"a", "an"}:
        # Replacing "IBM" in "an IBM unit" would also require changing an/a,
        # adding a grammatical confound outside the named entity.
        return None
    if external_article == "the":
        if label in {"GPE", "ORG", "LOC"}:
            return label + ":article"
        if label == "PERSON":
            return None
    if label == "DATE":
        if re.fullmatch(r"(?:18|19|20|21)\d{2}", text):
            return "DATE:year"
        if lowered in {"today", "tomorrow", "yesterday"}:
            return "DATE:relative"
        if lowered in {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}:
            return "DATE:weekday"
        if lowered in {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}:
            return "DATE:month"
        if re.fullmatch(r"[A-Za-z]+\s+\d{1,2},?\s+\d{4}", text):
            return "DATE:calendar"
        return None
    if label == "CARDINAL":
        # Changing singular to plural would introduce an unnecessary grammatical
        # confound. Restrict this optional numeric condition to unambiguous >1.
        if re.fullmatch(r"\d+", text) and int(text) > 1:
            return "CARDINAL:digit"
        if lowered in {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"}:
            return "CARDINAL:word"
        return None
    return label if label in ENTITY_REPLACEMENTS else None


def _entity_substitution(doc: Any, rng: random.Random) -> dict[str, Any] | None:
    candidates = []
    seen = set()
    for entity in doc.ents:
        key = _replacement_key(entity)
        canonical = entity.text.casefold()
        if key is None or canonical in seen:
            continue
        seen.add(canonical)
        replacements = [value for value in ENTITY_REPLACEMENTS[key] if value.casefold() != canonical and value.casefold() not in doc.text.casefold()]
        if replacements:
            candidates.append((entity, key, replacements))
    if not candidates:
        return None
    entity, key, replacements = rng.choice(candidates)
    replacement = rng.choice(replacements)
    # Every case-insensitive exact surface occurrence is changed consistently,
    # even when NER failed to label a repeated occurrence. No entity linking or
    # pronoun/coreference resolution is claimed.
    matches = list(re.finditer(r"(?<!\w)" + re.escape(entity.text) + r"(?!\w)", doc.text, re.I))
    edits = [_edit(match.start(), match.end(), match.group(), replacement) for match in matches]
    sentence = entity.sent
    local_edits = [
        dict(edit, start_char=edit["start_char"] - sentence.start_char, end_char=edit["end_char"] - sentence.start_char)
        for edit in edits if sentence.start_char <= edit["start_char"] and edit["end_char"] <= sentence.end_char
    ]
    if not edits or not local_edits:
        return None
    sentence_index = next(index for index, span in enumerate(doc.sents) if span.start == sentence.start)
    return {
        "modified": _replace_edits(doc.text, edits),
        "target_original": sentence.text,
        "target_modified": _replace_edits(sentence.text, local_edits),
        "metadata": {
            "entity_type": entity.label_, "replacement_dictionary_key": key,
            "source_entity": entity.text, "replacement_entity": replacement,
            "edits": edits, "surface_occurrences_replaced": len(edits),
            "sentence_index": sentence_index, "target_start_char": sentence.start_char,
            "target_end_char": sentence.end_char,
            "semantic_interpretation": "surface_referent_change_not_verified_contradiction",
            "coreference_resolved": False,
        },
    }


def _clean_candidate_key(entity: Any) -> str | None:
    """Replacement key for the clean policy, or None when the span is unsafe."""
    text = entity.text.strip()
    if "\n" in entity.text or not text:
        return None  # NER span crossing a title/sentence boundary
    if re.search(r"(?:['’]s|['’])$", text, re.I):
        return None  # The replacement must not swallow an attached possessive marker.
    if entity.label_ == "ORDINAL" and not re.fullmatch(
            r"\d+(?:th|st|nd|rd)|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|"
            r"thirteenth|fourteenth|fifteenth|twentieth", text, re.I):
        return None  # "Firstly", "first of all", "second" as a unit of time
    if entity.label_ == "PERSON" and (text.isupper() or not _PERSON_NAME.fullmatch(text)):
        return None  # e.g. COVID-19 mislabelled as a person
    if entity.label_ == "PERSON" and len(entity) == 1 and entity.start == entity.sent.start:
        return None  # sentence-initial capital: "Lockdowns" is not a person
    if entity.label_ == "MONEY" and not re.search(r"\d", text):
        return None  # "millions of dollars" cannot be swapped for "$83 million"
    if entity.label_ == "MONEY" and entity.start and entity.doc[entity.start - 1].text in {"$", "€", "£", "¥"}:
        return None  # NER omitted an external currency symbol: avoid producing $$83 million.
    if entity.label_ == "GPE" and (text.isupper() or "." in text) and len(text) <= 6:
        return None  # abbreviation such as US/UK: replacement could be the same entity
    if entity.label_ == "CARDINAL" and any(t.text == "," for t in entity.doc[entity.end:entity.end + 5]):
        return None  # "two main characters, A and B": the count belongs to an enumerated list
    key = _replacement_key(entity)
    return key if key in CLEAN_ENTITY_KEYS else None


def _entity_substitution_clean(doc: Any, rng: random.Random) -> dict[str, Any] | None:
    """Entity swap restricted to unambiguous, type-compatible entities.

    Differences from ``_entity_substitution``: no ORG/LOC/TIME/relative dates, no
    spans containing a newline, and a multi-word PERSON also replaces standalone
    surname mentions with the replacement's surname, so the document does not keep
    "Looney said" after "Bernard Looney" became "Alex Morgan". Coreference beyond
    exact surname matching is still not resolved.
    """
    candidates, seen = [], set()
    # A surface string is eligible only if no NER instance fails the guards
    # (e.g. "two" before a comma list). Homographic auxiliaries, verbs and
    # lower-case common nouns are not aliases of a capitalized proper name.
    unsafe = {entity.text.casefold() for entity in doc.ents if _clean_candidate_key(entity) is None}
    for entity in doc.ents:
        key = _clean_candidate_key(entity)
        canonical = entity.text.casefold()
        if key is None or canonical in seen or canonical in unsafe:
            continue
        seen.add(canonical)
        matches = []
        partial_name = False
        for match in re.finditer(r"(?<!\w)" + re.escape(entity.text) + r"(?!\w)", doc.text, re.I):
            mention = doc.char_span(match.start(), match.end(), alignment_mode="expand")
            if mention is None:
                continue
            if key == "PERSON":
                for owner in doc.ents:
                    if not (owner.start_char <= match.start() and match.end() <= owner.end_char):
                        continue
                    bare_owner = re.sub(r"(?:['’]s|['’])$", "", owner.text, flags=re.I)
                    if (owner.label_ in _NON_PERSON_ENT_TYPES or
                            (owner.label_ == "PERSON" and bare_owner.casefold() != canonical)):
                        partial_name = True
                        break
                if partial_name:
                    break  # Do not turn Alison Cooper into Alison Alex Morgan.
            if key in {"DATE:month", "DATE:weekday"} and not any(
                    other.label_ == "DATE" and other.start_char <= match.start()
                    and match.end() <= other.end_char for other in doc.ents):
                continue  # Ambiguous month/day words need affirmative DATE evidence.
            if entity.label_ in {"PERSON", "GPE", "DATE"}:
                if any(t.pos_ in {"AUX", "VERB"} for t in mention):
                    continue  # May/may, Will/will, March/march are not equivalent mentions.
                if (any(c.isupper() for c in entity.text) and match.group().islower()
                        and any(t.pos_ != "PROPN" for t in mention)):
                    continue
            matches.append(match)
        if partial_name or not any(m.start() == entity.start_char and m.end() == entity.end_char for m in matches):
            continue
        replacements = [value for value in ENTITY_REPLACEMENTS[key]
                        if value.casefold() != canonical and value.casefold() not in doc.text.casefold()]
        if key == "PERSON" and len(entity.text.split()) >= 2:
            replacements = [value for value in replacements if not re.search(
                r"(?<!\w)" + re.escape(value.split()[-1]) + r"(?!\w)", doc.text, re.I)]
        if replacements:
            candidates.append((entity, key, replacements, matches))
    if not candidates:
        return None
    entity, key, replacements, matches = rng.choice(candidates)
    replacement = rng.choice(replacements)
    spans = [(m.start(), m.end(), replacement) for m in matches]
    surname_edits = 0
    parts, new_parts = entity.text.split(), replacement.split()
    if key == "PERSON" and len(parts) >= 2 and len(new_parts) >= 2 and len(parts[-1]) > 2:
        for m in re.finditer(r"(?<!\w)" + re.escape(parts[-1]) + r"(?!\w)", doc.text):
            if any(a <= m.start() < b for a, b, _ in spans):
                continue
            mention = doc.char_span(m.start(), m.end(), alignment_mode="expand")
            if mention is not None and any(t.ent_type_ in _NON_PERSON_ENT_TYPES for t in mention):
                continue  # e.g. the company "McDonald's" must not become "Chen's"
            if any(other.label_ == "PERSON" and other.start_char <= m.start() < other.end_char
                   and other.text.casefold() != parts[-1].casefold() for other in doc.ents):
                continue  # Jane Looney is not a surname-only mention of Bernard Looney.
            if mention is not None and any(t.pos_ in {"AUX", "VERB"} for t in mention):
                continue
            spans.append((m.start(), m.end(), new_parts[-1]))
            surname_edits += 1
    spans.sort()
    edits = [_edit(a, b, doc.text[a:b], new) for a, b, new in spans]
    sentence = entity.sent
    local_edits = [
        dict(edit, start_char=edit["start_char"] - sentence.start_char, end_char=edit["end_char"] - sentence.start_char)
        for edit in edits if sentence.start_char <= edit["start_char"] and edit["end_char"] <= sentence.end_char
    ]
    if not edits or not local_edits:
        return None
    sentence_index = next(index for index, span in enumerate(doc.sents) if span.start == sentence.start)
    return {
        "modified": _replace_edits(doc.text, edits),
        "target_original": sentence.text,
        "target_modified": _replace_edits(sentence.text, local_edits),
        "metadata": {
            "entity_type": entity.label_, "replacement_dictionary_key": key,
            "source_entity": entity.text, "replacement_entity": replacement,
            "edits": edits, "surface_occurrences_replaced": len(edits),
            "surname_occurrences_replaced": surname_edits,
            "sentence_index": sentence_index, "target_start_char": sentence.start_char,
            "target_end_char": sentence.end_char,
            "entity_policy": "clean_v2",
            "guard_revision": "review_fixes_3",
            "consistent_rename": key == "PERSON",
            "semantic_interpretation": "surface_referent_change_not_verified_contradiction",
            "coreference_resolved": False,
        },
    }


def _sentence_shuffle(doc: Any, rng: random.Random) -> dict[str, Any] | None:
    sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
    if len(sentences) < 2 or len(set(sentences)) < 2:
        return None
    permutation = list(range(len(sentences)))
    for _ in range(10):
        rng.shuffle(permutation)
        if [sentences[index] for index in permutation] != sentences:
            break
    else:
        permutation = list(range(1, len(sentences))) + [0]
    return {
        "modified": "\n".join(sentences[index] for index in permutation),
        "target_original": None, "target_modified": None,
        "metadata": {
            "permutation": permutation, "original_sentence_count": len(sentences),
            "moved_sentence_count": sum(index != original for index, original in enumerate(permutation)),
            "sentence_text_multiset_preserved": True,
            "sentence_separator_normalization": "stripped_sentence_texts_joined_with_newline",
            "semantic_interpretation": "order_change_often_proposition_preserving",
        },
    }


def _truncate(doc: Any, retained_fraction: float = 0.7) -> dict[str, Any] | None:
    sentences = list(doc.sents)
    if len(sentences) < 2:
        return None
    keep = max(1, min(len(sentences) - 1, math.floor(len(sentences) * retained_fraction)))
    cut = sentences[keep - 1].end_char
    return {
        "modified": doc.text[:cut], "target_original": None, "target_modified": None,
        "metadata": {
            "edits": [_edit(cut, len(doc.text), doc.text[cut:], "")],
            "retained_sentence_count": keep, "original_sentence_count": len(sentences),
            "removed_sentence_indices": list(range(keep, len(sentences))),
            "requested_retained_sentence_fraction": retained_fraction,
            "semantic_interpretation": "tail_information_loss_not_necessarily_contradiction",
        },
    }


def _random_deletion(doc: Any, rng: random.Random, deletion_fraction: float = 0.15) -> dict[str, Any] | None:
    words = [token for token in doc if token.is_alpha]
    if len(words) < 2:
        return None
    n_delete = max(1, min(len(words) - 1, round(len(words) * deletion_fraction)))
    selected = sorted(rng.sample(words, n_delete), key=lambda token: token.i)
    edits = [_edit(token.idx, token.idx + len(token.text), token.text, "", token_index=token.i) for token in selected]
    return {
        "modified": _replace_edits(doc.text, edits),
        "target_original": None, "target_modified": None,
        "metadata": {
            "edits": edits, "deleted_lexical_token_count": n_delete,
            "requested_lexical_deletion_fraction": deletion_fraction,
            "actual_lexical_deletion_fraction": n_delete / len(words),
            "semantic_interpretation": "information_loss_and_grammatical_damage_not_uniform_contradiction",
        },
    }


def generate_perturbation_pairs(
    records: Iterable[dict[str, Any]], seed: int = 42, nlp: Any = None,
    batch_size: int = 16, conditions: Iterable[str] = PERTURBATIONS,
) -> list[dict[str, Any]]:
    """Generate all six conditions per source, including explicit skipped rows.

    Filter ``status == 'applied'`` before metric computation. Skipped rows have
    ``modified=None`` and a reason, rather than a fake identity transformation.
    Negation/entity rows expose the edited sentence for local challenges as well
    as whole-document pairs. Each transformation uses its own SHA-seeded RNG.
    """
    records = list(records)
    conditions = tuple(conditions)
    allowed = set(PERTURBATIONS) | {"entity_substitution_clean"}
    unknown = set(conditions) - allowed
    if unknown:
        raise ValueError(f"Unknown perturbation conditions: {sorted(unknown)}")
    nlp = load_perturbation_nlp() if nlp is None else nlp
    pairs = []
    for record, doc in zip(records, nlp.pipe((record["text"] for record in records), batch_size=batch_size)):
        text = record["text"]
        source_id = _source_identifier(record)
        original_words = sum(token.is_alpha for token in doc)
        for condition in conditions:
            base = {
                "source_id": source_id, "domain": str(record.get("domain", "unknown")),
                "generator": str(record.get("generator", "unknown")),
                "perturbation": condition, "original": text,
                "source_group_id": record.get("group_id", record.get("pair_id")),
                "pair_id": f"{source_id}:{condition}:{_stable_hex(seed, source_id, text, condition)[:16]}",
            }
            random_state = _rng(seed, source_id, text, condition)
            if condition == "identity":
                change = {"modified": text, "target_original": None, "target_modified": None,
                          "metadata": {"edits": [], "semantic_interpretation": "exact_identity"}}
            elif condition == "negation":
                change = _negation(doc, random_state)
            elif condition == "entity_substitution":
                change = _entity_substitution(doc, random_state)
            elif condition == "entity_substitution_clean":
                change = _entity_substitution_clean(doc, random_state)
            elif condition == "sentence_shuffle":
                change = _sentence_shuffle(doc, random_state)
            elif condition == "truncate":
                change = _truncate(doc)
            else:
                change = _random_deletion(doc, random_state)

            seed_hex = _stable_hex(seed, source_id, text, condition)
            if change is None or (condition != "identity" and change["modified"] == text):
                clean_metadata = ({"entity_policy": "clean_v2", "guard_revision": "review_fixes_3"}
                                  if condition == "entity_substitution_clean" else {})
                pairs.append({**base, "status": "skipped", "modified": None,
                              "target_original": None, "target_modified": None,
                              "metadata": {"reason": "no_applicable_change", "rng_sha256": seed_hex,
                                           "seed": seed, "original_lexical_token_count": original_words,
                                           "automatic_semantic_uncertainty": True,
                                           "semantic_label_verified": False, **clean_metadata}})
                continue
            metadata = change["metadata"]
            modified_words = sum(token.is_alpha for token in nlp.make_doc(change["modified"]))
            metadata.update({
                "seed": seed, "rng_sha256": seed_hex,
                "original_lexical_token_count": original_words,
                "modified_lexical_token_count": modified_words,
                "lexical_token_count_delta": modified_words - original_words,
                "character_count_delta": len(change["modified"]) - len(text),
                "original_sha256": sha256(text.encode()).hexdigest(),
                "modified_sha256": sha256(change["modified"].encode()).hexdigest(),
                "automatic_semantic_uncertainty": condition != "identity",
                "semantic_label_verified": condition == "identity",
            })
            pairs.append({**base, **change, "status": "applied"})
    return pairs


def sanity_fixtures() -> list[dict[str, Any]]:
    """Small manual relation checks, separate from unlabeled document challenges.

    Expected directions describe premise→hypothesis for original→modified and
    the reverse. They assume ordinary literal readings of a single referent.
    They are sanity checks, not an estimate of model accuracy on the corpus.
    """
    fixtures = [
        ("identity", "The committee approved the proposal.", "The committee approved the proposal.", "entailment", "entailment"),
        ("paraphrase", "The committee approved the proposal.", "The proposal was approved by the committee.", "entailment", "entailment"),
        ("negation", "The company reported a profit.", "The company did not report a profit.", "contradiction", "contradiction"),
        ("entity_substitution", "The meeting started at 9:00 a.m.", "The meeting started at 3:00 p.m.", "contradiction", "contradiction"),
        ("sentence_shuffle", "The library opened on Monday. The museum opened on Tuesday.", "The museum opened on Tuesday. The library opened on Monday.", "entailment", "entailment"),
        ("truncate", "The library opened on Monday. The museum opened on Tuesday.", "The library opened on Monday.", "entailment", "neutral"),
        ("unrelated", "The company reported a profit.", "A blue bicycle stood beside the library.", "neutral", "neutral"),
    ]
    return [
        {
            "source_id": f"sanity:{name}", "domain": "sanity", "generator": "manual",
            "pair_id": f"sanity:{name}",
            "perturbation": name, "status": "applied", "original": original,
            "modified": modified, "target_original": original, "target_modified": modified,
            "metadata": {"manual_fixture": True, "expected_forward": forward,
                         "expected_reverse": reverse, "semantic_interpretation": "manually_specified_relation"},
        }
        for name, original, modified, forward, reverse in fixtures
    ]
