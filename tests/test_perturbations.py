from collections import Counter

import pytest

from humanization.semantics.perturbations import (
    PERTURBATIONS,
    _entity_substitution,
    _negation,
    _replace_edits,
    generate_perturbation_pairs,
    load_perturbation_nlp,
    sanity_fixtures,
    select_applicable_records,
    select_base_records,
)


@pytest.fixture(scope="module")
def nlp():
    return load_perturbation_nlp()


def record(text, source_id="example", domain="Essay", generator="test"):
    return {"id": source_id, "text": text, "domain": domain, "generator": generator, "label": 1}


def test_balanced_sampling_is_input_order_independent_and_filters_humans():
    records = [record(f"Text {domain} {generator} {i}", f"{domain}:{generator}:{i}", domain, generator)
               for domain in ("Essay", "Reuters", "WritingPrompts")
               for generator in ("a", "b", "c", "d", "e", "f", "g") for i in range(12)]
    records.append(dict(records[0], id="duplicate"))
    records.append(dict(records[1], id="human", label=0))
    selected = select_base_records(records, 200, 42)
    assert selected == select_base_records(reversed(records), 200, 42)
    assert len(selected) == len({item["text"] for item in selected}) == 200
    counts = Counter(item["domain"] for item in selected)
    assert max(counts.values()) - min(counts.values()) <= 1
    generator_counts = Counter(item["generator"] for item in selected)
    assert max(generator_counts.values()) - min(generator_counts.values()) <= 1
    for domain in counts:
        cell_counts = Counter(item["generator"] for item in selected if item["domain"] == domain)
        assert max(cell_counts.values()) - min(cell_counts.values()) <= 1


@pytest.mark.parametrize(("original", "expected"), [
    ("The company announced a large profit yesterday.", "The company did not announce a large profit yesterday."),
    ("The small dog likes the nearby park.", "The small dog does not like the nearby park."),
    ("The committee is meeting in London today.", "The committee is not meeting in London today."),
    ("The library was closed during the afternoon.", "The library was not closed during the afternoon."),
])
def test_negation_grammar_exact_edit_and_target(original, expected, nlp):
    from random import Random
    change = _negation(nlp(original), Random(42))
    assert change is not None
    assert change["modified"] == expected
    assert change["target_original"] == original
    assert change["target_modified"] == expected
    assert _replace_edits(original, change["metadata"]["edits"]) == expected


def test_negation_skips_negative_question_modal_and_embedded_clauses(nlp):
    from random import Random
    for text in ("The company did not announce a profit yesterday.",
                 "Did the company announce a profit yesterday?",
                 "The company might announce a profit tomorrow.",
                 "The manager believes the company reported a profit.",
                 "Some people visited the library yesterday.",
                 "Every member attended the meeting on Tuesday."):
        assert _negation(nlp(text), Random(42)) is None


def test_entity_changes_are_global_consistent_and_preserve_others(nlp):
    text = "Microsoft opened an office in London. Microsoft hired a team in London."
    pairs = generate_perturbation_pairs([record(text)], nlp=nlp)
    pair = next(item for item in pairs if item["perturbation"] == "entity_substitution")
    assert pair["status"] == "applied"
    entity = pair["metadata"]["source_entity"]
    replacement = pair["metadata"]["replacement_entity"]
    assert entity != replacement
    assert pair["modified"] == text.replace(entity, replacement)
    assert entity not in pair["modified"]
    assert pair["metadata"]["surface_occurrences_replaced"] == 2
    assert pair["target_original"] != pair["target_modified"]


def test_all_conditions_deterministic_and_severity_auditable(nlp):
    source = record("The company announced a large profit yesterday. "
                    "Microsoft hired seven engineers in London. "
                    "The library opened during the afternoon. "
                    "The museum welcomed many visitors on Tuesday.")
    pairs = generate_perturbation_pairs([source], seed=123, nlp=nlp)
    assert pairs == generate_perturbation_pairs([source], seed=123, nlp=nlp)
    assert tuple(pair["perturbation"] for pair in pairs) == PERTURBATIONS
    by_name = {pair["perturbation"]: pair for pair in pairs}
    assert all(pair["status"] == "applied" for pair in pairs)
    assert by_name["identity"]["modified"] == source["text"]
    shuffled = by_name["sentence_shuffle"]
    original_sentences = [sentence.text.strip() for sentence in nlp(source["text"]).sents]
    assert sorted(shuffled["modified"].split("\n")) == sorted(original_sentences)
    assert shuffled["metadata"]["moved_sentence_count"] >= 2
    truncated = by_name["truncate"]
    assert source["text"].startswith(truncated["modified"])
    assert truncated["metadata"]["retained_sentence_count"] == 2
    deleted = by_name["random_deletion"]
    assert _replace_edits(source["text"], deleted["metadata"]["edits"]) == deleted["modified"]
    assert deleted["metadata"]["lexical_token_count_delta"] == -deleted["metadata"]["deleted_lexical_token_count"]


def test_entity_replacements_respect_external_articles(nlp):
    from random import Random
    from spacy.tokens import Span
    doc = nlp("The company opened an office in the United States.")
    united = next(token.i for token in doc if token.text == "United")
    doc.ents = (Span(doc, united, united + 2, label="GPE"),)
    change = _entity_substitution(doc, Random(42))
    assert change is not None
    assert change["metadata"]["replacement_dictionary_key"] == "GPE:article"
    assert "the the " not in change["modified"]
    assert all(country not in change["modified"] for country in ("the Canada", "the Japan", "the Brazil"))
    doc = nlp("The company hired an IBM engineer.")
    ibm = next(token.i for token in doc if token.text == "IBM")
    doc.ents = (Span(doc, ibm, ibm + 1, label="ORG"),)
    assert _entity_substitution(doc, Random(42)) is None


def test_inapplicable_rows_are_skips_not_identity_fakes(nlp):
    pairs = generate_perturbation_pairs([record("Hello.")], nlp=nlp)
    assert pairs[0]["status"] == "applied"
    for pair in pairs[1:]:
        assert pair["status"] == "skipped"
        assert pair["modified"] is None
        assert pair["metadata"]["reason"] == "no_applicable_change"


def test_sanity_relations_distinguish_information_loss_from_contradiction():
    fixtures = {pair["perturbation"]: pair for pair in sanity_fixtures()}
    assert fixtures["negation"]["metadata"]["expected_forward"] == "contradiction"
    assert fixtures["negation"]["metadata"]["expected_reverse"] == "contradiction"
    assert fixtures["truncate"]["metadata"]["expected_forward"] == "entailment"
    assert fixtures["truncate"]["metadata"]["expected_reverse"] == "neutral"
    assert fixtures["sentence_shuffle"]["metadata"]["expected_forward"] == "entailment"


def test_eligibility_screening_is_explicit_and_keeps_only_actual_changes(nlp):
    sources = [record(f"The company announced {i + 2} large profits yesterday. "
                      "Microsoft opened an office in London.",
                      source_id=f"{domain}:{generator}:{i}", domain=domain, generator=generator)
               for domain in ("Essay", "Reuters", "WritingPrompts")
               for generator in ("a", "b") for i in range(3)]
    # Ogni fonte richiede un testo unico per il campionamento; aggiunge una frase stabile e innocua.
    for index, source in enumerate(sources):
        source["text"] += f" The reference code was {index + 100}."
    sources.append(record("Hello.", source_id="inapplicable"))
    selected, manifest = select_applicable_records(sources, n=12, seed=42, nlp=nlp)
    assert len(selected) == 12
    assert manifest["target_count_met"]
    assert manifest["unique_machine_candidate_count"] == 19
    assert manifest["eligible_count"] == 18
    assert manifest["excluded_count"] == 1
    assert all(pair["status"] == "applied" for pair in generate_perturbation_pairs(selected, nlp=nlp))
    selected, manifest = select_applicable_records(sources, n=20, seed=42, nlp=nlp)
    assert len(selected) == 18
    assert not manifest["target_count_met"]


def test_clean_entity_policy_drops_unsafe_categories_and_fixes_surnames(nlp):
    from random import Random
    from spacy.tokens import Span
    from humanization.semantics.perturbations import _entity_substitution_clean

    # ORG e TIME non fanno parte della politica clean.
    for text, label, tokens in [("The firm hired a Toyota engineer.", "ORG", ("Toyota",)),
                                ("The sun rose at 9:00 a.m. today.", "TIME", ("9:00", "a.m."))]:
        doc = nlp(text)
        start = next(t.i for t in doc if t.text == tokens[0])
        doc.ents = (Span(doc, start, start + len(tokens), label=label),)
        assert _entity_substitution_clean(doc, Random(42)) is None

    # Un segmento che attraversa un'interruzione di riga viene rifiutato.
    doc = nlp("Strike by Workers\n\nContract talks have stalled.")
    end = next(t.i for t in doc if t.text == "Contract") + 1
    doc.ents = (Span(doc, 0, end, label="PERSON"),)
    assert "\n" in doc.ents[0].text
    assert _entity_substitution_clean(doc, Random(42)) is None

    # Il nome completo e il cognome isolato vengono rinominati insieme.
    doc = nlp('CEO Bernard Looney spoke. "We did well," Looney said.')
    start = next(t.i for t in doc if t.text == "Bernard")
    doc.ents = (Span(doc, start, start + 2, label="PERSON"),)
    change = _entity_substitution_clean(doc, Random(42))
    assert change is not None
    assert "Looney" not in change["modified"]
    assert change["metadata"]["surname_occurrences_replaced"] == 1
    assert change["metadata"]["entity_policy"] == "clean_v2"


def test_clean_entity_policy_guards(nlp):
    from random import Random
    from spacy.tokens import Span
    from humanization.semantics.perturbations import _entity_substitution_clean

    def apply(text, label, first, length=1):
        doc = nlp(text)
        start = next(t.i for t in doc if t.text.startswith(first))
        doc.ents = (Span(doc, start, start + length, label=label),)
        return _entity_substitution_clean(doc, Random(42))

    covid = nlp("The COVID-19 pandemic spread quickly.")
    covid.ents = (Span(covid, 1, 1 + len(nlp.make_doc("COVID-19")), label="PERSON"),)
    assert _entity_substitution_clean(covid, Random(42)) is None                          # etichettato erroneamente
    assert apply("Firstly, we should act now.", "ORDINAL", "Firstly") is None            # avverbio
    assert apply("Exports to the US grew in 2020.", "GPE", "US") is None                 # abbreviazione
    assert apply("The two main characters, Ann and Bo, differ.", "CARDINAL", "two") is None  # elenco enumerato
    change = apply("The office is on the 4th floor.", "ORDINAL", "4th")
    assert change is not None and "4th" not in change["modified"]


def test_clean_entity_policy_more_guards(nlp):
    from random import Random
    from spacy.tokens import Span
    from humanization.semantics.perturbations import _entity_substitution_clean

    doc = nlp("Lockdowns reduced travel.")
    doc.ents = (Span(doc, 0, 1, label="PERSON"),)
    assert _entity_substitution_clean(doc, Random(42)) is None            # nome comune all'inizio della frase
    doc = nlp("The bank paid millions of dollars in fines.")
    start = next(t.i for t in doc if t.text == "millions")
    doc.ents = (Span(doc, start, start + 3, label="MONEY"),)
    assert _entity_substitution_clean(doc, Random(42)) is None            # nessuna cifra
    doc = nlp("The two men met. The two main characters, Ann and Bo, argued.")
    firsts = [t.i for t in doc if t.text == "two"]
    doc.ents = tuple(Span(doc, i, i + 1, label="CARDINAL") for i in firsts)
    assert _entity_substitution_clean(doc, Random(42)) is None            # un'occorrenza è un elenco
