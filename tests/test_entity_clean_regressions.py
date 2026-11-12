"""Regressions found while reviewing the supplied clean-v2 patch."""
from random import Random

import pytest

from semantics.perturbations import (
    _entity_substitution_clean,
    generate_perturbation_pairs,
    load_perturbation_nlp,
)


@pytest.fixture(scope="module")
def nlp():
    return load_perturbation_nlp()


def set_entities(doc, items):
    spans = []
    for text, label in items:
        start = doc.text.index(text)
        spans.append(doc.char_span(start, start + len(text), label=label))
    assert all(span is not None for span in spans)
    doc.ents = spans


@pytest.mark.parametrize("text,month,unchanged", [
    ("The meeting is in May. They may attend.", "May", "They may attend."),
    ("The meeting is in May. May I attend?", "May", "May I attend?"),
    ("The meeting is in March. March toward the gate.", "March", "March toward the gate."),
])
def test_month_replacement_does_not_change_a_modal_or_verb(nlp, text, month, unchanged):
    doc = nlp(text)
    set_entities(doc, [(month, "DATE")])
    change = _entity_substitution_clean(doc, Random(42))
    assert change is not None
    assert change["target_modified"] != change["target_original"]
    assert change["modified"].endswith(unchanged)
    assert change["metadata"]["surface_occurrences_replaced"] == 1


def test_surname_inside_another_persons_full_name_is_preserved(nlp):
    doc = nlp("CEO Bernard Looney spoke. Jane Looney disagreed.")
    set_entities(doc, [("Bernard Looney", "PERSON"), ("Jane Looney", "PERSON")])
    change = _entity_substitution_clean(doc, Random(42))
    assert change is not None
    assert change["metadata"]["source_entity"] == "Bernard Looney"
    assert "Jane Looney disagreed." in change["modified"]
    assert change["metadata"]["surname_occurrences_replaced"] == 0


@pytest.mark.parametrize("full_name,short_name", [
    ("Alison Cooper", "Cooper"),
    ("Jack Ma", "Ma"),
    ("Bernard Looney", "Looney"),
    ("John Bernard Looney", "Bernard Looney"),
])
def test_short_name_cannot_expand_inside_another_full_name(nlp, full_name, short_name):
    doc = nlp(f"CEO {full_name} spoke. {short_name} replied.")
    first_start = doc.text.index(full_name)
    short_start = doc.text.rindex(short_name)
    doc.ents = [doc.char_span(first_start, first_start + len(full_name), label="PERSON"),
                doc.char_span(short_start, short_start + len(short_name), label="PERSON")]
    change = _entity_substitution_clean(doc, Random(0))
    if change is not None:
        for edit in change["metadata"]["edits"]:
            for owner in doc.ents:
                if owner.start_char <= edit["start_char"] and edit["end_char"] <= owner.end_char:
                    assert (owner.start_char, owner.end_char) == (edit["start_char"], edit["end_char"])


def test_person_name_cannot_expand_inside_an_organization_name(nlp):
    doc = nlp("John Deere announced its results. John replied.")
    first_start = doc.text.index("John Deere")
    second_start = doc.text.rindex("John")
    doc.ents = [doc.char_span(first_start, first_start + len("John Deere"), label="ORG"),
                doc.char_span(second_start, second_start + len("John"), label="PERSON")]
    assert _entity_substitution_clean(doc, Random(42)) is None


def test_replacement_surname_does_not_collide_with_an_existing_mention(nlp):
    doc = nlp("The CEO Bernard Looney spoke. Morgan disagreed.")
    set_entities(doc, [("Bernard Looney", "PERSON")])
    change = _entity_substitution_clean(doc, Random(42))
    assert change is not None
    assert change["metadata"]["replacement_entity"].split()[-1] != "Morgan"
    assert "Morgan disagreed." in change["modified"]


@pytest.mark.parametrize("text,entity,label", [
    ("Phineas Taylor Barnum's marketing strategies worked.", "Phineas Taylor Barnum's", "PERSON"),
    ("Lena’s head was tilted up.", "Lena’s", "PERSON"),
    ("Hong Kong's air cargo handling increased.", "Hong Kong's", "GPE"),
    ("The James’ house was empty.", "James’", "PERSON"),
])
def test_an_entity_span_cannot_swallow_the_possessive_marker(nlp, text, entity, label):
    doc = nlp(text)
    set_entities(doc, [(entity, label)])
    assert _entity_substitution_clean(doc, Random(42)) is None


@pytest.mark.parametrize("text,amount", [
    ("Its previous price was $69.99.", "69.99"),
    ("They bought $800 shoes.", "800"),
    ("The price was € 800.", "800"),
])
def test_money_span_cannot_leave_an_external_currency_symbol(nlp, text, amount):
    doc = nlp(text)
    set_entities(doc, [(amount, "MONEY")])
    assert _entity_substitution_clean(doc, Random(42)) is None


def test_conditions_iterator_is_reused_for_every_source(nlp):
    records = [{"id": name, "text": "The company opened its office.", "label": 1}
               for name in ["first", "second"]]
    pairs = generate_perturbation_pairs(records, nlp=nlp, conditions=iter(["identity"]))
    assert len(pairs) == 2
    assert [row["source_id"] for row in pairs] == ["first", "second"]


def test_unknown_condition_cannot_silently_produce_random_deletion():
    with pytest.raises(ValueError, match="Unknown perturbation conditions"):
        generate_perturbation_pairs([], conditions=["entity_substitution_celan"])
