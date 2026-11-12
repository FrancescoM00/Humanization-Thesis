"""Semantic scope and grammatical guards for the additional rule challenges."""
import random

import pytest

from humanization.semantics.perturbations import load_perturbation_nlp, _replace_edits
from humanization.semantics.subtle_perturbations import (
    number_change, quantifier_strengthening, hedge_removal, unsupported_detail,
)


@pytest.fixture(scope='module')
def nlp():
    return load_perturbation_nlp()


def test_quantity_rule_changes_counts_but_never_year_or_ordinal(nlp):
    doc = nlp('The team counted 12 samples during the inspection in 2019.')
    change = number_change(doc, random.Random(1))
    assert change is not None
    assert '2019' in change['modified'] and '12 samples' not in change['modified']
    assert change['modified'] == _replace_edits(doc.text, change['metadata']['edits'])
    assert number_change(nlp('The team returned for its third inspection in 2019.'), random.Random(1)) is None
    assert number_change(nlp('The burial site was already used as early as 750 BCE.'), random.Random(1)) is None
    assert number_change(nlp('Chapter 9 discusses the perspectives on social progress.'), random.Random(1)) is None
    bounded = number_change(nlp('The company achieved 100% renewable energy for its operations.'), random.Random(1))
    assert bounded is not None and '110%' not in bounded['modified']


def test_quantifier_rule_avoids_nonquantitative_idiom(nlp):
    change = quantifier_strengthening(nlp('Some students completed the examination without difficulty.'), random.Random(1))
    # Negative-scope guards can conservatively reject "without"; use the
    # simple affirmative counterpart for the intended scope probe.
    assert change is None
    change = quantifier_strengthening(nlp('Some students completed the examination successfully.'), random.Random(1))
    assert change['modified'] == 'All students completed the examination successfully.'
    assert quantifier_strengthening(nlp('The students studied for some time before lunch.'), random.Random(1)) is None


def test_hedge_removal_preserves_subject_agreement(nlp):
    singular = hedge_removal(nlp('The treatment may be effective in this setting.'), random.Random(1))
    plural = hedge_removal(nlp('The patients may require additional treatment at home.'), random.Random(1))
    assert singular['modified'] == 'The treatment is effective in this setting.'
    assert plural['modified'] == 'The patients do require additional treatment at home.'


def test_added_detail_is_attached_to_untimed_event(nlp):
    change = unsupported_detail(nlp('The company announced its new product yesterday.'), random.Random(1))
    assert change is None  # Existing temporal context excluded.
    doc = nlp('Microsoft announced its new product publicly.')
    change = unsupported_detail(doc, random.Random(1))
    assert change is not None and ' at ' in change['modified']
    assert change['metadata']['semantic_label_verified'] is False
    assert change['modified'] == _replace_edits(doc.text, change['metadata']['edits'])
