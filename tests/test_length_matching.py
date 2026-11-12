"""Test degli invarianti dell'abbinamento, senza trattare le fixture come dati di ricerca."""
import numpy as np
import pandas as pd
import pytest
import spacy

from humanization.datasets.length_matching import _assign, length_balance, sentence_prefixes


@pytest.fixture(scope="module")
def tokenizer():
    return spacy.blank("en").make_doc


def test_prefixes_stop_only_at_sentence_end_and_count_lexical_words(tokenizer):
    text = "One small sentence. Another complete sentence! A trailing unfinished fragment"
    prefixes = sentence_prefixes(text, tokenizer, minimum=3, maximum=6)
    assert prefixes == {3: "One small sentence.", 6: "One small sentence. Another complete sentence!"}
    assert all(sum(t.is_alpha for t in tokenizer(text)) == count for count, text in prefixes.items())


def test_no_arbitrary_cut_or_abbreviation_boundary(tokenizer):
    assert sentence_prefixes("Dr. Smith gives the value 3.14 without an ending", tokenizer, 1, 20) == {}
    text = "The first sentence ends. Dr. Smith starts an unfinished observation"
    assert list(sentence_prefixes(text, tokenizer, 1, 20).values()) == ["The first sentence ends."]


def test_quoted_ending_retains_quote(tokenizer):
    text = 'She said, "This sentence is complete." Later unfinished'
    prefixes = sentence_prefixes(text, tokenizer, 1, 20)
    assert list(prefixes.values()) == ['She said, "This sentence is complete."']


def test_assignment_uses_source_once_and_obeys_generator_quota():
    candidates = [{"source_row": i, "matches": {g: (f"human {i}.", f"{g} machine {i}.", 150+i)
                                                  for g in ["A", "B"]}} for i in range(6)]
    selected = _assign(candidates, {"A": 2, "B": 2}, 150, np.random.default_rng(42), set())
    assert len({c["source_row"] for c, _ in selected}) == 4
    assert sorted(g for _, g in selected) == ["A", "A", "B", "B"]
    assert {c["source_row"] for c, _ in selected} == {0, 1, 2, 3}


def test_assignment_fails_instead_of_relaxing_exact_requirement():
    candidates = [{"source_row": i, "matches": {"A": (f"human {i}.", f"machine {i}.", 150)}} for i in range(3)]
    with pytest.raises(ValueError, match="cannot satisfy"):
        _assign(candidates, {"A": 1, "B": 1}, 150, np.random.default_rng(42), set())


def test_length_balance_is_zero_for_exact_pairs():
    records = []
    for i, n in enumerate([120, 130, 150, 180]):
        for label in [0, 1]:
            records.append({"pair_id": str(i), "domain": "Essay", "generator": "human" if label == 0 else "A",
                            "label": label, "n_words": n, "n_characters": n*5+label*10, "n_sentences": 5+label})
    balance = length_balance(pd.DataFrame(records))
    words = balance[balance.measure == "n_words"]
    assert words.standardized_mean_difference.eq(0).all()
    assert words.ks_statistic.eq(0).all()
    assert balance[balance.measure == "n_characters"].standardized_mean_difference.gt(0).all()
