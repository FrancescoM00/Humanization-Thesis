import math
import random

import pytest
import spacy
from spacy.tokens import Doc

from humanization.stylometry.features import (
    FEATURE_SCHEMA,
    POS_TAGS,
    extract_features,
    mattr,
)


def test_mattr_matches_naive_over_many_windows():
    rng = random.Random(123)
    for n in (1, 3, 50, 51, 300):
        tokens = [rng.choice(("a", "b", "c", "d", "e")) for _ in range(n)]
        for window in (1, 3, 50):
            value = mattr(tokens, window)
            if n < window:
                assert math.isnan(value)
            else:
                expected = sum(len(set(tokens[i:i + window])) / window for i in range(n - window + 1)) / (n - window + 1)
                assert value == pytest.approx(expected)
    with pytest.raises(ValueError):
        mattr(["a"], 0)


def make_doc(words, pos, sent_starts):
    return Doc(spacy.blank("en").vocab, words=words, pos=pos, sent_starts=sent_starts)


def test_feature_denominators_sentence_variation_and_schema():
    # Four alpha words, one number, four punctuation tokens; sentences 2 and 2.
    doc = make_doc(
        ["The", "cat", ",", "42", ".", "It", "runs", "—", "."],
        ["DET", "NOUN", "PUNCT", "NUM", "PUNCT", "PRON", "VERB", "PUNCT", "PUNCT"],
        [True, False, False, False, False, True, False, False, False],
    )
    features = extract_features(doc)
    assert set(features) == set(FEATURE_SCHEMA)
    assert features["n_words"] == 4
    assert features["n_sentences"] == 2
    assert features["sentence_length_mean"] == 2
    assert features["sentence_length_std"] == 0
    assert features["pos_num"] == pytest.approx(1 / 5)
    assert sum(features[f"pos_{tag.lower()}"] for tag in POS_TAGS) == pytest.approx(1)
    assert features["punct_comma_per100"] == 25
    assert features["punct_em_dash_per100"] == 25
    assert features["punct_total_per100"] == 100
    assert features["fw_the"] == 0.25
    assert features["fw_it"] == 0.25
    assert features["fw_total"] == 0.5
    assert math.isnan(features["mattr_50"])


def test_population_sentence_std_and_casefolded_diversity():
    doc = make_doc(
        ["Cat", ".", "cat", "cat", "dog", "."],
        ["NOUN", "PUNCT", "NOUN", "NOUN", "NOUN", "PUNCT"],
        [True, False, True, False, False, False],
    )
    features = extract_features(doc)
    assert features["sentence_length_mean"] == 2
    assert features["sentence_length_std"] == 1
    assert features["sentence_length_cv"] == 0.5
    assert features["ttr"] == 0.5
    assert features["hapax_ratio"] == 0.25


def test_wikipedia_proxies_match_phrases_not_substrings():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    doc = nlp("In conclusion, pivotal moments are not only useful but also crucial. "
              "In conclusional prose, rather than delve, use a tapestry. "
              "\n## Heading\n**bold** — Let me know.")
    features = extract_features(doc)
    scale = 1000 / features["n_words"]
    assert features["wiki_conclusion_per1000"] == pytest.approx(scale)
    assert features["wiki_chat_per1000"] == pytest.approx(scale)
    assert features["wiki_ai_vocabulary_per1000"] == pytest.approx(4 * scale)
    assert features["wiki_negative_parallelism_per1000"] == pytest.approx(2 * scale)
    assert features["wiki_markdown_per1000"] == pytest.approx(2 * scale)


def test_empty_input_is_missing_instead_of_false_zero_rates():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    features = extract_features(nlp(""))
    assert features["n_words"] == 0
    assert features["n_sentences"] == 0
    assert math.isnan(features["fw_total"])
    assert math.isnan(features["sentence_length_mean"])
    assert math.isnan(features["mattr_50"])
