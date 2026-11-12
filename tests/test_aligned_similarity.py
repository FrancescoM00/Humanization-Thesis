import numpy as np
import pytest

from humanization.semantics.aligned_similarity import candidate_support, directional_row, matrix_support, summarize_support


def test_maxima_use_all_units_and_direction_is_correct():
    # The best evidence changes positions; positional alignment would fail.
    forward, reverse = matrix_support([[.2, .8, .1], [.9, .3, .4]])
    np.testing.assert_allclose(forward, [.9, .8, .4])
    np.testing.assert_allclose(reverse, [.8, .9])
    row = directional_row({'id':'x'}, forward, reverse, 'aligned')
    assert row['aligned_forward_min'] == .4
    assert row['aligned_reverse_min'] == .8
    assert row['aligned_bidirectional_min'] == .4
    assert row['aligned_bidirectional_mean'] == pytest.approx(.7)


def test_one_corrupt_unit_dilutes_mean_but_not_minimum():
    support = summarize_support([1.0]*19 + [0.0])
    assert support['min'] == 0
    assert support['mean'] == pytest.approx(.95)
    assert support['q10'] == 1


def test_retrieval_restriction_does_not_use_unretrieved_best_evidence():
    matrix = [[.3, .9], [.2, .1], [.1, .2], [.95, .8]]
    forward, reverse = candidate_support(matrix, [[0, 1, 2], [0, 1, 3]], [[0], [0], [0], [0]])
    np.testing.assert_allclose(forward, [.3, .9])
    np.testing.assert_allclose(reverse, [.3, .2, .1, .95])


def test_invalid_or_missing_units_are_not_silently_dropped():
    with pytest.raises(ValueError):
        summarize_support([1, np.nan])
    with pytest.raises(ValueError):
        matrix_support(np.empty((1, 0)))
