import numpy as np
import pytest

from humanization.stylometry.workflows.score_humanizer_style import digest, fit_scales, style_table, validate_inputs
from humanization.stylometry.features import FEATURE_SCHEMA, POS_TAGS


def cohort():
    return [dict(id=str(i), source_group_id=str(i), domain='Essay', generator='g',
                 split='calibration' if i < 2 else 'test', human=f'human {i}', machine=f'machine {i}')
            for i in range(3)]


def pair():
    return dict(id='p', source_id='2', source_group_id='2', domain='Essay', generator='g',
                humanizer='identity', original='machine 2', modified='machine 2', status='ok')


def test_identity_distance_and_scales_depend_only_on_calibration_humans():
    rows = cohort()
    features = {}
    for index, text in enumerate(['human 0', 'human 1', 'machine 2']):
        values = {name: float(index+1) for name in FEATURE_SCHEMA}
        values.update({'pos_'+tag.lower(): 1/len(POS_TAGS) for tag in POS_TAGS})
        features[digest(text)] = values
    calibration, pairs = validate_inputs(rows, [pair()])
    names, means, scales, _ = fit_scales(calibration, features)
    assert means['word_length_mean'] == pytest.approx(1.5)
    assert scales['word_length_mean'] == pytest.approx(np.sqrt(.5))
    # Un valore anomalo del test non può cambiare le scale di calibrazione o le distanze dell'identità.
    features[digest('machine 2')]['word_length_mean'] = 10000.
    scored = style_table(pairs, features, names, means, scales).iloc[0]
    assert scored.style_rms == 0
    assert scored.pos_js == 0
    assert scored.word_ratio == 1
    assert scored.delta_human_centroid_distance == 0
    assert np.isfinite(scored.rms_original_to_human_centroid)


def test_calibration_source_cannot_reappear_in_test():
    rows = cohort()
    rows[-1]['source_group_id'] = rows[0]['source_group_id']
    with pytest.raises(ValueError, match='overlaps calibration and test'):
        validate_inputs(rows, [pair()])
