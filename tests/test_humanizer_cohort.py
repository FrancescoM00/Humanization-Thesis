from humanization.datasets.cohort import select_cohorts


def test_panels_are_nested_balanced_and_source_disjoint():
    rows = [{'id': f'{d}-{g}-{i}', 'domain': d, 'generator': g,
             'source_group_id': f'{d}-{g}-{i}', 'human': 'H', 'machine': 'M'}
            for d in ['Essay', 'Reuters', 'WritingPrompts']
            for g in ['one', 'two'] for i in range(12)]
    panels = select_cohorts(rows, per_cell=2, reserve_per_cell=4, calibration_per_domain=5)
    test_ids = {}
    calibrations = []
    for name, panel in panels.items():
        test = [r for r in panel if r['split'] == 'test']
        cal = [r for r in panel if r['split'] == 'calibration']
        test_ids[name] = {r['id'] for r in test}
        calibrations.append({r['id'] for r in cal})
        assert not ({r['source_group_id'] for r in cal} & {r['source_group_id'] for r in test})
        assert len(cal) == 15
    assert test_ids['smoke'] <= test_ids['pilot'] <= test_ids['study']
    assert len(test_ids['smoke']) == 3
    assert len(test_ids['pilot']) == 12
    assert len(test_ids['study']) == 24
    assert calibrations[0] == calibrations[1] == calibrations[2]
    # Selection is independent of input ordering and arbitrary score fields.
    other = select_cohorts([dict(r, detector_score=1) for r in reversed(rows)],
                           per_cell=2, reserve_per_cell=4, calibration_per_domain=5)
    assert panels == {k: [{key: val for key, val in r.items() if key != 'detector_score'}
                          for r in v] for k, v in other.items()}
