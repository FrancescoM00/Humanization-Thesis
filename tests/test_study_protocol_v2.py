import json
import pytest
from humanizers.study_protocol import prepare_study_protocol, source_screen


def rows():
    return [
        dict(id='cal', source_group_id='cal', split='calibration'),
        dict(id='x', source_group_id='x', split='test', domain='Essay', generator='StableLM',
             machine='Write a one page paper in APA format.'),
        dict(id='y', source_group_id='y', split='test', domain='Essay', generator='Claude',
             machine='The weather changed during the night.'),
    ]


def test_instruction_flags_do_not_exclude_sources_or_create_labels():
    result = source_screen(rows())
    assert len(result) == 2 and all(r['include_primary'] for r in result)
    assert result[0]['input_form_candidate'] == 'review_instruction_or_mixed'
    assert result[1]['input_form_candidate'] == 'not_flagged'
    assert all(r['input_form_manual_prose_instructions_mixed'] == '' for r in result)


def test_group_overlap_is_rejected():
    data = rows(); data[1]['source_group_id'] = 'cal'
    with pytest.raises(ValueError, match='disjoint'):
        source_screen(data)


def test_changed_panel_cannot_be_relabelled_as_frozen_study(tmp_path):
    cohort = tmp_path/'cohort.jsonl'
    cohort.write_text(''.join(json.dumps(r)+'\n' for r in rows()))
    with pytest.raises(ValueError, match='previously frozen'):
        prepare_study_protocol(cohort, tmp_path/'output')
    assert not (tmp_path/'output').exists()
