"""Percorsi reali dei profili di esecuzione e test di regressione del pacchetto autonomo."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from humanization.paths import project_root,resolve_path
from tools.build_delivery import build,delivery_files,validate_portability


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_published_path_is_resolved_once(tmp_path):
    relative=Path('results/published/02_stylometry/baseline')
    first=resolve_path(tmp_path,relative)
    assert first==tmp_path/relative
    assert resolve_path(tmp_path,first)==first


def test_duplicated_published_is_a_configuration_error(tmp_path):
    with pytest.raises(ValueError,match='duplicated'):
        resolve_path(tmp_path,'results/published/published/02_stylometry')


def test_execution_profiles_bind_actual_feature_table_and_parent():
    root=project_root();profiles=root/'configs/protocols'
    for name in ['shortlist','shortlist_robustness']:
        profile=json.loads((profiles/(name+'.json')).read_text())
        assert digest(resolve_path(root,profile['input_csv']))==profile['input_sha256']
    child=json.loads((profiles/'shortlist_robustness.json').read_text())
    parent=resolve_path(root,child['parent_results'])
    protocol=parent/child['parent_protocol_file']
    manifest=json.loads((parent/child['parent_manifest_file']).read_text())
    assert digest(protocol)==child['parent_protocol_sha256']==manifest['protocol_sha256']
    assert digest(parent/'predictions.npz')==child['parent_predictions_sha256']
    assert digest(parent/'split_membership.csv')==child['parent_splits_sha256']
    assert manifest['manifest_scope']=='portable_reference_descriptor'


def test_hedging_exclusions_and_csv_hashes_are_current_project_files():
    root=project_root();profile=json.loads((root/'configs/protocols/hedging.json').read_text())
    for name,expected in {**profile['raw_sha256'],**profile['exclusion_sha256']}.items():
        assert digest(resolve_path(root,name))==expected


def test_packaging_uses_an_allowlist_without_git(tmp_path):
    (tmp_path/'README.md').write_text('Standalone project')
    for name in ['src/humanization/example.py','configs/example.yaml',
                 'archive/legacy_experiments/original.md','scratch/private.txt',
                 'data/models/checkpoint.bin','results/new_runs/temporary.csv']:
        file=tmp_path/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('example')
    names={p.relative_to(tmp_path).as_posix() for p in delivery_files(tmp_path)}
    assert names=={'README.md','src/humanization/example.py','configs/example.yaml'}
    archive=build(tmp_path,tmp_path/'delivery.zip')
    with zipfile.ZipFile(archive) as result:
        assert 'Humanization/DELIVERY_MANIFEST.json' in result.namelist()
        assert all('/archive/' not in n for n in result.namelist())


def test_packaging_rejects_retired_repository_locations(tmp_path):
    file=tmp_path/'README.md'
    file.write_text('https://github.com/'+'FrancescoM00/'+'Humanization')
    with pytest.raises(ValueError,match='Non-portable'):
        validate_portability(tmp_path,[file])
