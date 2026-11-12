"""Kaggle recovery paths are checked without a GPU, pip or network requests."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from notebooks import run_nli_calibration as launcher


def model(directory, *, revision=launcher.MODEL_REVISION):
    directory.mkdir(parents=True, exist_ok=True)
    names = ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors']
    for name in names:
        (directory / name).write_bytes(b'complete')
    manifest = dict(repository=launcher.MODEL_ID, revision=revision, kind='nli',
        files=[dict(name=name, bytes=len(b'complete')) for name in names])
    (directory / 'DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest))


def test_matching_readable_old_venv_is_reused_without_pip_or_venv_writes(tmp_path):
    old_runtime, working = tmp_path / 'old', tmp_path / 'working'
    python = old_runtime / 'venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('readable interpreter')
    python.chmod(0o444)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    result = launcher.choose_python(tmp_path / 'bundle', working, old_runtime, {}, runner=run)
    assert result == python
    assert len(calls) == 1 and calls[0][:2] == [str(python), '-c']
    assert not working.exists()
    assert python.read_text() == 'readable interpreter'


def test_model_revision_and_all_files_are_required_and_old_cache_is_read_only_input(tmp_path):
    working, old_runtime = tmp_path / 'working', tmp_path / 'old'
    old_model = old_runtime / 'models/nli'
    model(old_model)
    wrong = working / '.humanization_semantic_models/nli'
    model(wrong, revision='wrong')
    before = {p: p.read_bytes() for p in old_model.iterdir()}
    result = launcher.ensure_model(tmp_path / 'bundle', working, old_runtime, Path('python'), {},
                                   runner=lambda *a, **k: pytest.fail('No download of complete pinned weights'))
    assert result == old_model
    assert before == {p: p.read_bytes() for p in old_model.iterdir()}
    (old_model / 'model.safetensors').unlink()
    assert not launcher.complete_model(old_model)

    def download(command, **kwargs):
        assert command[command.index('--revision') + 1] == launcher.MODEL_REVISION
        destination = Path(command[command.index('--output') + 1]) / 'nli'
        assert destination == working / '.humanization_nli_calibration_models/nli'
        model(destination)

    result = launcher.ensure_model(tmp_path / 'bundle', working, old_runtime, Path('python'), {}, runner=download)
    assert result == working / '.humanization_nli_calibration_models/nli'
    assert json.loads((wrong / 'DOWNLOAD_MANIFEST.json').read_text())['revision'] == 'wrong'


def test_cached_bundle_is_frozen_and_mutated_input_is_not_overwritten(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    with zipfile.ZipFile(run / 'source_bundle.zip', 'w') as archive:
        for name in launcher.REQUIRED_BUNDLE_FILES:
            archive.writestr(name, 'original frozen content')
    bundle, digest = launcher.install_bundle(run)
    (run / 'runtime.json').write_text(json.dumps(dict(bundle_sha256=digest)))
    assert launcher.install_bundle(run)[0] == bundle
    (bundle / 'inputs/labels.csv').write_text('mutated label')
    with pytest.raises(RuntimeError, match='modificato'):
        launcher.install_bundle(run)
    assert (bundle / 'inputs/labels.csv').read_text() == 'mutated label'


def test_small_review_archive_retains_cache_on_disk_and_excludes_weights_outside_run(tmp_path):
    run = tmp_path / 'run'
    (run / '_tmp').mkdir(parents=True)
    (run / '_tmp/probe').write_text('temporary')
    (run / 'scores').mkdir()
    cache = run / 'scores/contextual_probability_cache.jsonl'
    cache.write_text('resume cache')
    (run / 'scores/contextual_nli_units.jsonl').write_text('detailed evidence')
    (run / 'scores/contextual_nli.csv').write_text('id,score\n1,0.9\n')
    (run / 'runtime.json').write_text('{}')
    outside = tmp_path / 'model.safetensors'
    outside.write_text('weights')
    archive = launcher.export_review(run)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {'scores/contextual_nli.csv', 'runtime.json'}
    assert cache.read_text() == 'resume cache'
    assert outside.read_text() == 'weights'
