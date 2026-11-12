"""Verifiche della validità degli output e della ripresa delle riscritture vincolata alla provenienza."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from humanization.rewriting.pipeline import (
    _load_model, _rewrite_one, dipper_prompt, generation_status, read_jsonl, run_humanizers,
)


@pytest.mark.parametrize('oversized_second', [False, True])
def test_dipper_three_sentence_context_keeps_prefix_and_records_incomplete_windows(monkeypatch, oversized_second):
    import torch
    import humanization.rewriting.pipeline as rewriting
    monkeypatch.setattr(__import__('humanization.rewriting.dipper',fromlist=['']), '_dipper_sentences', lambda text: ['A.', 'B.', 'C.', 'D.'])
    source = dict(id='a', source_group_id='g', domain='Essay', generator='Claude', machine='A. B. C. D.')
    system = dict(name='dipper_context3', kind='dipper', window_protocol='three_sentence_context_v2',
                  max_input_tokens=12, max_new_tokens=8, lexical_diversity=60, order_diversity=0)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9
        prompts = []

        def __call__(self, text, **kwargs):
            assert kwargs['truncation'] is False
            self.prompts.append(text)
            length = 13 if oversized_second and len(self.prompts) == 2 else 4
            return {'input_ids': torch.ones((1, length), dtype=torch.long)}

        def decode(self, tokens, **kwargs):
            return 'First rewrite.' if tokens[0] == 2 else 'Second rewrite.'

    class Model:
        config = SimpleNamespace(decoder_start_token_id=0)
        generation_config = SimpleNamespace(eos_token_id=9)
        calls = 0

        def generate(self, **kwargs):
            assert kwargs['do_sample'] is False and kwargs['max_new_tokens'] == 8
            self.calls += 1
            return torch.tensor([[0, self.calls+1, 9]])

    tokenizer, model = Tokenizer(), Model()
    row = _rewrite_one(source, system, 42, tokenizer, model, torch.device('cpu'))
    assert tokenizer.prompts == ['lexical = 40, order = 100 <sent> A. B. C. </sent>',
                                 'lexical = 40, order = 100 First rewrite. <sent> D. </sent>']
    assert row['metadata']['n_windows_expected'] == 2
    if oversized_second:
        assert row['status'] == 'input_too_long' and model.calls == 1
        assert row['modified'] == 'First rewrite.' and not row['metadata']['finished_eos']
        assert row['metadata']['failed_window_index'] == 1
    else:
        assert row['status'] == 'ok' and model.calls == 2
        assert row['modified'] == 'First rewrite. Second rewrite.'
        assert row['metadata']['n_windows_completed'] == 2 and row['metadata']['finished_eos']


def inputs(tmp_path):
    cohort = tmp_path / "cohort.jsonl"
    sources = [
        dict(id="test-a", source_id="a", source_group_id="g-a", domain="Essay",
             generator="ChatGPT", split="test", human="A human passage.",
             machine="A machine passage."),
        dict(id="cal-b", source_id="b", source_group_id="g-b", domain="Reuters",
             generator="Claude", split="calibration", human="Calibration human.",
             machine="Calibration machine."),
    ]
    cohort.write_text("".join(json.dumps(row) + "\n" for row in sources))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"seed": 42, "humanizers": [{"name": "identity", "kind": "identity"}]}))
    return cohort, config


def test_identity_is_exact_and_excludes_calibration_and_resumes_without_duplicates(tmp_path):
    cohort, config = inputs(tmp_path)
    output = tmp_path / "output"
    first = run_humanizers(cohort, config, output, device="cpu")
    assert first["expected_rewrites"] == 1
    assert first["status_counts"] == {"ok": 1}
    rows = read_jsonl(output / "rewrites.jsonl")
    assert rows[0]["original"] == rows[0]["modified"] == "A machine passage."
    assert rows[0]["source_id"] == "test-a"
    assert rows[0]["source_group_id"] == "g-a"
    assert rows == read_jsonl(output / "pairs.jsonl")
    before = (output / "rewrites.jsonl").read_bytes()
    second = run_humanizers(cohort, config, output, device="cpu")
    assert second["resumed_completed_rewrites"] == 1
    assert (output / "rewrites.jsonl").read_bytes() == before


def test_resume_refuses_changed_text_or_generation_configuration(tmp_path):
    cohort, config = inputs(tmp_path)
    output = tmp_path / "output"
    run_humanizers(cohort, config, output, device="cpu")
    original = config.read_text()
    config.write_text(original.replace("42", "43"))
    with pytest.raises(ValueError, match="Resume configuration/cohort/device differs"):
        run_humanizers(cohort, config, output, device="cpu")
    config.write_text(original)
    cohort.write_text(cohort.read_text().replace("machine passage", "different passage"))
    with pytest.raises(ValueError, match="Resume configuration/cohort/device differs"):
        run_humanizers(cohort, config, output, device="cpu")


def test_empty_and_token_limited_generations_are_not_successful_pairs():
    assert generation_status("", 1, 512, True) == "generation_error"
    assert generation_status("A partial passage", 512, 512, False) == "truncated"
    assert generation_status("A finished passage.", 512, 512, True) == "ok"
    assert generation_status("A finished passage.", 20, 512, True) == "ok"


@pytest.mark.parametrize('requested,current,expected', [('cuda', 1, 1), ('cuda:0', 1, 0)])
def test_cuda_alias_and_explicit_gpu_select_an_integer_without_downloading(
        monkeypatch, requested, current, expected):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    monkeypatch.delenv('HUMANIZATION_REUSE_CACHED_MODELS', raising=False)
    selected = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: current)

    def set_device(index):
        # Riproduce l'API rigorosa che rifiutava torch.device('cuda').
        assert isinstance(index, int)
        selected.append(index)

    monkeypatch.setattr(torch.cuda, 'set_device', set_device)
    tokenizer = object()
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: tokenizer)

    class Model:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    model = Model()

    def from_pretrained(*args, **kwargs):
        assert args == ('test/no-download',)
        assert kwargs['revision'] == '0'*40
        assert 'local_files_only' not in kwargs
        assert kwargs['torch_dtype'] == torch.float16
        return model

    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', from_pretrained)
    actual_tokenizer, actual_model, actual_device = _load_model(
        {'kind': 'causal', 'model_id': 'test/no-download', 'revision': '0'*40}, requested)
    assert selected == [expected]
    assert actual_device == torch.device('cuda', expected) == model.device
    assert actual_model is model and actual_tokenizer is tokenizer


def _checkpoint(root, repo_id, revision, *, complete=True):
    snapshot = root / ('models--' + repo_id.replace('/', '--')) / 'snapshots' / revision
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text('{}')
    (snapshot / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {'a': 'model-00001-of-00002.safetensors',
                       'b': 'model-00002-of-00002.safetensors'}}))
    (snapshot / 'model-00001-of-00002.safetensors').write_bytes(b'fake')
    if complete:
        (snapshot / 'model-00002-of-00002.safetensors').write_bytes(b'fake')
    return snapshot


def test_read_only_cache_roots_include_project_and_platform_temp(tmp_path, monkeypatch):
    import humanization.rewriting.loading as module
    project, platform_temp = tmp_path/'project', tmp_path/'system_temp'
    monkeypatch.setattr(module, 'project_root', lambda: project)
    monkeypatch.setattr(module.tempfile, 'gettempdir', lambda: str(platform_temp))
    roots = module._cached_hf_roots()
    assert project/'.cache/huggingface/hub' in roots
    assert platform_temp/'humanization/huggingface/hub' in roots
    assert not any('/desktop/' in str(root) for root in roots)
    assert not project.exists() and not platform_temp.exists()


@pytest.mark.parametrize('wrong_revision', [False, True])
def test_strict_cached_reuse_rejects_incomplete_or_unpinned_weights_before_cuda(
        tmp_path, monkeypatch, wrong_revision):
    import torch
    import humanization.rewriting.pipeline as rewriting
    from transformers import AutoModelForCausalLM, AutoTokenizer

    monkeypatch.setenv('HUMANIZATION_REUSE_CACHED_MODELS', '1')
    monkeypatch.setattr(__import__('humanization.rewriting.loading',fromlist=['']), '_cached_hf_roots', lambda: [tmp_path])
    _checkpoint(tmp_path, 'test/model', '1'*40 if wrong_revision else '0'*40,
                complete=wrong_revision)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: pytest.fail('Must not select CUDA'))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained',
                        lambda *a, **k: pytest.fail('Weights must be checked before the tokenizer'))
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained',
                        lambda *a, **k: pytest.fail('Must not download or allocate weights'))
    with pytest.raises(RuntimeError, match='no complete pinned checkpoint for test/model@' + '0'*40):
        _load_model({'kind': 'causal', 'model_id': 'test/model', 'revision': '0'*40}, 'cuda:0')


def test_strict_cached_reuse_rejects_missing_tokenizer_before_cuda(tmp_path, monkeypatch):
    import torch
    import humanization.rewriting.pipeline as rewriting
    from transformers import AutoModelForCausalLM, AutoTokenizer

    monkeypatch.setenv('HUMANIZATION_REUSE_CACHED_MODELS', '1')
    monkeypatch.setattr(__import__('humanization.rewriting.loading',fromlist=['']), '_cached_hf_roots', lambda: [tmp_path])
    _checkpoint(tmp_path, 'test/model', '0'*40)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: pytest.fail('Must not select CUDA'))
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained',
                        lambda *a, **k: pytest.fail('Must not allocate weights'))

    def tokenizer_missing(repo_id, **kwargs):
        assert repo_id == 'test/model'
        assert kwargs['revision'] == '0'*40 and kwargs['local_files_only'] is True
        assert kwargs['cache_dir'] == str(tmp_path)
        raise OSError('Tokenizer is not cached')

    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', tokenizer_missing)
    with pytest.raises(RuntimeError, match='tokenizer test/model@' + '0'*40):
        _load_model({'kind': 'causal', 'model_id': 'test/model', 'revision': '0'*40}, 'cuda:0')


@pytest.mark.parametrize('kind,device,quantized,free_gib', [
    ('causal', 'cpu', False, 10), ('causal', 'cuda:1', False, 10),
    ('causal', 'cuda:1', True, 10), ('dipper', 'cuda:1', True, 10),
    ('dipper', 'cuda:1', True, 7), ('dipper', 'cuda:1', False, 10),
])
def test_strict_cached_reuse_preserves_precision_and_checks_with_separate_tokenizer_cache(
        tmp_path, monkeypatch, kind, device, quantized, free_gib):
    import torch
    import transformers
    import humanization.rewriting.pipeline as rewriting

    model_root, tokenizer_root = tmp_path / 'weights', tmp_path / 'tokenizer'
    snapshot = _checkpoint(model_root, 'test/model', '0'*40)
    tokenizer_root.mkdir()
    monkeypatch.setenv('HUMANIZATION_REUSE_CACHED_MODELS', '1')
    monkeypatch.setattr(__import__('humanization.rewriting.loading',fromlist=['']), '_cached_hf_roots', lambda: [model_root, tokenizer_root])
    monkeypatch.setattr(rewriting.shutil, 'disk_usage',
                        lambda *a: pytest.fail('Complete checkpoints need no download space'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    selected = []
    monkeypatch.setattr(torch.cuda, 'set_device', selected.append)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *a: (free_gib * 2**30, 16 * 2**30))
    monkeypatch.setattr(transformers, 'BitsAndBytesConfig', lambda **kwargs: kwargs)
    tokenizer, tokenizer_calls, model_calls = object(), [], []
    tokenizer_class = transformers.T5Tokenizer if kind == 'dipper' else transformers.AutoTokenizer
    tokenizer_id = rewriting.DIPPER_TOKENIZER if kind == 'dipper' else 'test/model'
    tokenizer_revision = rewriting.DIPPER_TOKENIZER_REVISION if kind == 'dipper' else '0'*40

    def load_tokenizer(repo_id, **kwargs):
        assert repo_id == tokenizer_id and kwargs['revision'] == tokenizer_revision
        assert kwargs['local_files_only'] is True
        tokenizer_calls.append(kwargs['cache_dir'])
        if kwargs['cache_dir'] != str(tokenizer_root):
            raise OSError('Tokenizer lives in a different cache')
        return tokenizer

    monkeypatch.setattr(tokenizer_class, 'from_pretrained', load_tokenizer)

    class Model:
        moved_to = None

        def to(self, target):
            self.moved_to = target
            return self

        def eval(self):
            return self

    model = Model()
    def load_weights(location, **kwargs):
        model_calls.append((location, kwargs))
        return model

    model_class = transformers.T5ForConditionalGeneration if kind == 'dipper' else transformers.AutoModelForCausalLM
    monkeypatch.setattr(model_class, 'from_pretrained', load_weights)
    system = dict(kind=kind, model_id='test/model', revision='0'*40, load_in_4bit=quantized)
    environment = dict(os.environ)
    if kind == 'dipper' and (free_gib < 8 or not quantized):
        error = RuntimeError if free_gib < 8 else ValueError
        message = '8 GiB' if free_gib < 8 else 'load_in_4bit=true'
        with pytest.raises(error, match=message):
            _load_model(system, device)
        assert model_calls == []
    else:
        actual_tokenizer, actual_model, actual_device = _load_model(system, device)
        assert actual_tokenizer is tokenizer and actual_model is model
        assert actual_device == torch.device(device)
        location, kwargs = model_calls[0]
        assert location == str(snapshot)
        assert kwargs['local_files_only'] is True and kwargs['revision'] == '0'*40
        assert kwargs['trust_remote_code'] is False and kwargs['low_cpu_mem_usage'] is True
        assert kwargs['torch_dtype'] == (torch.float32 if device == 'cpu' else torch.float16)
        if quantized:
            assert model.moved_to is None and kwargs['device_map'] == {'': 1}
            assert kwargs['quantization_config'] == dict(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            assert 'quantization_config' not in kwargs and model.moved_to == actual_device
    assert selected == ([] if device == 'cpu' else [1])
    assert tokenizer_calls == [str(model_root), str(tokenizer_root)]
    assert dict(os.environ) == environment


def test_dipper_prompt_uses_inverse_codes_and_does_not_drop_text():
    assert dipper_prompt("First sentence.\nSecond sentence.", 60, 0) == (
        "lexical = 40, order = 100 <sent> First sentence. Second sentence. </sent>")
    with pytest.raises(ValueError, match="DIPPER diversity"):
        dipper_prompt("Passage.", 55, 0)


def test_oversized_prompt_is_rejected_before_generation_without_truncation():
    import torch
    source = dict(id="a", source_group_id="g", domain="Essay",
                  generator="Claude", machine="A whole input passage.")
    system = dict(name="rewrite", kind="causal", prompt="Rewrite carefully.",
                  max_input_tokens=2, max_new_tokens=8)

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "A full formatted prompt"

        def __call__(self, text, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    class Model:
        config = SimpleNamespace(max_position_embeddings=2048)

        def generate(self, **kwargs):
            pytest.fail("An oversized input must never reach generation")

    row = _rewrite_one(source, system, 42, Tokenizer(), Model(), torch.device("cpu"))
    assert row["status"] == "input_too_long"
    assert row["modified"] == ""
    assert row["metadata"]["input_tokens"] == 3


def test_causal_output_does_not_include_prompt_and_eos_at_limit_is_complete():
    import torch
    source = dict(id="a", source_group_id="g", domain="Essay",
                  generator="Claude", machine="An input passage.")
    system = dict(name="rewrite", kind="causal", prompt="Rewrite carefully.",
                  max_input_tokens=10, max_new_tokens=3)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def apply_chat_template(self, *args, **kwargs):
            return "A full formatted prompt"

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[11, 12]])}

        def decode(self, tokens, **kwargs):
            assert tokens == [22, 23, 9]
            return "A rewritten passage."

    class Model:
        config = SimpleNamespace(max_position_embeddings=2048)
        generation_config = SimpleNamespace(eos_token_id=[8, 9])

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.tensor([[11, 12, 22, 23, 9]])

    row = _rewrite_one(source, system, 42, Tokenizer(), Model(), torch.device("cpu"))
    assert row["modified"] == "A rewritten passage."
    assert row["status"] == "ok"
    assert row["metadata"]["output_tokens"] == 3
    assert row["metadata"]["finished_eos"] is True
