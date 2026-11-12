"""Semantic metric plumbing: coverage, direction, model labels and caching.

Neural-model tests use deterministic fake logits/embeddings, not downloaded
checkpoints. They validate the protocol, not a model's semantic competence.
"""

import json
import re
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from semantics.metrics import (_cached_chunks, _hash, _nli_direction, _top_candidates, _resolve_device,
                               bert_scores, lexical_scores, nli_scores,
                               resolve_nli_labels, token_chunks)


class FakeTokenizer:
    model_max_length = 256
    sep_token_id = 1
    cls_token_id = 0

    def __init__(self):
        self.vocab = {}
        self.reverse = {}
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in text.split():
            if word not in self.vocab:
                index = len(self.vocab)+2
                self.vocab[word] = index
                self.reverse[index] = word
            ids.append(self.vocab[word])
        return [0]+ids+[1] if add_special_tokens else ids

    def decode(self, ids, **kwargs):
        return " ".join(self.reverse[i] for i in ids)

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def __call__(self, first, second, **kwargs):
        import torch
        self.calls.append(kwargs)
        ids = [self.encode(a, True)+self.encode(b) + [1] for a, b in zip(first, second)]
        n = max(map(len, ids))
        return {"input_ids": torch.tensor([row+[0]*(n-len(row)) for row in ids])}


class CoverageAndDirectionTests(unittest.TestCase):
    def test_explicit_cuda_request_fails_without_a_gpu_instead_of_using_cpu(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(_resolve_device('cpu'), 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'CUDA was requested but is unavailable'):
                _resolve_device('cuda')

    def test_cuda_index_must_exist(self):
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=1):
            self.assertEqual(_resolve_device('cuda:0'), 'cuda:0')
            with self.assertRaisesRegex(RuntimeError, 'device 1 is unavailable'):
                _resolve_device('cuda:1')

    def test_actual_fast_wordpiece_preserves_original_words(self):
        from transformers import BertTokenizerFast
        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'alpha', 'play', '##ing', 'café', 'beta', 'gamma']
        with tempfile.TemporaryDirectory() as path:
            vocab_path = Path(path) / 'vocab.txt'
            vocab_path.write_text('\n'.join(vocab) + '\n', encoding='utf-8')
            tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False, strip_accents=False)
            text = 'alpha playing café beta playing gamma'
            self.assertIn('##ing', tokenizer.tokenize(text))
            chunks = token_chunks(text, tokenizer, max_tokens=2)
            self.assertEqual(' '.join(chunks), text)
            self.assertTrue(all(len(tokenizer.encode(t, add_special_tokens=False)) <= 2 for t in chunks))
            self.assertNotIn('##', ' '.join(chunks))

    def test_actual_byte_bpe_preserves_unicode(self):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        from transformers import PreTrainedTokenizerFast
        text = 'naïve caffè 🙂 résumé déjà café naïve caffè 🙂'
        backend = Tokenizer(models.BPE(unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator([text], trainers.BpeTrainer(vocab_size=45, special_tokens=['[UNK]']))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]')
        chunks = token_chunks(text, tokenizer, max_tokens=5)
        self.assertEqual(re.sub(r'\s+', '', ''.join(chunks)), re.sub(r'\s+', '', text))
        self.assertNotIn('\ufffd', ''.join(chunks))
        self.assertTrue(all(len(tokenizer.encode(t, add_special_tokens=False)) <= 5 for t in chunks))

    def test_all_tokens_survive_chunking_beyond_model_limit(self):
        text = " ".join(f"word{i}" for i in range(1000))
        tokenizer = FakeTokenizer()
        chunks = token_chunks(text, tokenizer, max_tokens=79)
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(len(tokenizer.encode(t)) <= 79 for t in chunks))

    def test_nli_labels_are_resolved_from_checkpoint_in_nonstandard_order(self):
        config = SimpleNamespace(id2label={0: "NEUTRAL", 1: "contradiction", 2: "entailment"})
        self.assertEqual(resolve_nli_labels(config), {"neutral": 0, "contradiction": 1, "entailment": 2})
        with self.assertRaises(ValueError):
            resolve_nli_labels(SimpleNamespace(id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}))

    def test_omission_is_directional_and_not_forced_to_be_contradiction(self):
        labels = {"neutral": 0, "contradiction": 1, "entailment": 2}

        def predict(pairs):
            return np.array([[.05, .05, .9] if a == b else [.75, .05, .2] for a, b in pairs])

        forward = _nli_direction(["A", "B"], ["A"], [[0, 1]], FakeTokenizer(), predict, labels)
        reverse = _nli_direction(["A"], ["A", "B"], [[0], [0]], FakeTokenizer(), predict, labels)
        self.assertAlmostEqual(forward["mean"], .9)
        self.assertAlmostEqual(reverse["mean"], .55)
        self.assertAlmostEqual(reverse["min"], .2)
        self.assertAlmostEqual(reverse["contradiction_max"], .05)
        self.assertAlmostEqual(reverse["low_support_fraction"], .5)

    def test_retrieval_axes_and_stale_cache(self):
        forward, reverse = _top_candidates(np.array([[.9, .1], [.2, .8], [.3, .7]]), 1)
        self.assertEqual(forward, [[0], [1]])
        self.assertEqual(reverse, [[0], [1], [1]])
        pair = {"id": "1", "original": "A", "modified": "B"}
        cache = {"1": {"original_sha256": _hash("A"), "modified_sha256": _hash("other")}}
        with self.assertRaises(ValueError):
            _cached_chunks(pair, cache)

    def test_lexical_identity_and_high_overlap_false_friend(self):
        original = "Alice said the mayor approved the plan for a new city school."
        modified = "Alice said the mayor did not approve the plan for a new city school."
        scores = lexical_scores([{"id": "same", "original": original, "modified": original},
                                 {"id": "negated", "original": original, "modified": modified}])
        self.assertAlmostEqual(scores.iloc[0].lexical_bleu, 1)
        self.assertAlmostEqual(scores.iloc[0].lexical_rouge_l_f1, 1)
        self.assertGreater(scores.iloc[1].lexical_rouge_l_f1, .8)


class NeuralPlumbingTests(unittest.TestCase):
    def test_bert_token_embeddings_deduplicate_and_identity_is_actually_scored(self):
        import torch
        tokenizer = FakeTokenizer()
        model = SimpleNamespace(_tokenizer=tokenizer, _model=object())
        encoded_texts = []

        def embeddings(texts, *args, **kwargs):
            encoded_texts.extend(texts)
            ids = [tokenizer.encode(text, True) for text in texts]
            length = max(map(len, ids))
            vectors = torch.zeros((len(texts), length, 128))
            masks, weights = torch.zeros((len(texts), length)), torch.zeros((len(texts), length))
            for i, row in enumerate(ids):
                for j, token in enumerate(row):
                    vectors[i, j, token] = 1
                masks[i, :len(row)] = 1
                weights[i, 1:len(row)-1] = 1
            return vectors, masks, weights

        a, b = "Alice visited Rome.", "Alice visited Paris."
        pairs = [{"id": "same", "original": a, "modified": a, "target_original": a, "target_modified": a},
                 {"id": "changed", "original": a, "modified": b, "target_original": a, "target_modified": b}]
        with tempfile.TemporaryDirectory() as path, patch("bert_score.BERTScorer", return_value=model), \
                patch("bert_score.utils.get_bert_embedding", side_effect=embeddings):
            scores = bert_scores(pairs, path)
        self.assertCountEqual(encoded_texts, [a, b])
        self.assertAlmostEqual(scores.iloc[0].local_bert_f1, 1)
        self.assertLess(scores.iloc[1].local_bert_f1, 1)
        self.assertEqual(scores.iloc[1].local_bert_protocol, "standard_sentence")

    def test_nli_covers_long_pairs_and_reuses_persistent_model_probabilities(self):
        import torch
        tokenizer = FakeTokenizer()

        class FakeModel:
            config = SimpleNamespace(id2label={0: "neutral", 1: "contradiction", 2: "entailment"},
                                     max_position_embeddings=512)

            def __init__(self):
                self.calls = 0

            def to(self, *args):
                return self

            def eval(self):
                return self

            def __call__(self, input_ids):
                self.calls += 1
                self_outer.assertLessEqual(input_ids.shape[1], 256)
                return SimpleNamespace(logits=torch.log(torch.tensor([[.2, .7, .1]]*len(input_ids))))

        self_outer = self
        model = FakeModel()
        text = " ".join(f"word{i}" for i in range(400))+"."
        pairs = [{"id": "1", "original": text, "modified": text,
                  "target_original": text, "target_modified": text}]
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps({"id2label": model.config.id2label}))
            cache_path = Path(path, "probabilities.jsonl")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                    patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=model):
                scores = nli_scores(pairs, path, probability_cache_path=cache_path)
                self.assertGreater(model.calls, 0)
                self.assertAlmostEqual(scores.iloc[0].nli_forward_mean, .1, places=6)
                self.assertAlmostEqual(scores.iloc[0].nli_forward_contradiction_mean, .7, places=6)
                self.assertGreater(scores.iloc[0].nli_forward_n_hypothesis_chunks, 1)
                first_calls = model.calls
                repeated = nli_scores(pairs, path, probability_cache_path=cache_path)
                self.assertEqual(model.calls, first_calls)
                self.assertEqual(repeated.iloc[0].nli_new_pairs_scored, 0)
        self.assertTrue(all(call["truncation"] is False for call in tokenizer.calls))


class BertDevicePlacementTests(unittest.TestCase):
    def check_document_and_aligned_paths(self, device):
        """Match upstream: GPU token vectors, but CPU masks and IDF weights."""
        import torch
        from semantics.aligned_similarity import aligned_bert_scores

        tokenizer = FakeTokenizer()
        scorer = SimpleNamespace(_tokenizer=tokenizer,
                                 _model=SimpleNamespace(float=lambda: None))

        def embeddings(texts, *args, **kwargs):
            self.assertEqual(kwargs['device'], device)
            ids = [tokenizer.encode(text, True) for text in texts]
            length = max(map(len, ids))
            vectors = torch.zeros((len(texts), length, 128), device=device)
            masks = torch.zeros((len(texts), length), device='cpu')
            weights = torch.zeros((len(texts), length), device='cpu')
            for i, row in enumerate(ids):
                for j, token in enumerate(row):
                    vectors[i, j, token] = 1
                masks[i, :len(row)] = 1
                weights[i, 1:len(row)-1] = 1
            return vectors, masks, weights

        a, b = 'Alice visited Rome.', 'Alice visited Paris.'
        pairs = [{'id': 'same', 'original': a, 'modified': a},
                 {'id': 'changed', 'original': a, 'modified': b}]
        alignment = {p['id']: {
            'original_sha256': _hash(p['original']), 'modified_sha256': _hash(p['modified']),
            'original_chunks': [p['original']], 'modified_chunks': [p['modified']],
            'forward_candidates': [[0]], 'reverse_candidates': [[0]],
        } for p in pairs}
        with tempfile.TemporaryDirectory() as path, \
                patch('bert_score.BERTScorer', return_value=scorer), \
                patch('bert_score.utils.get_bert_embedding', side_effect=embeddings):
            document = bert_scores(pairs, path, device=device)
            aligned, _ = aligned_bert_scores(pairs, alignment, path, device=device)
        # Token-level expected scores are known; moving weights cannot change
        # the score definition or replace identity inference with a constant.
        np.testing.assert_allclose(document.bert_chunk_f1, [1, 2/3], atol=1e-6)
        np.testing.assert_allclose(aligned.aligned_bert_f1_bidirectional_min, [1, 2/3], atol=1e-6)

    def test_cpu_scores_retain_known_token_matching_values(self):
        self.check_document_and_aligned_paths('cpu')

    def test_cuda_embeddings_accept_cpu_idf_weights_in_both_paths(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Requires CUDA; CPU cannot reproduce a cross-device tensor error')
        self.check_document_and_aligned_paths('cuda:0')


if __name__ == "__main__":
    unittest.main()
