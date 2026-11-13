from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from humanization.detection.pipeline import validate_inputs
from humanization.rewriting.pipeline import pair_id


PROTOCOL = 'humanizer_null_controls_v1'

# Fissato prima del run esteso. Un token alfabetico sostituisce un altro token
# alfabetico; anche modifiche lessicali prudenti richiedono una verifica semantica umana.
LEXICAL_REPLACEMENTS = {
    'also': 'likewise',
    'although': 'though',
    'however': 'nevertheless',
    'therefore': 'thus',
    'use': 'utilize',
    'uses': 'utilizes',
    'used': 'utilized',
    'using': 'utilizing',
    'help': 'assist',
    'helps': 'assists',
    'helped': 'assisted',
    'helping': 'assisting',
    'begin': 'start',
    'begins': 'starts',
    'began': 'started',
    'beginning': 'starting',
    'large': 'big',
    'big': 'large',
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _digest(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def lexical_count(doc) -> int:
    """Usa la stessa definizione di token alfabetico adottata dalla stilometria."""
    return sum(bool(token.is_alpha) for token in doc)


def _sentence_ends(doc) -> set[int]:
    if not doc.has_annotation('SENT_START'):
        return set()
    return {sentence.end_char for sentence in doc.sents}


def prefix_at_word_count(text: str, n_words: int, nlp) -> tuple[str, dict]:

    if isinstance(n_words, bool) or not isinstance(n_words, int) or n_words < 0:
        raise ValueError('The target lexical count must be a nonnegative integer')
    doc = nlp(text)
    words = [token for token in doc if token.is_alpha]
    original_n = len(words)
    metadata = {
        'original_n_words': original_n,
        'target_n_words': n_words,
        'length_match_eligible': 0 < n_words <= original_n,
        'length_match_status': 'matched',
        'cut_policy': 'exact alphabetic-token prefix; attached suffix/punctuation retained',
        'hard_cut': False,
        'cut_at_sentence_boundary': None,
        'sentence_boundary_available': doc.has_annotation('SENT_START'),
    }
    if n_words > original_n:
        metadata.update(length_match_status='target_longer_than_original',
                        modified_n_words=original_n, unchanged=True,
                        cut_at_sentence_boundary=True)
        return text, metadata
    if n_words == 0:
        metadata.update(length_match_status='reference_has_no_lexical_words',
                        modified_n_words=0, unchanged=False)
        return '', metadata
    if n_words == original_n:
        metadata.update(modified_n_words=original_n, unchanged=True,
                        cut_at_sentence_boundary=True)
        return text, metadata
    last = words[n_words - 1]
    end = last.idx + len(last)
    candidate_end = end
    # Conserva, per esempio, "n't" dopo il token alfabetico "ca" di spaCy
    for token in doc[last.i + 1:]:
        if token.idx != candidate_end or token.is_alpha:
            break
        candidate_end = token.idx + len(token)
        if lexical_count(nlp.make_doc(text[:candidate_end])) == n_words:
            end = candidate_end
    prefix = text[:end]
    observed = lexical_count(nlp.make_doc(prefix))
    if observed != n_words:
        raise ValueError('The lexical prefix was retokenized to a different count')
    boundaries = _sentence_ends(doc)
    at_boundary = end in boundaries if boundaries else None
    metadata.update(modified_n_words=observed, unchanged=False,
                    hard_cut=at_boundary is not True,
                    cut_at_sentence_boundary=at_boundary,
                    cut_end_char=end)
    return prefix, metadata


def minimal_lexical_noise(text: str, nlp, *, seed: int = 42,
                          max_edits: int = 1) -> tuple[str, dict]:
    """Applica al massimo un numero fisso di sostituzioni da dizionario, ordinate per seme e ID.

    Non modifica entità riconosciute, token con iniziale maiuscola, numeri,
    PROPN/AUX/NUM/PART o parole esterne all'inventario. Questa tutela
    riguarda solo ciò che riconoscono tokenizer, tagger e NER.
    """
    if isinstance(max_edits, bool) or not isinstance(max_edits, int) or max_edits < 1:
        raise ValueError('noise max_edits must be a positive integer')
    doc = nlp(text)
    candidates = []
    for token in doc:
        replacement = LEXICAL_REPLACEMENTS.get(token.text)
        if (replacement is None or not token.is_alpha or token.text != token.text.lower()
                or token.pos_ in {'PROPN', 'AUX', 'NUM', 'PART'}
                or token.ent_iob_ not in {'', 'O'}):
            continue
        if lexical_count(nlp.make_doc(replacement)) != 1:
            raise ValueError('The noise inventory must preserve one alphabetic token')
        candidates.append((token.idx, token.idx + len(token), token.text, replacement))
    text_hash = _sha(text)
    candidates.sort(key=lambda c: _digest([seed, text_hash, c[0], c[2], c[3]]))
    chosen = candidates[:max_edits]
    modified = text
    for start, end, before, after in sorted(chosen, reverse=True):
        modified = modified[:start] + after + modified[end:]
    before_n = lexical_count(doc)
    after_n = lexical_count(nlp.make_doc(modified))
    if before_n != after_n:
        raise ValueError('Lexical noise changed the alphabetic word count')
    return modified, {
        'noise_policy': 'fixed one-token dictionary; at most max_edits; no score selection',
        'noise_max_edits': max_edits,
        'noise_candidate_count': len(candidates),
        'noise_applied': bool(chosen),
        'noise_eligible': bool(candidates),
        'noise_edits': [dict(start=start, end=end, before=before, after=after)
                        for start, end, before, after in sorted(chosen)],
        'noise_word_count_preserved': before_n == after_n,
        'noise_ner_available': doc.has_annotation('ENT_IOB'),
        'noise_is_semantic_gold': False,
        'modified_n_words': after_n,
        'unchanged': modified == text,
    }


def _deduplicate_references(references: list[dict], only: list[str] | None) -> list[dict]:
    by_key = {}
    for row in references:
        name = row.get('humanizer', '')
        if name == 'identity' or (only and name not in only):
            continue
        if name.startswith('null_'):
            raise ValueError('Null controls cannot be reused as reference humanizers')
        key = name, row.get('source_id')
        if key in by_key:
            before = by_key[key]
            fields = ['id', 'original', 'modified', 'status', 'source_group_id']
            if any(before.get(field) != row.get(field) for field in fields):
                raise ValueError(f'Conflicting reference rewrites under the same name: {key}')
        else:
            by_key[key] = row
    if only and set(only) != {name for name, _ in by_key}:
        raise ValueError('A selected reference humanizer is absent')
    if not by_key:
        raise ValueError('At least one non-identity reference humanizer is required')
    return [by_key[key] for key in sorted(by_key)]


def _row(source, condition, modified, metadata, status='ok'):
    original = source['machine']
    return {
        'id': pair_id(condition, source['id']),
        'source_id': source['id'], 'source_group_id': source['source_group_id'],
        'domain': source['domain'], 'generator': source['generator'],
        'humanizer': condition, 'dataset': 'MGTBench_humanizer_null_controls',
        'perturbation': 'null_control', 'original': original,
        'modified': modified, 'status': status,
        'metadata': dict(protocol=PROTOCOL, original_sha256=_sha(original),
                         detector_feedback_used=False, causal_attribution=False,
                         **metadata),
    }


def generate_null_controls(cohort: list[dict], references: list[dict], nlp, *,
                           seed: int = 42, noise_max_edits: int = 1,
                           include_rewrites: bool = False,
                           only: list[str] | None = None) -> list[dict]:
    references = _deduplicate_references(references, only)
    validate_inputs(cohort, references)
    sources = {str(r['id']): r for r in cohort if r['split'] == 'test'}
    generated = list(references) if include_rewrites else []
    for source in sorted(sources.values(), key=lambda r: str(r['id'])):
        original = source['machine']
        original_n = lexical_count(nlp.make_doc(original))
        generated.append(_row(source, 'null_identity', original, dict(
            control_kind='identity', comparison_eligible=True, seed=seed,
            original_n_words=original_n, modified_n_words=original_n,
            length_match_eligible=True, unchanged=True)))
        noisy, noise = minimal_lexical_noise(original, nlp, seed=seed, max_edits=noise_max_edits)
        generated.append(_row(source, 'null_lexical_noise', noisy, dict(
            control_kind='lexical_noise', comparison_eligible=noise['noise_eligible'],
            original_n_words=original_n, seed=seed, **noise)))
    for reference in references:
        source = sources[str(reference['source_id'])]
        reference_name = reference['humanizer']
        common = dict(reference_id=reference['id'], reference_humanizer=reference_name,
                      reference_status=reference['status'], seed=seed)
        if reference['status'] != 'ok':
            for kind, name in [('length', 'null_length__'),
                               ('length_and_lexical_noise', 'null_length_noise__')]:
                generated.append(_row(source, name + reference_name, '', dict(
                    control_kind=kind, comparison_eligible=False,
                    length_match_eligible=False, length_match_status='reference_unavailable',
                    target_n_words=None, noise_applied=False, **common),
                    status='reference_unavailable'))
            continue
        target = lexical_count(nlp.make_doc(reference['modified']))
        prefix, length = prefix_at_word_count(source['machine'], target, nlp)
        status = 'ok' if prefix.strip() else 'reference_has_no_lexical_words'
        generated.append(_row(source, 'null_length__' + reference_name, prefix, dict(
            control_kind='length', comparison_eligible=length['length_match_eligible'],
            **common, **length), status=status))
        noisy, noise = minimal_lexical_noise(prefix, nlp, seed=seed, max_edits=noise_max_edits)
        noise['unchanged'] = noisy == source['machine']
        generated.append(_row(source, 'null_length_noise__' + reference_name, noisy, dict(
            control_kind='length_and_lexical_noise',
            comparison_eligible=length['length_match_eligible'] and noise['noise_eligible'],
            **common, **(length | noise)), status=status))
    validate_inputs(cohort, generated)
    return generated


def _jsonl(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows)


def _write_frozen(path, body):
    if path.exists() and path.read_text() != body:
        raise ValueError(f'Refusing to replace a different frozen null-control file: {path}')
    path.write_text(body, encoding='utf-8')


def prepare(cohort_path: Path, rewrite_paths: list[Path], output: Path, *,
            seed: int = 42, noise_max_edits: int = 1,
            include_rewrites: bool = False, only: list[str] | None = None,
            nlp=None) -> dict:
    if nlp is None:
        import spacy
        
        nlp = spacy.load('en_core_web_sm')
    cohort_path, output = Path(cohort_path), Path(output)
    rewrite_paths = [Path(path) for path in rewrite_paths]
    cohort = read_jsonl(cohort_path)
    references = [row for path in rewrite_paths for row in read_jsonl(path)]
    rows = generate_null_controls(cohort, references, nlp, seed=seed,
        noise_max_edits=noise_max_edits, include_rewrites=include_rewrites, only=only)
    output.mkdir(parents=True, exist_ok=True)
    body = _jsonl(rows)
    eligible = [dict(id=r['id'], source_id=r['source_id'], domain=r['domain'],
                    generator=r['generator'], condition=r['humanizer'], status=r['status'],
                    **{key:r['metadata'].get(key, '') for key in [
                        'control_kind','reference_humanizer','reference_status',
                        'original_n_words','target_n_words','modified_n_words',
                        'length_match_eligible','length_match_status','comparison_eligible',
                        'hard_cut','cut_at_sentence_boundary','noise_eligible','noise_applied']})
                for r in rows if r['metadata'].get('protocol') == PROTOCOL]
    manifest = {
        'protocol': PROTOCOL, 'seed': seed, 'noise_max_edits': noise_max_edits,
        'cohort_sha256': hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        'reference_files': [{'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                            for path in rewrite_paths],
        'rows_sha256': _sha(body), 'n_rows': len(rows), 'include_rewrites': include_rewrites,
        'conditions': dict(Counter(r['humanizer'] for r in rows)),
        'dictionary': LEXICAL_REPLACEMENTS,
        'dictionary_sha256': _digest(LEXICAL_REPLACEMENTS),
        'nlp_model': getattr(nlp, 'meta', {}).get('name'),
        'nlp_model_version': getattr(nlp, 'meta', {}).get('version'),
        'nlp_pipes': list(nlp.pipe_names),
        'word_count_definition': 'spaCy is_alpha, as in existing stylometry',
        'prefix_policy': 'exact alphabetic count; may end mid-sentence; no padding',
        'longer_target_policy': 'original retained, length-match comparison ineligible',
        'failed_reference_policy': 'reference_unavailable; target not imputed',
        'calibration': 'unchanged human-only cohort; no recalibration on rewrites',
        'causal_attribution': False,
        'semantic_gold': False,
    }
   
    manifest_body = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    for path, value in [(output/'rewrites.jsonl', body), (output/'null_controls_run.json', manifest_body)]:
        if path.exists() and path.read_text() != value:
            raise ValueError(f'Refusing to replace a different frozen null-control file: {path}')
    _write_frozen(output/'rewrites.jsonl', body)
    _write_frozen(output/'null_controls_run.json', manifest_body)
    with (output/'eligibility.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eligible[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(eligible)
    return manifest


def summarize_null_controls(cohort, references, controls, scores, thresholds, *,
                            lower_is_ai=()):
    """Produce riepiloghi diagnostici 
    """
    references = _deduplicate_references(references, None)
    validate_inputs(cohort, references)
    tests = {str(r['id']): r for r in cohort if r['split'] == 'test'}
    reference_lookup = {(r['humanizer'], str(r['source_id'])): r for r in references}
    control_rows = [r for r in controls if r.get('metadata', {}).get('protocol') == PROTOCOL]
    validate_inputs(cohort, control_rows)
    score_records = scores.to_dict('records') if hasattr(scores, 'to_dict') else list(scores)
    lookup = {}
    detectors = set()
    for row in score_records:
        value = float(row['score'])
        if not math.isfinite(value):
            raise ValueError('Detector score is missing or nonfinite')
        humanizer = row.get('humanizer') or ''
        if isinstance(humanizer, float) and math.isnan(humanizer):
            humanizer = ''
        key = row['detector'], str(row['source_id']), row['role'], humanizer
        if key in lookup and not math.isclose(float(lookup[key]['score']), value,
                                              rel_tol=0, abs_tol=1e-12):
            raise ValueError('Conflicting saved scores for the same text key')
        lookup[key] = row
        detectors.add(row['detector'])

    def decision(detector, source, role, humanizer, expected_text):
        key = detector, str(source), role, humanizer
        if key not in lookup:
            raise ValueError(f'Missing detector score: {key}')
        row = lookup[key]
        if row.get('text_sha256') and row['text_sha256'] != _sha(expected_text):
            raise ValueError(f'Detector score text hash mismatch: {key}')
        if detector not in thresholds:
            raise ValueError(f'Missing frozen threshold for {detector}')
        threshold = float(thresholds[detector]['threshold'])
        return (float(row['score']) < threshold if detector in lower_is_ai
                else float(row['score']) > threshold)

    output = []
    conditions = sorted({r['humanizer'] for r in control_rows})
    for detector in sorted(detectors):
        for condition in conditions:
            condition_rows = [r for r in control_rows if r['humanizer'] == condition]
            eligible = [r for r in condition_rows
                        if r['status'] == 'ok' and r['metadata'].get('comparison_eligible')]
            baseline_n = control_success = reference_success = 0
            before_ai = after_ai = reference_ai = 0
            reference_name = condition_rows[0]['metadata'].get('reference_humanizer')
            for row in eligible:
                source = str(row['source_id'])
                before = decision(detector, source, 'machine', '', tests[source]['machine'])
                after = decision(detector, source, 'rewrite', condition, row['modified'])
                baseline_n += int(before)
                control_success += int(before and not after)
                before_ai += int(before)
                after_ai += int(after)
                if reference_name:
                    reference = reference_lookup[reference_name, source]
                    ref_ai = decision(detector, source, 'rewrite', reference_name, reference['modified'])
                    reference_success += int(before and not ref_ai)
                    reference_ai += int(ref_ai)
            control_asr = control_success / baseline_n if baseline_n else math.nan
            reference_asr = reference_success / baseline_n if baseline_n and reference_name else math.nan
            n = len(eligible)
            output.append(dict(
                detector=detector, condition=condition, reference_humanizer=reference_name or '',
                total_sources=len(condition_rows), eligible_sources=n,
                ineligible_sources=len(condition_rows)-n, eligible_original_detected=baseline_n,
                control_successes=control_success, control_asr=control_asr,
                reference_successes=reference_success if reference_name else '',
                reference_asr=reference_asr,
                delta_asr_reference_minus_control=reference_asr-control_asr,
                eligible_original_tpr=before_ai/n if n else math.nan,
                eligible_control_tpr=after_ai/n if n else math.nan,
                eligible_reference_tpr=reference_ai/n if n and reference_name else math.nan,
                threshold=thresholds[detector]['threshold'],
                ai_direction='lower' if detector in lower_is_ai else 'higher',
                causal_attribution=False,
            ))
    return output
