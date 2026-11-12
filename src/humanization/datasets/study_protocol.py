"""Conserva l'esperimento di 126 fonti e dichiara il trattamento degli input prima dell'inferenza."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

PANEL_HASHES = {
    'smoke': '9c6553dce584ce1ba8686efc48ef29745e16d8ae92fd26d66b0b6dcbb06eb21c',
    'pilot': 'ed6b343dc972038515ff6dc4ebdfe7e1ce6ce50386cb49c5e104732da440574a',
    'study': '84f9f6933b2ee72412d65eedbf8f1fbe70cf0139bcd3d6d395cb85250c0edb68',
}
PANEL_COUNTS = {'smoke': 3, 'pilot': 42, 'study': 126}
KNOWN_INSTRUCTION_SOURCES = {
    'Essay:245:198f5b62c8f94655',
    'Reuters:903:3d3ffb6dca295be8',
}
# Soltanto segnalazioni candidate, mai esclusioni o etichette certe sul tipo di input.
LEADING_TASK = re.compile(r'^\s*(?:\d+[.)]\s*)?(?:write|discuss|describe|explain|compose|provide|create|compare|analy[sz]e)\b', re.I)
TASK_CUE = re.compile(r'\b(?:write\s+(?:a|an|one)|\d+\s+words?|APA\s+format|MLA\s+format|cite\s+\w+\s+sources?)\b', re.I)


def validate_groups(rows):
    calibration = [r for r in rows if r['split'] == 'calibration']
    tests = [r for r in rows if r['split'] == 'test']
    cg = {r['source_group_id'] for r in calibration}
    tg = {r['source_group_id'] for r in tests}
    if len(cg) != len(calibration) or len(tg) != len(tests) or cg & tg:
        raise ValueError('Calibration and test groups must be unique and disjoint')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate cohort IDs')
    return calibration, tests


def source_screen(rows):
    _, tests = validate_groups(rows)
    screened = []
    for row in tests:
        text = row['machine']
        cues = []
        if LEADING_TASK.search(text):
            cues.append('leading_task_verb')
        if TASK_CUE.search(text):
            cues.append('task_or_format_cue')
        if row['id'] in KNOWN_INSTRUCTION_SOURCES:
            cues.append('instruction_like_in_previous_assistant_review')
        screened.append({
            'source_id': row['id'], 'source_group_id': row['source_group_id'],
            'domain': row['domain'], 'generator': row['generator'],
            'original_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'include_primary': True,
            'input_form_candidate': 'review_instruction_or_mixed' if cues else 'not_flagged',
            'candidate_reasons': '|'.join(cues), 'input_form_manual_prose_instructions_mixed': '',
            'reviewer_id': '', 'notes': '', 'original_machine': text,
        })
    return screened


def prepare_study_protocol(cohort, output, *, panel='study'):
    cohort, output = Path(cohort), Path(output)
    digest = hashlib.sha256(cohort.read_bytes()).hexdigest()
    if digest != PANEL_HASHES[panel]:
        raise ValueError('Cohort differs from the previously frozen panel; no reselection is allowed')
    rows = [json.loads(s) for s in cohort.read_text().splitlines() if s.strip()]
    calibration, tests = validate_groups(rows)
    if len(calibration) != 300 or len(tests) != PANEL_COUNTS[panel]:
        raise ValueError('Frozen panel counts differ')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        'protocol': 'humanizer_study_126_v2', 'panel': panel,
        'cohort_sha256': digest, 'calibration_humans': len(calibration), 'test_sources': len(tests),
        'input_policy': 'All frozen test sources stay in the primary result, including instructions and mixed inputs.',
        'screening': 'Original text only; candidate flags are not labels and never exclude a source.',
        'secondary_analysis': 'Prose-only results require completed source-only type labels; report missing labels and all exclusions.',
        'known_two_instruction_sources': sorted(KNOWN_INSTRUCTION_SOURCES),
        'human_type_labels_received': False,
        'planned_conditions': ['identity', 'qwen_humanize', 'qwen14b_humanize', 'dipper', 'dipper_context3'],
        'planned_detector_families': ['supervised RoBERTa classifiers', 'zero-shot Falcon Binoculars'],
        'primary_detection_question': 'Within-source change in AI detection, including newly detected originals; ASR is secondary to the full transition table.',
        'primary_length_controls': ['original prefix matched to each rewrite lexical count', 'minimal lexical noise', 'noise on length-matched prefix'],
        'semantic_interpretation': 'Continuous scores and independent labels; do not compare raw cosine/BERTScore/NLI means as accuracy.',
        'extension_420': 'deferred; not selected or executed',
    }
    manifest_path = output / 'study_protocol.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Refusing to replace a different study protocol')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    screen = output / 'source_screening.csv'
    if not screen.exists():
        screened = source_screen(rows)
        with screen.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(screened[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(screened)
    return manifest
