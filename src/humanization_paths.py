"""Resolve historical input paths after the thesis repository reorganization.

Frozen scientific protocols keep their original bytes and hashes. Only their
file locations change; this small adapter translates first path components.
Explicit absolute paths are always respected. User-selected output paths are
not passed through this function.
"""
from pathlib import Path


LEGACY_RESULTS = {
    "results": "results/02_stylometry/baseline",
    "results_length_matched": "results/02_stylometry/length_matched",
    "results_shortlist_validation": "results/02_stylometry/shortlist",
    "results_shortlist_length_robustness": "results/02_stylometry/shortlist_length_robustness",
    "results_semantic_full": "results/03_semantics/baseline",
    "results_semantic_extension": "results/03_semantics/extension",
    "results_semantic_clean": "results/03_semantics/entity_clean",
    "results_semantic_fairness": "results/03_semantics/fairness",
    "results_opai_style": "results/04_opai_style/document",
    "results_opai_touched_style": "results/04_opai_style/touched",
    "results_kaggle_pilot": "results/06_pilot_42/qwen",
    "results_kaggle_dipper_pilot": "results/06_pilot_42/dipper",
    "results_kaggle_v2_study": "results/07_study_126",
    "results_nli_context_pilot": "results/08_nli_calibration/context_pilot",
    "results_nli_context_calibration": "results/08_nli_calibration/calibration",
    "results_hedging_extension": "supplements/hedging/results",
}

LEGACY_INPUTS = {
    'semantic': 'semantic_initial',
    'semantic_full': 'semantic_baseline',
    'semantic_clean': 'semantic_clean',
    'semantic_extension': 'semantic_extension',
    'semantic_fairness': 'semantic_fairness',
}


def resolve_legacy_path(root, value):
    """Return a local input location without rewriting protocol contents."""
    root, path = Path(root), Path(value)
    if path.is_absolute():
        return path
    if not path.parts:
        return root / path
    first, *rest = path.parts
    if first == 'data' and rest and rest[0] in LEGACY_INPUTS:
        return root / 'inputs' / LEGACY_INPUTS[rest[0]] / Path(*rest[1:])
    # New-layout paths are not migrated a second time.
    if first == "results" and rest and rest[0] in {
        "02_stylometry", "03_semantics", "04_opai_style", "06_pilot_42",
        "07_study_126", "08_nli_calibration",
    }:
        return root / path
    if first in LEGACY_RESULTS:
        return root / LEGACY_RESULTS[first] / Path(*rest)
    return root / path
