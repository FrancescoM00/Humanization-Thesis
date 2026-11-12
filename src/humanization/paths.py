"""Resolve historical input paths after the thesis repository reorganization.

Frozen scientific protocols keep their original bytes and hashes. Only their
file locations change; this small adapter translates first path components.
Explicit absolute paths are always respected. User-selected output paths are
not passed through this function.
"""
from pathlib import Path


def project_root():
    """Locate an extracted project independently of OS and working directory."""
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file() and (parent / 'configs').is_dir():
            return parent
    raise RuntimeError('Open the extracted Humanization project containing pyproject.toml')


LEGACY_RESULTS = {
    "results": "results/published/02_stylometry/baseline",
    "results_length_matched": "results/published/02_stylometry/length_matched",
    "results_shortlist_validation": "results/published/02_stylometry/shortlist",
    "results_shortlist_length_robustness": "results/published/02_stylometry/shortlist_length_robustness",
    "results_semantic_full": "results/published/03_semantics/baseline",
    "results_semantic_extension": "results/published/03_semantics/extension",
    "results_semantic_clean": "results/published/03_semantics/entity_clean",
    "results_semantic_fairness": "results/published/03_semantics/fairness",
    "results_opai_style": "results/published/04_opai_style/document",
    "results_opai_touched_style": "results/published/04_opai_style/touched",
    "results_desktop_pilot": "results/published/06_pilot_42/qwen",
    "results_desktop_dipper_pilot": "results/published/06_pilot_42/dipper",
    "results_desktop_v2_study": "results/published/07_study_126",
    "results_nli_context_pilot": "results/published/08_nli_calibration/context_pilot",
    "results_nli_context_calibration": "results/published/08_nli_calibration/calibration",
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
    if first == 'inputs':
        return root / 'data/processed' / Path(*rest)
    if first == 'runs':
        return root / 'results/new_runs' / Path(*rest)
    if first == 'data' and rest and rest[0] == 'raw':
        if len(rest) == 1 or rest[1] in {'Essay_LLMs.csv','Reuters_LLMs.csv','WP_LLMs.csv','SOURCE.json'}:
            return root / 'data/raw/mgtbench' / Path(*rest[1:])
    if first == 'data' and rest and rest[0] in LEGACY_INPUTS:
        return root / 'data/processed' / LEGACY_INPUTS[rest[0]] / Path(*rest[1:])
    if first == 'results' and rest and rest[0] in {'published','new_runs'}:
        return root / path
    # New-layout paths are not migrated a second time.
    if first == "results" and rest and rest[0] in {
        "02_stylometry", "03_semantics", "04_opai_style", "06_pilot_42",
        "07_study_126", "08_nli_calibration",
    }:
        return root / 'results/published' / Path(*rest)
    if first in LEGACY_RESULTS:
        mapped=LEGACY_RESULTS[first]
        if mapped.startswith('results/'):
            mapped=mapped.replace('results/','results/published/',1)
        elif mapped=='supplements/hedging/results':
            mapped='results/published/hedging'
        return root / mapped / Path(*rest)
    return root / path
