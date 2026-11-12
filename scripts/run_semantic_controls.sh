#!/usr/bin/env bash
# Run from an already prepared extension; reuse unchanged real model scores.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
python_bin=.venv/bin/python
test -s data/semantic_extension/pairs.jsonl
for stage in lexical embedding bertscore; do
    "$python_bin" scripts/score_semantic_extension.py --stage "$stage" --threads 4
done
"$python_bin" scripts/rescale_bertscore.py
"$python_bin" scripts/plan_nli_jobs.py --data-dir data/semantic_extension --output results/03_semantics/extension
"$python_bin" scripts/precompute_nli_onnx.py --data-dir data/semantic_extension --output results/03_semantics/extension --threads 4 --batch-size 8
"$python_bin" scripts/run_semantic_experiment.py --stage nli --data-dir data/semantic_extension --output results/03_semantics/extension --threads 4 --require-nli-cache
"$python_bin" scripts/run_semantic_experiment.py --stage report --data-dir data/semantic_extension --output results/03_semantics/extension
"$python_bin" scripts/analyze_semantic_extension.py --bootstrap 1000
