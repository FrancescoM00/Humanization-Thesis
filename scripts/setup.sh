#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
profile="${1:-style}"
case "$profile" in
  style) requirements_file="requirements/style-lock.txt" ;;
  semantic) requirements_file="requirements/semantic-lock.txt" ;;
  *) echo 'Uso: bash scripts/setup.sh [style|semantic]' >&2; exit 2 ;;
esac
python3 -m venv .venv
if [[ "$profile" == semantic ]]; then
  .venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r "$requirements_file"
else
  .venv/bin/python -m pip install -r "$requirements_file"
fi
.venv/bin/python -m pip install --no-deps -e .
NLTK_ALLOW_PROXIED_URLOPEN=1 .venv/bin/python -m nltk.downloader -q -e -d "$PWD/.venv/nltk_data" cmudict
.venv/bin/python -m pip check
.venv/bin/python scripts/import_mgtbench.py --verify-only
echo 'Ambiente pronto. Attivazione: source .venv/bin/activate'
