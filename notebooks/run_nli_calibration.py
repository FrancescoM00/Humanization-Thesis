#!/usr/bin/env python3
"""Run the frozen NLI calibration panel on Kaggle without rerunning humanizers.

The parent process uses only the standard library. Model loading, scoring and
analysis run in child processes using Kaggle's existing CUDA-enabled PyTorch.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


REPOSITORY_URL = "https://github.com/FrancescoM00/Humanization-Thesis.git"
MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MODEL_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
VERSIONS = {"transformers": "4.57.3", "numpy": "2.2.6", "pandas": "2.3.3",
            "scikit-learn": "1.7.2", "sentencepiece": "0.2.1"}
REQUIRED_BUNDLE_FILES = (
    "inputs/pairs.jsonl", "inputs/alignment.json", "inputs/labels.csv",
    "inputs/policy.json", "inputs/preparation_manifest.json",
)
ARCHIVE_EXCLUSIONS = {"contextual_probability_cache.jsonl", "contextual_nli_units.jsonl"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkout_repository(working):
    """Use an existing checkout, or clone the thesis repository on Kaggle."""
    configured = os.environ.get("HUMANIZATION_REPO_DIR")
    local = Path(configured) if configured else Path(__file__).resolve().parents[1]
    if (local / "src/semantics").is_dir() and (local / "inputs/nli_calibration").is_dir():
        return local
    if configured:
        raise FileNotFoundError("HUMANIZATION_REPO_DIR non contiene codice e input NLI della consegna.")
    repository = Path(working) / ".humanization_code/Humanization_Thesis_nli_calibration"
    if not (repository / ".git").is_dir():
        repository.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPOSITORY_URL, str(repository)], check=True)
    reference = os.environ.get("HUMANIZATION_REPO_REF", "main")
    subprocess.run(["git", "fetch", "--depth", "1", "origin", reference], cwd=repository, check=True)
    subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repository, check=True)
    return repository


def install_bundle(run, *, repository=None):
    """Freeze input files locally; code stays in the separately recorded checkout."""
    run = Path(run)
    source = run / "source_bundle.zip"
    if not source.is_file():
        repository = Path(repository) if repository else Path(__file__).resolve().parents[1]
        inputs = repository / "inputs/nli_calibration"
        temporary = source.with_suffix(".download")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in REQUIRED_BUNDLE_FILES:
                    path = inputs / Path(name).name
                    if not path.is_file():
                        raise FileNotFoundError(f"Input congelato mancante: {path}")
                    archive.write(path, name)
            temporary.replace(source)
        finally:
            temporary.unlink(missing_ok=True)
    digest = sha256(source)
    previous = run / "runtime.json"
    if previous.is_file() and json.loads(previous.read_text()).get("bundle_sha256") != digest:
        raise RuntimeError("Il bundle è cambiato: usare un nuovo HUMANIZATION_NLI_RUN_SUFFIX.")
    destination = run / "bundle"
    with zipfile.ZipFile(source) as archive:
        names = set(archive.namelist())
        if not set(REQUIRED_BUNDLE_FILES) <= names:
            raise RuntimeError("Bundle incompleto: non avvio lo scoring.")
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Percorso non valido nel bundle.")
            if member.is_dir():
                continue
            target = destination / relative
            data = archive.read(member)
            if target.exists():
                if not target.is_file() or target.read_bytes() != data:
                    raise RuntimeError(f"Input/sorgente modificato: {relative}. Usare un nuovo suffisso.")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    return destination, digest


def child_environment(run, working):
    scratch = run / "_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    cache = working / ".humanization_runtime_cache" / "nli_calibration"
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({name: str(scratch) for name in ("TMPDIR", "TMP", "TEMP")})
    env.update(HF_HOME=str(cache / "huggingface"), HF_HUB_CACHE=str(cache / "huggingface/hub"),
               HF_XET_CACHE=str(cache / "xet"), HF_HUB_DISABLE_XET="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0", USE_TORCH="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2",
               MPLCONFIGDIR=str(cache / "matplotlib"), XDG_CACHE_HOME=str(cache / "cache"))
    for variable in ("TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if variable in env:
            env[variable] = env["HF_HUB_CACHE"]
    return env


GPU_PROBE = """import json, torch
assert torch.cuda.is_available(), 'Abilitare una GPU Kaggle prima di questa calibrazione.'
x = torch.ones((16, 16), dtype=torch.float32, device='cuda:0')
assert bool(torch.isfinite(x @ x).all())
torch.cuda.synchronize()
print(json.dumps({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}))
"""
DEPENDENCY_PROBE = """import importlib.metadata, json
expected = {versions!r}
for package, version in expected.items():
    assert importlib.metadata.version(package) == version, (package, version)
import numpy, pandas, sklearn, sentencepiece, requests
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification
"""


def choose_python(bundle, working, old_runtime, env, *, runner=subprocess.run):
    """Matching readable venvs are never recreated or written."""
    candidates = [old_runtime / "venv/bin/python",
                  working / ".humanization_runtime_cache/nli_venv/bin/python",
                  working / ".humanization_runtime_cache/venv/bin/python"]
    probe = DEPENDENCY_PROBE.format(versions=VERSIONS)
    for candidate in candidates:
        if candidate.is_file():
            result = runner([str(candidate), "-c", probe], env=env, capture_output=True, text=True)
            if result.returncode == 0:
                print("Riutilizzo venv senza modificarla:", candidate.parent.parent, flush=True)
                return candidate
    venv = working / ".humanization_runtime_cache/nli_venv"
    venv.mkdir(parents=True, exist_ok=True)
    python = venv / "bin/python"
    runner([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv)],
           env=env, check=True)
    pip_probe = runner([str(python), "-m", "pip", "--version"], env=env, capture_output=True, text=True)
    if not (venv / "bin/pip").is_file() or pip_probe.returncode != 0:
        runner([sys.executable, "-m", "pip", "--python", str(python),
                "install", "--upgrade", "--ignore-installed", "pip"], env=env, check=True)
    runner([str(python), "-m", "pip", "install", "-r", str(bundle / "requirements/nli_calibration.txt")],
           env=env, check=True)
    runner([str(python), "-c", probe], env=env, check=True)
    return python


def complete_model(path):
    """Recognize only the exact pinned model with all downloaded files present."""
    try:
        manifest = json.loads((path / "DOWNLOAD_MANIFEST.json").read_text())
        if (manifest.get("repository") != MODEL_ID or manifest.get("revision") != MODEL_REVISION
                or manifest.get("kind") != "nli"):
            return False
        rows = manifest["files"]
        names = {row["name"] for row in rows}
        if not {"config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"} <= names:
            return False
        return all(not Path(row["name"]).is_absolute() and ".." not in Path(row["name"]).parts
                   and (path / row["name"]).is_file()
                   and (path / row["name"]).stat().st_size == int(row["bytes"]) for row in rows)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def ensure_model(bundle, working, old_runtime, python, env, *, runner=subprocess.run):
    shared = working / ".humanization_semantic_models" / "nli"
    for path in [shared, old_runtime / "models/nli",
                 working / ".humanization_nli_calibration_models/nli"]:
        if complete_model(path):
            print("Riutilizzo DeBERTa-v3 base:", path, flush=True)
            return path
    # Preserve an existing incompatible model rather than replacing its weights.
    target = (working / ".humanization_nli_calibration_models/nli") if shared.exists() else shared
    target.parent.mkdir(parents=True, exist_ok=True)
    runner([str(python), str(bundle / "scripts/download_semantic_models.py"),
            "--model", "nli", "--revision", MODEL_REVISION, "--output", str(target.parent)],
           cwd=bundle, env=env, check=True)
    if not complete_model(target):
        raise RuntimeError("Download NLI incompleto o revisione diversa: non avvio lo scoring.")
    return target


def export_review(run):
    """Keep resumable heavy caches on Kaggle; return a small analysis archive."""
    archive = run.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(run.rglob("*")):
            if (not path.is_file() or "_tmp" in path.relative_to(run).parts
                    or path.name in ARCHIVE_EXCLUSIONS):
                continue
            output.write(path, str(path.relative_to(run)))
    return archive


def main():
    working = Path("/kaggle/working")
    if not working.is_dir():
        raise RuntimeError("Eseguire questa cella nel notebook Kaggle con una GPU abilitata.")
    suffix = os.environ.get("HUMANIZATION_NLI_RUN_SUFFIX", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]*", suffix):
        raise ValueError("HUMANIZATION_NLI_RUN_SUFFIX deve contenere solo lettere, numeri, _ o -.")
    run = working / ("nli_calibration_thesis_v1" + suffix)
    run.mkdir(parents=True, exist_ok=True)
    old_runtime = Path("/kaggle/temp/humanization_runtime")
    env = child_environment(run, working)
    repository = checkout_repository(working)
    env["PYTHONPATH"] = str(repository / "src") + os.pathsep + str(repository)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    previous = run / "runtime.json"
    if previous.is_file() and json.loads(previous.read_text()).get("repository_commit") != commit:
        raise RuntimeError("Codice diverso dal run precedente: scegliere un nuovo suffisso.")
    for key in ("TMPDIR", "TMP", "TEMP"):
        os.environ[key] = env[key]
    tempfile.tempdir = None
    # Check the image's CUDA before installing packages or downloading weights.
    subprocess.run([sys.executable, "-c", GPU_PROBE], env=env, check=True)
    bundle, digest = install_bundle(run, repository=repository)
    python = choose_python(repository, working, old_runtime, env)
    subprocess.run([str(python), "-c", GPU_PROBE], env=env, check=True)
    model = ensure_model(repository, working, old_runtime, python, env)
    runtime = dict(workflow="nli_context_calibration_v1", bundle_sha256=digest,
                   repository_commit=commit, repository_path=str(repository),
                   created_at=datetime.now(timezone.utc).isoformat(), python=str(python),
                   model_repository=MODEL_ID, model_revision=MODEL_REVISION, model_path=str(model),
                   device="cuda:0", precision="float32", batch_size=4, max_pair_tokens=384,
                   review_archive_exclusions=sorted(ARCHIVE_EXCLUSIONS),
                   excluded_files_remain_on_kaggle_for_resume=True)
    (run / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print("Calibrazione: 139 positivi presunti, 297 alterazioni e 137 identità; test storico escluso.", flush=True)
    print("Stima orientativa: 45–90 minuti di GPU. Gli score completati restano in cache.", flush=True)
    subprocess.run([str(python), str(repository / "scripts/score_contextual_nli.py"),
        "--pairs", str(bundle / "inputs/pairs.jsonl"), "--alignment", str(bundle / "inputs/alignment.json"),
        "--model", str(model), "--output", str(run / "scores"), "--device", "cuda:0",
        "--batch-size", "4", "--threads", "2", "--max-pair-tokens", "384"],
        cwd=repository, env=env, check=True)
    subprocess.run([str(python), str(repository / "scripts/calibrate_contextual_nli.py"), "analyze",
        "--input", str(bundle / "inputs"), "--scores", str(run / "scores/contextual_nli.csv"),
        "--output", str(run / "analysis"), "--bootstrap", "2000", "--seed", "42"],
        cwd=repository, env=env, check=True)
    selected = run / "analysis/selection.json"
    if selected.is_file():
        print("Scelta sullo split di calibrazione:", selected.read_text(), flush=True)
    archive = export_review(run)
    print("Archivio da restituire:", archive, flush=True)
    print("Cache ed evidenze dettagliate restano su Kaggle; modelli e venv sono fuori dall'archivio.", flush=True)
    try:
        from IPython.display import FileLink, display
        display(FileLink(str(archive)))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
