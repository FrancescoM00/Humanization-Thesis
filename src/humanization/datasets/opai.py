"""Allinea rigorosamente le traiettorie dei dati OpAI-Bench pubblicati.

Le coppie primarie sono le versioni adiacenti v3->v4 (Compress) e v4->v5 (Expand).
Il confronto di v0 con uno dei due output è soltanto cumulativo e secondario:
le modifiche precedenti non si possono separare dall'operazione di interesse.

Questo caricatore non genera dati sostitutivi e non chiama modelli di editing.
Gli alias dello schema provengono dall'adattatore HF e dal codice di costruzione
ufficiali; schemi sconosciuti o ambigui causano un errore esplicito.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Iterable

OFFICIAL_REPOSITORY = "https://github.com/VILA-Lab/OpAI-Bench"
OFFICIAL_CODE_COMMIT = "739d647bab87c7d3c7a86d3df6c008945af5c4a2"
OFFICIAL_DATASET = "OpAI-Bench1/OpAI-Bench"


def _first(record, names, required=True):
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return value
    if required:
        raise ValueError(f"Missing required OpAI field; accepted names: {names}")
    return None


def _version(record):
    value = _first(record, ["version_index", "version"])
    match = re.fullmatch(r"(?:v)?([0-8])(?:\.0)?", str(value).strip().casefold())
    if not match:
        raise ValueError(f"Unsupported OpAI version: {value!r}")
    return int(match.group(1))


def _array(value):
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip() in {"", "null", "None"}:
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("OpAI sentence arrays must be JSON/list values")
    return list(value)


def _normalize(record):
    # record_id identifica un record versionato nei dati pubblicati, non una fonte
    # stabile: non viene MAI interpretato implicitamente come chiave della traiettoria.
    document_id = str(_first(record, ["document_hash_id", "document_id", "doc_id", "id", "essay_id"]))
    domain = str(_first(record, ["domain"]))
    generator = str(_first(record, ["generator", "model_used", "ai_model"]))
    text = _first(record, ["text", "text_clean"])
    if not isinstance(text, str):
        raise ValueError("OpAI text must be a nonempty string")
    return {
        "document_id": document_id, "domain": domain, "generator": generator,
        "version": _version(record), "text": text,
        "operation": str(_first(record, ["edit_operation", "operation"], required=False) or "").casefold(),
        "record_id": str(record.get("record_id", "")),
        "sentences": _array(record.get("sentences")),
        "sentence_ids": _array(record.get("sentence_ids")),
    }


def _balanced_selection(candidates, n, rng):
    """Seleziona a rotazione dagli strati dominio×generatore, con ordine interno fissato dal seme."""
    strata = {}
    for candidate in candidates:
        strata.setdefault((candidate["domain"], candidate["generator"]), []).append(candidate)
    for values in strata.values():
        rng.shuffle(values)
    keys = sorted(strata)
    rng.shuffle(keys)
    chosen = []
    while len(chosen) < min(n, len(candidates)):
        any_added = False
        for key in keys:
            if strata[key] and len(chosen) < n:
                chosen.append(strata[key].pop())
                any_added = True
        if not any_added:
            break
    return chosen


def align_opai_records(
    records: Iterable[dict], n_per_operation: int = 200, seed: int = 42,
    include_cumulative: bool = True, source_metadata: dict | None = None,
) -> tuple[list[dict], dict]:
    """Restituisce coppie valide di documenti e un manifest completo di selezione e provenienza.

    Per ogni operazione primaria seleziona a rotazione dagli strati dominio×editor
    disponibili. Le coppie cumulative usano esattamente le traiettorie adiacenti
    selezionate, se è presente v0. Le versioni mancanti sono conteggiate senza
    inventare sostituti.
    """
    if n_per_operation < 1:
        raise ValueError("n_per_operation must be positive")
    trajectories, duplicates, rows = {}, 0, 0
    for raw in records:
        row = _normalize(raw)
        rows += 1
        key = (row["document_id"], row["domain"], row["generator"])
        versions = trajectories.setdefault(key, {})
        if row["version"] in versions:
            if versions[row["version"]]["text"] != row["text"]:
                raise ValueError(f"Ambiguous duplicate OpAI trajectory/version: {key}/{row['version']}")
            duplicates += 1
            continue
        versions[row["version"]] = row
    if not trajectories:
        raise ValueError("No OpAI records found")
    rng, pairs, counts = random.Random(seed), [], Counter()

    def make_pair(before, after, operation, comparison):
        identity = f"{after['document_id']}|{after['domain']}|{after['generator']}|{before['version']}|{after['version']}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
        original, modified = before["text"], after["text"]
        n_before, n_after = len(original.split()), len(modified.split())
        changes = []
        # Solo identificatori di frase espliciti e stabili giustificano l’allineamento locale.
        # Array di pari lunghezza non provano da soli che le posizioni delle frasi equivalgano.
        if before["sentence_ids"] is not None and after["sentence_ids"] is not None:
            if before["sentences"] is None or after["sentences"] is None:
                raise ValueError("sentence_ids require sentence text arrays")
            if len(before["sentences"]) != len(before["sentence_ids"]) or len(after["sentences"]) != len(after["sentence_ids"]):
                raise ValueError("OpAI sentence_ids/text lengths differ")
            left = dict(zip(map(str, before["sentence_ids"]), before["sentences"]))
            right = dict(zip(map(str, after["sentence_ids"]), after["sentences"]))
            changes = [{"sentence_id": sid, "original": left[sid], "modified": right[sid]}
                       for sid in left if sid in right and left[sid] != right[sid]]
        return {
            "id": "opai:" + digest, "pair_id": "opai:" + digest,
            "source_id": after["document_id"], "domain": after["domain"], "generator": after["generator"],
            "perturbation": "opai_" + operation, "original": original, "modified": modified,
            "target_original": None, "target_modified": None,
            "metadata": {
                "dataset": OFFICIAL_DATASET, "comparison": comparison,
                "from_version": "v" + str(before["version"]), "to_version": "v" + str(after["version"]),
                "source_record_id": before["record_id"], "target_record_id": after["record_id"],
                "original_whitespace_words": n_before, "modified_whitespace_words": n_after,
                "length_ratio": n_after / n_before if n_before else None,
                "expected_length_direction_observed": n_after < n_before if operation == "compress" else n_after > n_before,
                "text_changed": original != modified, "aligned_changed_sentences": changes,
                "directionality_is_hypothesis_not_gold_label": True,
            },
        }

    for before_version, after_version, operation in [(3, 4, "compress"), (4, 5, "expand")]:
        candidates = []
        for key, versions in trajectories.items():
            if before_version not in versions or after_version not in versions:
                counts[f"{operation}:incomplete_trajectory"] += 1
                continue
            after = versions[after_version]
            if after["operation"] and after["operation"] not in {operation, "compression" if operation == "compress" else "expansion"}:
                raise ValueError(f"OpAI v{after_version} operation disagrees: {after['operation']!r}")
            candidates.append({"key": key, "domain": key[1], "generator": key[2]})
        selected = _balanced_selection(candidates, n_per_operation, rng)
        counts[f"{operation}:available_complete_trajectories"] = len(candidates)
        counts[f"{operation}:selected_adjacent"] = len(selected)
        for candidate in selected:
            versions = trajectories[candidate["key"]]
            pairs.append(make_pair(versions[before_version], versions[after_version], operation, "adjacent_operation"))
            if include_cumulative:
                if 0 in versions:
                    pairs.append(make_pair(versions[0], versions[after_version], operation, "cumulative_from_human_v0"))
                else:
                    counts[f"{operation}:missing_v0_for_selected"] += 1
    if not any(p["metadata"]["comparison"] == "adjacent_operation" for p in pairs):
        raise ValueError("No complete v3/v4 or v4/v5 OpAI trajectories; provide aligned released records")
    manifest = {
        "official_repository": OFFICIAL_REPOSITORY, "official_code_commit": OFFICIAL_CODE_COMMIT,
        "official_dataset": OFFICIAL_DATASET, "source_metadata": source_metadata or {"method": "local_records", "dataset_revision": None},
        "rows_read": rows, "n_trajectories": len(trajectories), "exact_duplicate_rows_ignored": duplicates,
        "n_per_operation_requested": n_per_operation, "seed": seed, "counts": dict(counts),
        "pair_counts": dict(Counter((p["perturbation"] + ":" + p["metadata"]["comparison"]) for p in pairs)),
        "expected_direction": {"compress": "before entails after more strongly than reverse if content was removed",
                               "expand": "after entails before more strongly than reverse only if expansion adds information"},
        "caveats": ["OpAI v0 is human-written, later versions are cumulatively AI-edited",
                    "Adjacent versions isolate the named stage more closely than v0 comparisons",
                    "Expansion without new unsupported facts can legitimately preserve bidirectional entailment",
                    "Operation names and length ratios are not gold entailment labels",
                    "Sentence arrays without explicit stable IDs are not presumed aligned"],
    }
    return pairs, manifest


def read_opai_file(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No local OpAI data at {path}; released data: https://huggingface.co/datasets/{OFFICIAL_DATASET}")
    if path.suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("OpAI JSON input must be a list of records")
        return records
    import pandas as pd
    if path.suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False).to_dict(orient="records")
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError("OpAI input must be CSV, Parquet, JSON or JSONL")


def download_official_records(output: str | Path, revision: str, split: str = "test") -> dict:
    """Scarica frammenti Parquet ufficiali verificati e salva le versioni necessarie.

    La repository ufficiale include copie Parquet viewer/default/{split} nella
    stessa versione fissata. Sono molto più piccole dell'esportazione CSV e
    conservano lo schema completo pubblicato. Prima della lettura, ogni frammento
    in cache viene confrontato con la SHA-256 LFS autorevole della revisione.
    Nel JSONL dell'esperimento entrano soltanto le colonne di identificazione,
    versione, testo e frasi delle versioni v0/v3/v4/v5. I frammenti originali e
    le loro impronte restano invariati.

    Richiede requests e pyarrow. Riusa la cache solo se il checksum coincide;
    non sostituisce file dell'utente né output completati. Un errore non lascia
    un JSONL finale parzialmente valido. TLS rimane abilitato.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin the exact 40-character Hugging Face dataset revision SHA")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if split not in {"train", "dev", "test"}:
        raise ValueError("OpAI split must be train, dev or test")
    import requests
    import pyarrow.parquet as pq
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    tree_url = f"https://huggingface.co/api/datasets/{OFFICIAL_DATASET}/tree/{revision}/viewer/default/{split}"
    response = requests.get(tree_url, params={"recursive": False, "expand": False}, timeout=60)
    response.raise_for_status()
    if response.links.get("next"):
        raise RuntimeError("Official shard listing is paginated; refusing incomplete discovery")
    files = sorted((entry for entry in response.json() if entry.get("type") == "file" and entry["path"].endswith(".parquet")),
                   key=lambda entry: entry["path"])
    if not files:
        raise RuntimeError("Pinned official release has no viewer Parquet shards for this split")
    for entry in files:
        expected = (entry.get("lfs") or {}).get("oid")
        if not expected or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError(f"No authoritative LFS SHA-256 for {entry['path']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.parent / "official_parquet"
    cache.mkdir(parents=True, exist_ok=True)

    def file_hash(path):
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def fetch(entry):
        path = cache / Path(entry["path"]).name
        expected = entry["lfs"]["oid"]
        if path.exists():
            if path.stat().st_size != entry["size"] or file_hash(path) != expected:
                raise RuntimeError(f"Existing official source cache fails checksum; preserve and inspect: {path}")
        else:
            url = f"https://huggingface.co/datasets/{OFFICIAL_DATASET}/resolve/{revision}/{entry['path']}"
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".download", dir=cache, delete=False) as handle:
                    temporary_path = Path(handle.name)
                    with requests.get(url, stream=True, timeout=(30, 180)) as download:
                        download.raise_for_status()
                        for block in download.iter_content(128 * 1024):
                            handle.write(block)
                if temporary_path.stat().st_size != entry["size"] or file_hash(temporary_path) != expected:
                    raise RuntimeError(f"Official artifact size/checksum mismatch: {entry['path']}")
                temporary_path.replace(path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        return {"repository_path": entry["path"], "local_file": str(path), "bytes": entry["size"], "sha256": expected}

    with ThreadPoolExecutor(max_workers=2) as executor:
        source_files = list(executor.map(fetch, files))

    requested_columns = ["record_id", "id", "document_hash_id", "split", "domain", "generator", "subset",
                         "ablation_type", "edit_operation", "version", "version_index", "text", "sentences", "sentence_ids"]
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Partial output already exists; preserve and inspect before retrying: {temporary}")
    count = 0
    source_rows = 0
    columns_saved = set()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for entry in source_files:
                parquet = pq.ParquetFile(entry["local_file"])
                columns = [column for column in requested_columns if column in parquet.schema_arrow.names]
                columns_saved.update(columns)
                source_rows += parquet.metadata.num_rows
                for batch in parquet.iter_batches(batch_size=512, columns=columns):
                    for record in batch.to_pylist():
                        if _version(record) in {0, 3, 4, 5}:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            count += 1
        if not count:
            raise ValueError("Official split had no v0/v3/v4/v5 records")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"method": "official_huggingface_viewer_parquet_verified", "dataset": OFFICIAL_DATASET,
            "dataset_revision": revision, "split": split, "source_rows": source_rows, "rows_saved": count,
            "source_files": source_files, "columns_saved": sorted(columns_saved),
            "versions_saved": ["v0", "v3", "v4", "v5"], "sha256": file_hash(output)}
