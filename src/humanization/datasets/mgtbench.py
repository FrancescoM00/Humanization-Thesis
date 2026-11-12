"""Prepare a small, balanced, paired MGTBench sample without hiding provenance.

Input is the original wide CSV schema used by TH-Bench's ``mgt1`` loader,
not the machine-only length-bin CSVs emitted by its attack script.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

DOMAINS = {"Essay": "Essay", "WritingPrompts": "WP", "Reuters": "Reuters"}
GENERATORS = ["ChatGPT-turbo", "Claude", "ChatGLM", "Dolly", "ChatGPT", "GPT4All", "StableLM"]
TH_BENCH_COMMIT = "5c1d99cefee49208206b6fa75d3a628ac9935f86"
TERMINAL_BOUNDARY = re.compile(r"[.!?…]+[\"'”’\)\]]*(?=\s|$)")
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
                 "fig", "no", "inc", "ltd", "co", "approx", "dept", "e.g", "i.e"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(text: str) -> str:
    """Only for duplicate/group detection; never used as model input."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def text_hash(text: str) -> str:
    return hashlib.sha256(canonical(text).encode("utf-8")).hexdigest()


def process_spaces(text: str) -> str:
    """The exact spacing repair from the pinned TH-Bench mgt1 loader.

    It does NOT lowercase, remove punctuation, strip AI vocabulary, trim
    incomplete sentences or remove chatbot boilerplate. Those interventions
    could erase the very style signal being measured.
    """
    for old, new in [
        (" ,", ","), (" .", "."), (" ?", "?"), (" !", "!"),
        (" ;", ";"), (" '", "'"), (" ’ ", "'"), (" :", ":"),
        ("<newline>", "\n"), ('`` ', '"'), (" ''", '"'), ("''", '"'),
        (".. ", "... "), (" )", ")"), ("( ", "("), (" n't", "n't"),
        (" i ", " I "), (" i'", " I'"), ("\\'", "'"), ("\n ", "\n"),
    ]:
        text = text.replace(old, new)
    return text.strip()


def _cell(value: object) -> str:
    return process_spaces(value) if isinstance(value, str) else ""


def trim_incomplete_tail(text: str) -> str:
    """Keep through the last plausible sentence-ending mark, symmetrically.

    This punctuation heuristic is NOT grammatical completeness verification.
    Decimal points and ordinary internal abbreviations/initials are ignored.
    Quotes/brackets immediately after an ending mark stay in the retained text.
    When there is no plausible ending, return an empty string so the subsequent
    minimum-length filter excludes the text rather than inventing punctuation.
    """
    text = text.rstrip()
    for match in reversed(list(TERMINAL_BOUNDARY.finditer(text))):
        prefix = text[:match.end()]
        # Preserve an already terminated text, including abbreviations used at
        # the end of a sentence (e.g. "He lives in the U.S."). Internal titles
        # such as "Dr. Smith" must not become artificial sentence boundaries.
        if match.end() != len(text) and text[match.start()] == ".":
            before = text[:match.start()]
            token_match = re.search(r"([\w.]+)$", before)
            token = token_match.group(1) if token_match else ""
            if token.casefold() in ABBREVIATIONS or re.fullmatch(r"(?:[A-Za-z]\.)*[A-Za-z]", token):
                continue
        return prefix.rstrip()
    return ""


def _choose_pairs(candidates: list[dict], quotas: dict[str, int], rng: np.random.Generator,
                  used_texts: set[str]) -> list[tuple[dict, str]]:
    """Randomized matching enforces generator quotas and one machine per human.

    Identical generated strings are forbidden globally. If a tentative matching
    repeats a machine string, ban that row/generator edge and solve again.
    """
    slots = [gen for gen, n in quotas.items() for _ in range(n)]
    if len(candidates) < len(slots):
        raise ValueError(f"Only {len(candidates)} unique eligible humans; need {len(slots)}.")
    blocked = set()
    random_cost = rng.random((len(candidates), len(slots)))
    eligible = [{g: text_hash(t) not in used_texts for g, t in c["machines"].items()}
                for c in candidates]
    for _ in range(sum(len(c["machines"]) for c in candidates) + 1):
        costs = random_cost.copy()
        for i, candidate in enumerate(candidates):
            for j, gen in enumerate(slots):
                if not eligible[i].get(gen, False) or (i, gen) in blocked:
                    costs[i, j] = 1e9
        rows, cols = linear_sum_assignment(costs)
        if len(rows) != len(slots) or np.any(costs[rows, cols] >= 1e9):
            counts = {g: sum(g in c["machines"] for c in candidates) for g in quotas}
            raise ValueError(f"Cannot satisfy generator quotas {quotas}; eligible rows {counts}. "
                             "Use more data or explicitly request fewer pairs.")
        selected, seen, conflict = [], set(), False
        for i, j in zip(rows, cols):
            gen = slots[j]
            digest = text_hash(candidates[i]["machines"][gen])
            if digest in seen:
                blocked.add((int(i), gen))
                conflict = True
            else:
                seen.add(digest)
                selected.append((candidates[i], gen))
        if not conflict:
            return selected
    raise RuntimeError("Duplicate-resolution matching did not converge.")


def prepare_dataset(data_dir: Path | str, output_dir: Path | str, n_pairs: int = 300,
                    seed: int = 42, min_words: int = 50,
                    generators: list[str] | None = None,
                    clean_sentence_ends: bool = True) -> tuple[pd.DataFrame, dict]:
    """Save samples.jsonl + manifest.json and return (records, manifest).

    Each domain has ``n_pairs`` humans and ``n_pairs`` machines, with one
    randomly assigned valid generator per source pair and near-equal generator
    quotas. Human texts are not replicated seven times. ``group_id`` links
    repeated prompts and human texts and must be respected during splitting.
    The default removes trailing incomplete fragments BEFORE the symmetric
    minimum-word filter. Disable only for an explicitly labelled sensitivity
    run; the default remains a locally specified cleaning protocol.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    generators = list(GENERATORS if generators is None else generators)
    if n_pairs < len(generators) or min_words < 1 or not generators or len(set(generators)) != len(generators):
        raise ValueError("Need distinct generators, at least one pair per generator, and min_words >= 1.")
    if any(g not in GENERATORS for g in generators):
        raise ValueError(f"Unknown generator; expected a subset of {GENERATORS}.")
    frames, source_files = {}, {}
    for domain, short in DOMAINS.items():
        path = data_dir / f"{short}_LLMs.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Obtain all three original MGTBench wide CSVs "
                                    "using src/humanization/datasets/workflows/import_mgtbench.py, or supply their local directory.")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = {"human", *generators} - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name}: missing required columns {sorted(missing)}.")
        if frame.empty:
            raise ValueError(f"{path.name}: no data rows.")
        frames[domain] = frame
        source_files[domain] = {"file": path.name, "sha256": sha256_file(path), "rows": len(frame)}

    prepared, cleanup_diagnostics = {}, {}
    for domain, frame in frames.items():
        prepared[domain], cleanup_diagnostics[domain] = {}, {}
        for origin in ["human", *generators]:
            cells, counts = [], Counter()
            for value in frame[origin]:
                original = _cell(value)
                text = trim_incomplete_tail(original) if clean_sentence_ends else original
                removed_words = len(original.split()) - len(text.split())
                cells.append({"text": text, "changed": original != text, "removed_words": removed_words})
                counts["n_input_nonempty"] += bool(original)
                counts["n_tail_trimmed"] += bool(text and original != text)
                counts["n_no_terminal_boundary"] += bool(original and not text and clean_sentence_ends)
                counts["whitespace_words_removed"] += removed_words
                counts["n_below_min_after_cleaning"] += len(text.split()) < min_words
            prepared[domain][origin] = cells
            cleanup_diagnostics[domain][origin] = dict(counts)

    # A machine output copied verbatim from ANY source human is not usable as a
    # label-independent stylometric comparison. Remove the ambiguous machine.
    all_human_hashes = {text_hash(c["text"]) for texts in prepared.values()
                        for c in texts["human"] if c["text"]}
    seen_humans, candidates_by_domain, diagnostics = set(), {}, {}
    for domain, frame in frames.items():
        candidates, removed = [], Counter()
        available = Counter()
        for source_row, row in frame.iterrows():
            human = prepared[domain]["human"][source_row]["text"]
            if len(human.split()) < min_words:
                removed["human_short_or_missing"] += 1
                continue
            human_digest = text_hash(human)
            if human_digest in seen_humans:
                removed["duplicate_human"] += 1
                continue
            seen_humans.add(human_digest)
            machines = {}
            for gen in generators:
                machine = prepared[domain][gen][source_row]["text"]
                if len(machine.split()) < min_words:
                    removed[f"{gen}:short_or_missing"] += 1
                elif text_hash(machine) in all_human_hashes:
                    removed[f"{gen}:also_human"] += 1
                else:
                    machines[gen] = machine
                    available[gen] += 1
            if not machines:
                removed["human_without_valid_machine"] += 1
                continue
            prompt = _cell(row.get("prompt", ""))
            candidates.append({"human": human, "machines": machines, "prompt": prompt,
                               "source_row": int(source_row), "source_id": str(row.get("id", source_row)),
                               "cleanup": {origin: prepared[domain][origin][source_row]
                                           for origin in ["human", *generators]}})
        candidates_by_domain[domain] = candidates
        diagnostics[domain] = {"eligible_unique_humans": len(candidates),
                               "eligible_machine_by_generator": dict(available), "removed": dict(removed)}

    rng, records, grouping_keys = np.random.default_rng(seed), [], {}
    selected_cleanup = {}
    used_texts: set[str] = set()
    quotas = {gen: n_pairs // len(generators) + (i < n_pairs % len(generators))
              for i, gen in enumerate(generators)}
    for domain, candidates in candidates_by_domain.items():
        selected_cleanup[domain] = {}
        selected = _choose_pairs(candidates, quotas, rng, used_texts | all_human_hashes)
        for candidate, gen in selected:
            human_digest = text_hash(candidate["human"])
            pair_id = f"{domain}:{candidate['source_row']}:{human_digest[:16]}"
            grouping_keys[pair_id] = ["human:" + human_digest]
            if candidate["prompt"]:
                grouping_keys[pair_id].append("prompt:" + text_hash(candidate["prompt"]))
            for label, origin, text in [(0, "human", candidate["human"]), (1, gen, candidate["machines"][gen])]:
                counters = selected_cleanup[domain].setdefault(origin, {"n_selected": 0, "n_tail_trimmed": 0,
                                                                         "whitespace_words_removed": 0})
                counters["n_selected"] += 1
                counters["n_tail_trimmed"] += int(candidate["cleanup"][origin]["changed"])
                counters["whitespace_words_removed"] += candidate["cleanup"][origin]["removed_words"]
                digest = text_hash(text)
                if digest in used_texts:
                    raise RuntimeError("Duplicate text escaped input filtering.")
                used_texts.add(digest)
                records.append({"id": f"{origin}:{digest[:24]}", "pair_id": pair_id,
                                "domain": domain, "generator": origin, "label": label,
                                "text": text, "source_row": candidate["source_row"],
                                "source_id": candidate["source_id"]})
    # Connected components link shared source prompts even when human texts differ.
    parent = {p: p for p in grouping_keys}
    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item
    owners = {}
    for pair, keys in grouping_keys.items():
        for key in keys:
            if key in owners:
                a, b = find(pair), find(owners[key])
                parent[max(a, b)] = min(a, b)
            else:
                owners[key] = pair
    for record in records:
        record["group_id"] = find(record["pair_id"])
    result = pd.DataFrame(records).sort_values(["domain", "pair_id", "label"]).reset_index(drop=True)
    manifest = {
        "source_format": "MGTBench mgt1 wide CSV", "th_bench_code_commit": TH_BENCH_COMMIT,
        "official_source": "https://drive.google.com/drive/folders/1p4iBeM4r-sUKe8TnS4DcYlxvQagcmola",
        "n_pairs_per_domain": n_pairs, "seed": seed, "min_whitespace_words": min_words,
        "clean_sentence_ends": clean_sentence_ends,
        "generators": generators, "generator_quota_per_domain": quotas, "source_files": source_files,
        "cleaning": ["Pinned TH-Bench process_spaces spacing repair",
                     ("Symmetric trailing-fragment removal to last plausible sentence-end punctuation before minimum length"
                      if clean_sentence_ends else "Incomplete trailing fragments retained (sensitivity mode)"),
                     "Symmetric minimum whitespace-word count after tail handling",
                     "Global casefolded whitespace-normalized exact text deduplication",
                     "Machines identical to any source human excluded", "No attack/humanization or fixed-word truncation",
                     "No phrase/markup/refusal stripping: these remain measurable and potentially confounding"],
        "cleaning_status": "Locally prepared subset; NOT an officially released TH-Bench clean snapshot",
        "split_unit": "group_id: connected shared-source-prompt/human-text groups; human/machine pairs stay together",
        "limitations": ["Near-duplicate detection beyond case/whitespace is not performed",
                        "No independent English-language verification; punctuation heuristic does not prove grammatical completeness",
                        "Text length, prompt reuse and original corpus artifacts can confound classification",
                        "These historical generators do not establish generalization to current LLMs"],
        "diagnostics": diagnostics,
        "cleanup_diagnostics_all_source_rows": cleanup_diagnostics,
        "cleanup_diagnostics_selected": selected_cleanup,
        "counts": result.groupby(["domain", "label", "generator"]).size().rename("n").reset_index().to_dict("records"),
        "n_rows": len(result), "n_groups": result["group_id"].nunique(),
    }
    provenance_path = data_dir / "SOURCE.json"
    if provenance_path.is_file():
        manifest["download_provenance"] = json.loads(provenance_path.read_text(encoding="utf-8"))
    else:
        manifest["download_provenance"] = {"method": "user-provided local files", "official_bytes_verified": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_json(output_dir / "samples.jsonl", orient="records", lines=True, force_ascii=False)
    manifest["samples_sha256"] = sha256_file(output_dir / "samples.jsonl")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result, manifest
