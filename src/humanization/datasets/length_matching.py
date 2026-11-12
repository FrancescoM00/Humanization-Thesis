"""Controlla il numero esatto di parole con prefissi allineati per fonte e conclusi a fine frase.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp

from humanization.datasets.mgtbench import (ABBREVIATIONS, DOMAINS, GENERATORS, TERMINAL_BOUNDARY,
                   process_spaces, sha256_file, text_hash, trim_incomplete_tail)
from humanization.stylometry.features import load_nlp


def sentence_prefixes(text: str, tokenizer, minimum: int = 90,
                      maximum: int = 180) -> dict[int, str]:
    """Indicizza i prefissi iniziali conclusi a fine frase per numero esatto di parole lessicali.

    Si correggono soltanto gli spazi e un eventuale frammento finale incompleto.
    I tagli usano la punteggiatura, senza garantire la completezza grammaticale.
    I caratteri interni restano invariati: le parole non vengono troncate a 
    conteggi arbitrari né aggiunte.
    """
    text = trim_incomplete_tail(process_spaces(text))
    if not text:
        return {}
    doc = tokenizer(text)
    ends = np.array([token.idx + len(token.text) for token in doc if token.is_alpha])
    prefixes = {}
    for match in TERMINAL_BOUNDARY.finditer(text):
        n = int(np.searchsorted(ends, match.end(), side="right"))
        if n > maximum:
            break
        if n < minimum:
            continue
        if match.end() != len(text) and text[match.start()] == ".":
            preceding = re.search(r"([\w.]+)$", text[:match.start()])
            token = preceding.group(1) if preceding else ""
            if token.casefold() in ABBREVIATIONS or re.fullmatch(r"(?:[A-Za-z]\.)*[A-Za-z]", token):
                continue
        # Sceglie la prima chiusura valida per ogni conteggio di parole, evitando successivi
        # paragrafi di sola punteggiatura che non cambiano la lunghezza lessicale.
        prefixes.setdefault(n, text[:match.end()].rstrip())
    return prefixes


def _assign(candidates, quotas, target, rng, globally_used):
    slots = [g for g, n in quotas.items() for _ in range(n)]
    if len(candidates) < len(slots):
        raise ValueError(f"Only {len(candidates)} source rows have exact sentence-prefix matches; need {len(slots)}.")
    blocked = set()
    noise = rng.uniform(0, .001, size=(len(candidates), len(slots)))
    for _ in range(sum(len(c["matches"]) for c in candidates) + 1):
        cost = np.full(noise.shape, 1e9)
        for i, c in enumerate(candidates):
            for j, g in enumerate(slots):
                if g not in c["matches"] or (i, g) in blocked:
                    continue
                h, m, n = c["matches"][g]
                if text_hash(h) in globally_used or text_hash(m) in globally_used or text_hash(h) == text_hash(m):
                    continue
                cost[i, j] = abs(n - target) + noise[i, j]
        rows, cols = linear_sum_assignment(cost)
        if len(rows) < len(slots) or (cost[rows, cols] >= 1e9).any():
            eligible = {g: sum(g in c["matches"] for c in candidates) for g in quotas}
            raise ValueError(f"Exact matching cannot satisfy {quotas}; eligible source rows by generator {eligible}. "
                             "Explicitly widen the sentence-prefix range or reduce requested pairs; no automatic caliper fallback.")
        selected, seen, conflict = [], set(globally_used), False
        for i, j in zip(rows, cols):
            g = slots[j]
            h, m, n = candidates[i]["matches"][g]
            digests = {text_hash(h), text_hash(m)}
            if len(digests) != 2 or seen.intersection(digests):
                blocked.add((int(i), g))
                conflict = True
            else:
                seen.update(digests)
                selected.append((candidates[i], g))
        if not conflict:
            return selected
    raise RuntimeError("Exact matching duplicate resolution did not converge.")


def prepare_length_matched_dataset(raw_dir: Path | str, output_dir: Path | str,
                                   n_pairs: int = 300, seed: int = 42,
                                   minimum: int = 90, maximum: int = 180,
                                   target: int = 150, model: str = "en_core_web_sm"):
    """Salva campioni con lunghezza esattamente abbinata e provenienza; si interrompe se impossibile."""
    if not 50 <= minimum <= target <= maximum:
        raise ValueError("Require 50 <= minimum <= target <= maximum (MATTR uses 50 words).")
    if n_pairs < len(GENERATORS):
        raise ValueError("At least one source pair per generator is required.")
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    nlp = load_nlp(model)
    rng = np.random.default_rng(seed)
    quotas = {g: n_pairs // len(GENERATORS) + (i < n_pairs % len(GENERATORS))
              for i, g in enumerate(GENERATORS)}
    records, grouping_keys, globally_used = [], {}, set()
    diagnostics, source_files = {}, {}
    for domain, short in DOMAINS.items():
        path = raw_dir / f"{short}_LLMs.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = {"human", *GENERATORS}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        source_files[domain] = {"file": path.name, "sha256": sha256_file(path), "rows": len(frame)}
        candidates, available = [], Counter()
        for i, row in frame.iterrows():
            human = trim_incomplete_tail(process_spaces(row["human"]))
            human_prefix = sentence_prefixes(human, nlp.make_doc, minimum, maximum)
            if not human_prefix:
                continue
            matches = {}
            for g in GENERATORS:
                machine = trim_incomplete_tail(process_spaces(row[g]))
                machine_prefix = sentence_prefixes(machine, nlp.make_doc, minimum, maximum)
                shared = set(human_prefix).intersection(machine_prefix)
                if not shared:
                    continue
                n = min(shared, key=lambda length: (abs(length - target), length))
                h, m = human_prefix[n], machine_prefix[n]
                if text_hash(h) == text_hash(m):
                    continue
                matches[g] = (h, m, n)
                available[g] += 1
            if matches:
                candidates.append({"source_row": int(i), "source_id": str(row.get("id", i)),
                                   "original_human_hash": text_hash(human),
                                   "prompt": process_spaces(row.get("prompt", "")), "matches": matches})
        print(f"{domain}: exact prefix candidates {len(candidates)}; per generator {dict(available)}", flush=True)
        diagnostics[domain] = {"source_rows_with_exact_match": len(candidates),
                               "eligible_by_generator": dict(available)}
        selected = _assign(candidates, quotas, target, rng, globally_used)
        for candidate, g in selected:
            h, m, n = candidate["matches"][g]
            pair = f"{domain}:{candidate['source_row']}:{candidate['original_human_hash'][:16]}"
            grouping_keys[pair] = ["source-human:" + candidate["original_human_hash"]]
            if candidate["prompt"]:
                grouping_keys[pair].append("source-prompt:" + text_hash(candidate["prompt"]))
            for label, gen, text in [(0, "human", h), (1, g, m)]:
                digest = text_hash(text)
                globally_used.add(digest)
                records.append({"id": f"{gen}:{digest[:24]}", "pair_id": pair, "domain": domain,
                                "generator": gen, "label": label, "text": text,
                                "source_row": candidate["source_row"], "source_id": candidate["source_id"],
                                "matched_n_words": n})
    parent = {pair: pair for pair in grouping_keys}
    def find(pair):
        while parent[pair] != pair:
            parent[pair] = parent[parent[pair]]
            pair = parent[pair]
        return pair
    owner = {}
    for pair, keys in grouping_keys.items():
        for key in keys:
            if key in owner:
                a, b = find(pair), find(owner[key])
                parent[max(a, b)] = min(a, b)
            else:
                owner[key] = pair
    for record in records:
        record["group_id"] = find(record["pair_id"])
    samples = pd.DataFrame(records).sort_values(["domain", "pair_id", "label"]).reset_index(drop=True)
    manifest = {"method": "source-aligned exact lexical-count matching of sentence-ended initial prefixes",
                "clean_sentence_ends": True, "n_pairs_per_domain": n_pairs, "seed": seed,
                "minimum_lexical_words": minimum, "maximum_lexical_words": maximum, "target_lexical_words": target,
                "caliper": 0, "word_definition": "spaCy model tokenizer token.is_alpha, identical to feature extractor",
                "generator_quota_per_domain": quotas, "source_files": source_files,
                "diagnostics": diagnostics, "n_rows": len(samples), "n_groups": samples.group_id.nunique(),
                "source_alignment": "Each paired human and machine derive from the SAME original CSV row; one generator per row.",
                "split_unit": "group_id joins original human/source prompt; no cross-source matching edges added",
                "selection": "Minimum-cost bipartite source-row/generator-quota assignment: minimize abs(prefix_word_count-target), seeded tie breaks",
                "cleaning": ["Pinned TH-Bench spacing normalization", "Symmetric incomplete-tail removal",
                             "Initial sentence-ended prefixes with EXACT equal lexical count in each original human/machine pair",
                             "Global exact casefold/whitespace text deduplication; no arbitrary-word cuts or padding"],
                "limitations": ["Prefix cropping changes the estimand to short initial passages, not full documents",
                                "Sentence-ended prefixes are punctuation heuristics, not a grammar guarantee",
                                "Selection conditions on availability of exact matched sentence endpoints and may favor particular styles",
                                "Character count and sentence count are NOT matched: at fixed word count these encode word/sentence style",
                                "Historical generators and mirrored-source provenance retain prior limitations",
                                "No causal claim: source prompt, topic and selection effects can remain"]}
    provenance = raw_dir / "SOURCE.json"
    if provenance.exists():
        manifest["download_provenance"] = json.loads(provenance.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_json(output_dir / "samples.jsonl", orient="records", lines=True, force_ascii=False)
    manifest["samples_sha256"] = sha256_file(output_dir / "samples.jsonl")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return samples, manifest


def length_balance(features: pd.DataFrame) -> pd.DataFrame:
    """Scarto medio con segno diviso per la SD combinata e diagnostica KS per dominio e generatore."""
    rows = []
    scopes = [("pooled", "ALL", "ALL", features)]
    scopes += [("domain", d, "ALL", s) for d, s in features.groupby("domain")]
    for g in sorted(features.loc[features.label == 1, "generator"].unique()):
        pairs = set(features.loc[features.generator == g, "pair_id"])
        sub = features[features.pair_id.isin(pairs)]
        scopes.append(("generator", "ALL", g, sub))
        scopes += [("domain_generator", d, g, s) for d, s in sub.groupby("domain")]
    for scope, domain, generator, frame in scopes:
        for column in ["n_words", "n_characters", "n_sentences"]:
            h, m = [frame.loc[frame.label == y, column].to_numpy(float) for y in [0, 1]]
            sd = np.sqrt(((len(h)-1)*h.var(ddof=1)+(len(m)-1)*m.var(ddof=1))/(len(h)+len(m)-2))
            smd = (m.mean()-h.mean())/sd if sd else (0. if m.mean() == h.mean() else float("nan"))
            ks = ks_2samp(h, m)
            rows.append({"scope": scope, "domain": domain, "generator": generator, "measure": column,
                         "n_human": len(h), "n_machine": len(m), "human_mean": h.mean(), "machine_mean": m.mean(),
                         "human_median": np.median(h), "machine_median": np.median(m),
                         "standardized_mean_difference": smd, "ks_statistic": ks.statistic, "ks_p": ks.pvalue})
    return pd.DataFrame(rows)
