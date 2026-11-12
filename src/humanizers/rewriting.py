"""Run fixed, non-adaptive rewriting systems and preserve unsuccessful attempts.

The identity system requires only the Python standard library. Neural models
are imported lazily, loaded one at a time, and never queried through a detector.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import time


PROTOCOL = "humanizer_fixed_single_attempt_v1"
DIPPER_TOKENIZER = "google/t5-v1_1-xxl"
DIPPER_TOKENIZER_REVISION = "3db67ab1af984cf10548a73467f0e5bca2aaaeb2"
DIPPER_WINDOW_PROTOCOLS = {"entire_passage_no_prefix", "three_sentence_context_v2"}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _jsonl(rows) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def pair_id(system: str, source_id: str) -> str:
    payload = json.dumps([system, source_id], ensure_ascii=False).encode()
    return "humanizer_" + _digest(payload)[:24]


def generation_status(text: str, generated_tokens: int, max_new_tokens: int,
                      finished_eos: bool) -> str:
    """A generation at the token limit without EOS is never a valid rewrite."""
    if not text.strip():
        return "generation_error"
    if generated_tokens >= max_new_tokens and not finished_eos:
        return "truncated"
    return "ok"


def dipper_prompt(text: str, lexical_diversity: int, order_diversity: int) -> str:
    allowed = {0, 20, 40, 60, 80, 100}
    if lexical_diversity not in allowed or order_diversity not in allowed:
        raise ValueError("DIPPER diversity must be one of 0, 20, 40, 60, 80, 100")
    # Single whole-passage window, no prefix: deliberately not the paper's
    # default three-sentence windows with cumulative rewritten context.
    normalized = " ".join(text.split())
    return (f"lexical = {100 - lexical_diversity}, "
            f"order = {100 - order_diversity} <sent> {normalized} </sent>")


def dipper_context_prompt(text: str, prefix: str, lexical_diversity: int,
                          order_diversity: int) -> str:
    """Model-card format: all previous rewritten context precedes <sent>."""
    prompt = dipper_prompt(text, lexical_diversity, order_diversity)
    prefix = " ".join(prefix.split())
    if not prefix:
        return prompt
    head, marker, tail = prompt.partition(" <sent> ")
    return f"{head} {prefix}{marker}{tail}"


@lru_cache(maxsize=1)
def _dipper_sentence_nlp():
    from stylometry.features import load_nlp
    return load_nlp()


def _dipper_sentences(text: str) -> list[str]:
    # The release recipe normalizes whitespace before segmentation. It uses
    # NLTK; this condition deliberately uses the project's installed spaCy
    # parser instead, and records that distinction in the result metadata.
    normalized = " ".join(text.split())
    return [sentence.text.strip() for sentence in _dipper_sentence_nlp()(normalized).sents
            if sentence.text.strip()]


class _DipperWindowGenerationError(RuntimeError):
    """Retain completed windows if a later model call raises an exception."""
    def __init__(self, rewrite_row: dict):
        super().__init__("DIPPER window generation failed; inspect the notebook error output.")
        self.rewrite_row = rewrite_row


def _base_row(source: dict, system: dict, seed: int) -> dict:
    original = source["machine"]
    return {
        "id": pair_id(system["name"], source["id"]),
        "source_id": source["id"],
        "source_group_id": source["source_group_id"],
        "domain": source["domain"],
        "generator": source["generator"],
        "humanizer": system["name"],
        "dataset": "MGTBench_humanizer_pilot",
        "perturbation": "identity" if system["kind"] == "identity" else "rewrite",
        "original": original,
        "modified": "",
        "status": "pending",
        "metadata": {
            "protocol": PROTOCOL,
            "model_id": system.get("model_id"),
            "revision": system.get("revision"),
            "seed": seed,
            "original_sha256": _digest(original.encode()),
            "input_tokens": None,
            "output_tokens": None,
            "finished_eos": None,
            "unchanged": False,
            "do_sample": False,
            "num_return_sequences": 1,
            "detector_feedback_used": False,
        },
    }


def _validate_system(system: dict) -> None:
    if system.get("kind") not in {"identity", "causal", "dipper"}:
        raise ValueError(f"Unsupported humanizer kind: {system.get('kind')}")
    if not isinstance(system.get("name"), str) or not system["name"]:
        raise ValueError("Every humanizer needs a nonempty name")
    if system["kind"] == "identity":
        return
    if not system.get("model_id") or not re.fullmatch("[0-9a-f]{40}", system.get("revision", "")):
        raise ValueError(f"{system['name']}: specify model_id and an immutable 40-character revision")
    if int(system.get("max_input_tokens", 0)) < 1 or int(system.get("max_new_tokens", 0)) < 1:
        raise ValueError(f"{system['name']}: positive input and generation budgets are required")
    if system["kind"] == "causal" and not str(system.get("prompt", "")).strip():
        raise ValueError("The causal humanizer requires a fixed instruction prompt")
    if system.get("tokenizer_id") and not re.fullmatch("[0-9a-f]{40}", system.get("tokenizer_revision", "")):
        raise ValueError("An external tokenizer must have an immutable tokenizer_revision")
    if system["kind"] == "dipper":
        if system.get("window_protocol", "entire_passage_no_prefix") not in DIPPER_WINDOW_PROTOCOLS:
            raise ValueError("Unsupported DIPPER window_protocol")
        dipper_prompt("Budget check.", system.get("lexical_diversity", 60),
                      system.get("order_diversity", 0))


def _cached_hf_roots() -> list[Path]:
    """Read existing caches without probing writes or changing HF globals."""
    from huggingface_hub import constants
    candidates = [os.environ.get("HF_HUB_CACHE"),
                  os.environ.get("HUMANIZATION_HF_READ_CACHE"),
                  constants.HF_HUB_CACHE,
                  Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub",
                  "/kaggle/temp/humanization_runtime/huggingface/hub",
                  "/kaggle/temp/humanization_runtime/hf/hub",
                  "/kaggle/temp/humanization_downloads_http/hub",
                  "/tmp/humanization_downloads_http/hub",
                  "/kaggle/working/.humanization_downloads_http/hub",
                  Path.home() / ".cache/huggingface/hub"]
    return list(dict.fromkeys(Path(root) for root in candidates if root))


def _load_cached_tokenizer(system: dict, roots: list[Path], tokenizer_cls):
    dipper = system["kind"] == "dipper"
    repo_id = system.get("tokenizer_id", DIPPER_TOKENIZER if dipper else system["model_id"])
    revision = system.get("tokenizer_revision", DIPPER_TOKENIZER_REVISION if dipper else system["revision"])
    kwargs = {} if dipper else {"trust_remote_code": False}
    for root in roots:
        try:
            return tokenizer_cls.from_pretrained(
                repo_id, revision=revision, cache_dir=str(root),
                local_files_only=True, **kwargs)
        except OSError:
            # In particular, DIPPER's external T5 tokenizer may be in another
            # cache from its weights. Never fall back to a network request.
            continue
    raise RuntimeError(
        f"HUMANIZATION_REUSE_CACHED_MODELS=1: tokenizer {repo_id}@{revision} "
        "is not available in the existing caches. Populate its pinned cache "
        "in a writable session before enabling strictly local model reuse.")


def _load_model(system: dict, device: str):
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        T5ForConditionalGeneration, T5Tokenizer,
    )
    selected_device = torch.device(device)
    if selected_device.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu, cuda or cuda:N")
    reuse_cached = os.environ.get("HUMANIZATION_REUSE_CACHED_MODELS") == "1"
    snapshot = None
    if reuse_cached:
        from humanizers.zero_shot import cached_weight_snapshot
        roots = _cached_hf_roots()
        snapshot = cached_weight_snapshot(system["model_id"], system["revision"], roots)
        if snapshot is None:
            raise RuntimeError(
                f"HUMANIZATION_REUSE_CACHED_MODELS=1: no complete pinned checkpoint "
                f"for {system['model_id']}@{system['revision']} in existing caches. "
                "All safetensors shards and config.json are required; no download was attempted.")
        # Check both assets before selecting/allocating a CUDA device.
        tokenizer = _load_cached_tokenizer(
            system, roots, T5Tokenizer if system["kind"] == "dipper" else AutoTokenizer)
    if selected_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Enable a Kaggle GPU before this stage.")
        # `Tensor.to('cuda')` accepts the current GPU alias, but set_device
        # requires an explicit index. Resolve it before model/device_map use.
        if selected_device.index is None:
            selected_device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(selected_device.index)
    quantized = bool(system.get("load_in_4bit", False))
    if quantized and selected_device.type != "cuda":
        raise ValueError("The 4-bit rewriting configuration requires CUDA")
    if system["kind"] == "dipper":
        if not reuse_cached:
            # F32 checkpoint shards occupy ~45.1 GB even when loading in 4-bit.
            from huggingface_hub import constants
            cache = Path(constants.HF_HUB_CACHE)
            cache.mkdir(parents=True, exist_ok=True)
            model_cache = cache / ("models--" + system["model_id"].replace("/", "--"))
            cached = sum(p.stat().st_size for p in model_cache.rglob("*")
                         if p.is_file() and not p.is_symlink()) if model_cache.exists() else 0
            minimum = max(0, 55 * 1024**3 - cached)
            if shutil.disk_usage(cache).free < minimum:
                raise RuntimeError("DIPPER needs approximately 45 GB of checkpoint files. "
                                   "Provide at least 55 GiB of free model-cache space before its first download.")
        if selected_device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(selected_device)
            if free < 8 * 1024**3:
                raise RuntimeError("DIPPER preflight requires at least 8 GiB of free GPU memory; "
                                   "this is a preliminary check, not a guarantee against OOM.")
        if not quantized:
            raise ValueError("This Kaggle DIPPER preset requires load_in_4bit=true")
        if not reuse_cached:
            tokenizer = T5Tokenizer.from_pretrained(
                system.get("tokenizer_id", DIPPER_TOKENIZER),
                revision=system.get("tokenizer_revision", DIPPER_TOKENIZER_REVISION))
        model_cls = T5ForConditionalGeneration
    else:
        if not reuse_cached:
            tokenizer = AutoTokenizer.from_pretrained(
                system.get("tokenizer_id", system["model_id"]),
                revision=system.get("tokenizer_revision", system["revision"]),
                trust_remote_code=False)
        model_cls = AutoModelForCausalLM
    kwargs = {
        "revision": system["revision"],
        "trust_remote_code": False,
        "torch_dtype": torch.float16 if selected_device.type == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
    }
    if quantized:
        kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16),
            device_map={"": selected_device.index if selected_device.index is not None else 0})
    if reuse_cached:
        kwargs["local_files_only"] = True
    model = model_cls.from_pretrained(str(snapshot) if reuse_cached else system["model_id"], **kwargs)
    if not quantized:
        model.to(selected_device)
    model.eval()
    return tokenizer, model, selected_device


def _rewrite_dipper_context(source: dict, system: dict, seed: int, tokenizer, model,
                             device) -> dict:
    """Three-sentence windows with cumulative rewritten prefix, no truncation.

    The public recipe's context placement/inverse codes are retained. Sentence
    splitting (spaCy), NF4 loading, greedy decoding and max_new_tokens per window
    are explicit experimental choices, not a claim of full paper replication.
    A partial output remains auditable but never enters successful pairs.
    """
    import torch
    row = _base_row(source, system, seed)
    metadata = row["metadata"]
    started = time.monotonic()
    sentences = _dipper_sentences(source["machine"])
    windows = [" ".join(sentences[i:i + 3]) for i in range(0, len(sentences), 3)]
    details = []
    outputs = []
    metadata.update(
        dipper_window_protocol="three_sentence_context_v2",
        dipper_sentence_segmentation="spaCy en_core_web_sm parser after whitespace normalization",
        dipper_context_policy="all prior rewritten windows; no rolling or dropped context",
        dipper_recipe_source="https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl/blob/main/README.md",
        lexical_diversity=system.get("lexical_diversity", 60),
        order_diversity=system.get("order_diversity", 0),
        window_size_sentences=3,
        n_input_sentences=len(sentences),
        n_windows_expected=len(windows),
        n_windows_generated=0,
        n_windows_completed=0,
        input_tokens=0,
        input_token_aggregation="sum of full window prompts including cumulative prefix",
        output_tokens=0,
        finished_eos=False,
        max_new_tokens_per_window=int(system["max_new_tokens"]),
        generation_budget_scope="per window; total generation budget grows with window count",
        window_generation=details,
        elapsed_seconds=0.0)
    maximum_output = int(system["max_new_tokens"])
    budget = int(system["max_input_tokens"])
    torch.manual_seed(seed)

    def finish(status):
        row["status"] = status
        row["modified"] = " ".join(outputs).strip()
        metadata["unchanged"] = (" ".join(row["modified"].split())
                                 == " ".join(source["machine"].split()))
        metadata["finished_eos"] = bool(windows and metadata["n_windows_completed"] == len(windows))
        metadata["window_processing_elapsed_seconds"] = time.monotonic() - started
        return row

    if not windows:
        metadata["error"] = "Sentence segmentation produced no nonempty window."
        return finish("generation_error")
    for index, window in enumerate(windows):
        prefix = " ".join(outputs)
        prompt = dipper_context_prompt(window, prefix, metadata["lexical_diversity"],
                                       metadata["order_diversity"])
        detail = dict(window_index=index, sentence_start=index * 3,
                      sentence_end_exclusive=min(index * 3 + 3, len(sentences)),
                      original_window_sha256=_digest(window.encode()),
                      full_prompt_sha256=_digest(prompt.encode()),
                      prefix_characters=len(prefix), input_tokens=None,
                      output_tokens=0, finished_eos=None, status="pending")
        details.append(detail)
        try:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=False,
                                add_special_tokens=True)
            input_tokens = int(encoded["input_ids"].shape[-1])
            detail["input_tokens"] = input_tokens
            metadata["input_tokens"] += input_tokens
            if input_tokens > budget:
                detail["status"] = "input_too_long"
                metadata["error"] = ("Full window with cumulative rewritten prefix exceeds the declared "
                                     "input budget; no text or context was truncated.")
                metadata["failed_window_index"] = index
                return finish("input_too_long")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated_started = time.monotonic()
            with torch.inference_mode():
                result = model.generate(
                    **encoded, do_sample=False, num_beams=1, num_return_sequences=1,
                    max_new_tokens=maximum_output,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            sequence = result[0].tolist()
            start_id = getattr(model.config, "decoder_start_token_id", None)
            generated = sequence[1:] if sequence and sequence[0] == start_id else sequence
            eos = getattr(model.generation_config, "eos_token_id", None)
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
            eos_ids.add(tokenizer.eos_token_id)
            finished = bool(generated and generated[-1] in eos_ids)
            decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
            elapsed = time.monotonic() - generated_started
            status = generation_status(decoded, len(generated), maximum_output, finished)
            detail.update(output_tokens=len(generated), finished_eos=finished,
                          status=status, elapsed_seconds=elapsed)
            metadata["elapsed_seconds"] += elapsed
            metadata["output_tokens"] += len(generated)
            metadata["n_windows_generated"] += 1
            if decoded:
                outputs.append(decoded)
            if status != "ok":
                metadata["failed_window_index"] = index
                return finish(status)
            metadata["n_windows_completed"] += 1
        except Exception as error:
            detail["status"] = "generation_error"
            metadata["error_type"] = type(error).__name__
            metadata["error"] = "Window generation failed; partial outputs are retained for audit."
            metadata["failed_window_index"] = index
            finish("generation_error")
            raise _DipperWindowGenerationError(row) from error
    return finish("ok")


def _rewrite_one(source: dict, system: dict, seed: int, tokenizer, model, device) -> dict:
    import torch
    if (system["kind"] == "dipper"
            and system.get("window_protocol", "entire_passage_no_prefix") == "three_sentence_context_v2"):
        return _rewrite_dipper_context(source, system, seed, tokenizer, model, device)
    row = _base_row(source, system, seed)
    text = source["machine"]
    if system["kind"] == "dipper":
        model_input = dipper_prompt(text, system.get("lexical_diversity", 60),
                                   system.get("order_diversity", 0))
        row["metadata"]["dipper_window_protocol"] = "entire_passage_no_prefix"
        row["metadata"]["lexical_diversity"] = system.get("lexical_diversity", 60)
        row["metadata"]["order_diversity"] = system.get("order_diversity", 0)
        add_special_tokens = True
    else:
        model_input = tokenizer.apply_chat_template(
            [{"role": "system", "content": system["prompt"]},
             {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)
        row["metadata"]["prompt_sha256"] = _digest(system["prompt"].encode())
        add_special_tokens = False
    encoded = tokenizer(model_input, return_tensors="pt", truncation=False,
                        add_special_tokens=add_special_tokens)
    input_tokens = int(encoded["input_ids"].shape[-1])
    row["metadata"]["input_tokens"] = input_tokens
    budget = int(system["max_input_tokens"])
    maximum_output = int(system["max_new_tokens"])
    model_limit = getattr(model.config, "max_position_embeddings", None)
    total_exceeds = (system["kind"] == "causal" and isinstance(model_limit, int)
                     and input_tokens + maximum_output > model_limit)
    if input_tokens > budget or total_exceeds:
        row["status"] = "input_too_long"
        row["metadata"]["error"] = "Full input exceeds the declared/model token budget; nothing was truncated."
        return row
    encoded = {key: value.to(device) for key, value in encoded.items()}
    started = time.monotonic()
    torch.manual_seed(seed)
    with torch.inference_mode():
        result = model.generate(
            **encoded, do_sample=False, num_beams=1, num_return_sequences=1,
            max_new_tokens=maximum_output,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    sequence = result[0].tolist()
    if system["kind"] == "causal":
        generated = sequence[input_tokens:]
    else:
        # Encoder-decoder output begins with decoder_start_token_id; this is
        # not one of the max_new_tokens generated tokens.
        start_id = getattr(model.config, "decoder_start_token_id", None)
        generated = sequence[1:] if sequence and sequence[0] == start_id else sequence
    eos = getattr(model.generation_config, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    eos_ids.add(tokenizer.eos_token_id)
    finished = bool(generated and generated[-1] in eos_ids)
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    row["modified"] = decoded
    row["status"] = generation_status(decoded, len(generated), maximum_output, finished)
    row["metadata"].update(
        output_tokens=len(generated), finished_eos=finished,
        elapsed_seconds=time.monotonic() - started,
        unchanged=" ".join(decoded.split()) == " ".join(text.split()))
    return row


def run_humanizers(cohort_path: Path, config_path: Path, output: Path, *,
                   device: str = "cuda", only: list[str] | None = None) -> dict:
    cohort_path, config_path, output = Path(cohort_path), Path(config_path), Path(output)
    cohort = read_jsonl(cohort_path)
    sources = [row for row in cohort if row.get("split") == "test"]
    if not sources or len({row["id"] for row in sources}) != len(sources):
        raise ValueError("Need nonempty test rows with unique ids")
    for row in sources:
        if not isinstance(row.get("machine"), str) or not row["machine"].strip():
            raise ValueError("Every test source needs a nonempty machine text")
        for field in ("source_group_id", "domain", "generator"):
            if not row.get(field):
                raise ValueError(f"Missing cohort field: {field}")
    config = json.loads(config_path.read_text())
    configured = config["humanizers"]
    names = [system.get("name") for system in configured]
    if len(set(names)) != len(names):
        raise ValueError("Humanizer names must be unique")
    if only and set(only) - set(names):
        raise ValueError(f"Unknown --only humanizer: {set(only) - set(names)}")
    # An explicit --only is also the explicit opt-in for disabled presets.
    systems = [s for s in configured if
               (s["name"] in only if only else s.get("enabled", True))]
    if not systems:
        raise ValueError("No humanizers selected")
    for system in systems:
        _validate_system(system)
    seed = int(config.get("seed", 42))
    fingerprint = {
        "protocol": PROTOCOL,
        "cohort_sha256": _digest(cohort_path.read_bytes()),
        "config_sha256": _digest(config_path.read_bytes()),
        "code_sha256": _digest(Path(__file__).read_bytes()),
        "selected_humanizers": [s["name"] for s in systems],
        "device": device,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, rows_path = output / "humanizer_run.json", output / "rewrites.jsonl"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in fingerprint.items()):
            raise ValueError("Resume configuration/cohort/device differs; choose a new output directory")
    elif rows_path.exists():
        raise ValueError("Rewrites exist without a matching run manifest; choose a new output directory")
    rows = read_jsonl(rows_path) if rows_path.exists() else []
    ids = [row["id"] for row in rows]
    expected = {pair_id(s["name"], source["id"]) for s in systems for source in sources}
    if len(set(ids)) != len(ids) or not set(ids).issubset(expected):
        raise ValueError("Existing rewrite rows contain duplicate or foreign ids")
    known = set(ids)
    created = previous["created_at"] if manifest_path.exists() else datetime.now(timezone.utc).isoformat()
    versions = {}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "sentencepiece"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = dict(fingerprint, created_at=created, expected_rewrites=len(expected),
                    n_sources=len(sources), seed=seed,
                    systems=systems, versions=versions,
                    generation_attempts_per_system_source=1,
                    resumed_completed_rewrites=len(rows))

    def save(status: str) -> None:
        manifest.update(status=status, completed_rewrites=len(rows),
                        status_counts=dict(Counter(row["status"] for row in rows)))
        _atomic(rows_path, _jsonl(rows))
        _atomic(output / "pairs.jsonl", _jsonl(row for row in rows if row["status"] == "ok"))
        _atomic(manifest_path, _json(manifest))

    save("running")
    for system in systems:
        pending = [source for source in sources if pair_id(system["name"], source["id"]) not in known]
        if not pending:
            continue
        tokenizer = model = selected_device = None
        try:
            if system["kind"] != "identity":
                tokenizer, model, selected_device = _load_model(system, device)
            for source in pending:
                if system["kind"] == "identity":
                    row = _base_row(source, system, seed)
                    row.update(modified=source["machine"], status="ok")
                    row["metadata"]["unchanged"] = True
                else:
                    try:
                        row = _rewrite_one(source, system, seed, tokenizer, model, selected_device)
                    except Exception as error:
                        row = (error.rewrite_row if isinstance(error, _DipperWindowGenerationError)
                               else _base_row(source, system, seed))
                        row["status"] = "generation_error"
                        row["metadata"].setdefault("error_type", type(error).__name__)
                        # No signed download URLs, paths or credential-bearing
                        # exception messages are copied to public result files.
                        row["metadata"].setdefault("error", "Generation failed; inspect the notebook error output.")
                        rows.append(row)
                        known.add(row["id"])
                        save("interrupted")
                        raise
                rows.append(row)
                known.add(row["id"])
                save("running")
                print(f"{system['name']}: {len(rows)}/{len(expected)} total; {row['status']}", flush=True)
        except BaseException:
            save("interrupted")
            raise
        finally:
            del model, tokenizer
            gc.collect()
            if system["kind"] != "identity":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    save("completed")
    return manifest
