"""Verifiche della gestione dell'ordine, dei limiti di richieste, dell'identità del modello e della provenienza dei riferimenti."""
import csv
import json

import pytest

from humanization.naturalness.groq import APIError, GroqClient, parse_judgment
from humanization.naturalness.pairwise import load_pairs, make_plan
from humanization.naturalness.pipeline import run_plan
from humanization.evaluation.naturalness import canonical_preference, summarize_orders, validate_ratings


MODEL = "qwen/qwen3.8-27b"


def pair(source="source-1"):
    return dict(source_id=source, source_group_id=source, domain="Essay", generator="Claude",
                system_a="qwen_humanize", system_b="dipper", text_a="First passage.", text_b="Second passage.")


def response(preference="A", model=MODEL):
    return dict(model=model, choices=[dict(finish_reason="stop", message=dict(content=json.dumps(
        dict(preference=preference, naturalness_A=4, naturalness_B=3))))],
        usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120))


class FakeClient(GroqClient):
    def __init__(self, responses=None, models=None):
        super().__init__(api_key="test-only-not-a-real-key")
        self.models = [MODEL] if models is None else models
        self.responses = list(responses or [response("A"), response("B")])
        self.posts = []
        self.gets = 0

    def request(self, method, endpoint, payload=None):
        if method == "GET":
            self.gets += 1
            assert endpoint == "/models"
            return {"data": [{"id": name, "active": True} for name in self.models]}
        assert endpoint == "/chat/completions"
        self.posts.append(payload)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_both_orders_have_same_texts_and_disable_thinking_explicitly():
    plan = make_plan([pair()], MODEL)
    first, second = plan["requests"]
    assert first["order"] == "AB" and second["order"] == "BA"
    before = json.loads(first["payload"]["messages"][1]["content"])
    after = json.loads(second["payload"]["messages"][1]["content"])
    assert before["passage_A"] == after["passage_B"]
    assert before["passage_B"] == after["passage_A"]
    assert first["payload"]["reasoning_effort"] == "none"
    assert first["payload"]["max_completion_tokens"] <= 200
    assert "reasoning_format" not in first["payload"]
    assert canonical_preference("A", "AB") == canonical_preference("B", "BA") == "system_a"
    with pytest.raises(ValueError, match="reasoning_effort"):
        make_plan([pair()], MODEL, reasoning_effort="hidden")


def test_load_pairs_checks_same_original_and_does_not_filter_failed_outputs(tmp_path):
    path = tmp_path / "rewrites.jsonl"
    a = dict(id="a", source_id="source-1", source_group_id="source-1", humanizer="a",
             domain="Essay", generator="Claude", status="ok", original="Original.", modified="Rewritten.")
    b = dict(a, id="b", humanizer="b", original="A different source.")
    path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    with pytest.raises(ValueError, match="same original"):
        load_pairs([path], "a", "b")
    b.update(original="Original.", status="generation_error", modified="")
    path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    with pytest.raises(ValueError, match="failed generation"):
        load_pairs([path], "a", "b")


def rewriting_rows(sources=("source-1", "source-2", "source-3")):
    return [dict(id=f"{source}-{system}", source_id=source,
                 source_group_id=source, humanizer=system, domain="Essay",
                 generator="Claude", status="ok", original=f"Original {source}.",
                 modified=f"Rewritten {source} by {system}.")
            for source in sources for system in ("a", "b")]


def write_rewrites(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("changes,reasons", [
    ({"status": "generation_error"}, ["status_not_ok"]),
    ({"modified": " \n "}, ["empty_output"]),
    ({"modified": None}, ["empty_output"]),
    ({"status": "input_too_long", "modified": ""}, ["status_not_ok", "empty_output"]),
])
def test_reported_generation_failure_preserves_selected_denominator(tmp_path, changes, reasons):
    rows = rewriting_rows()
    rows[1].update(changes)
    path = tmp_path / "rewrites.jsonl"
    write_rewrites(path, rows)
    report = {}
    pairs = load_pairs([path], "a", "b", max_pairs=3,
                       failure_policy="report", selection_report=report)
    assert {p["source_id"] for p in pairs} == {"source-2", "source-3"}
    assert report["n_selected_sources"] == report["n_candidate_sources"] == 3
    assert report["n_eligible_pairs"] == 2
    assert report["n_generation_failure_pairs"] == report["n_missing_naturalness_pairs"] == 1
    assert report["n_failed_generations"] == 1
    failed = report["failures"][0]
    assert failed["source_id"] == "source-1"
    assert failed["status_a"] == "ok" and failed["status_b"] == changes.get("status", "ok")
    assert failed["invalid_outputs"] == [dict(humanizer="b", status=changes.get("status", "ok"), reasons=reasons)]


def test_pair_limit_is_applied_before_failures_with_no_replacement(tmp_path):
    path = tmp_path / "rewrites.jsonl"
    rows = rewriting_rows()
    write_rewrites(path, rows)
    first = load_pairs([path], "a", "b", max_pairs=1)[0]["source_id"]
    for row in rows:
        if row["source_id"] == first:
            row.update(status="generation_error", modified="")
    write_rewrites(path, rows)
    report = {}
    assert load_pairs([path], "a", "b", max_pairs=1,
                      failure_policy="report", selection_report=report) == []
    assert report["selected_source_ids"] == [first]
    assert report["n_candidate_sources"] == 3 and report["n_selected_sources"] == 1
    assert report["n_eligible_pairs"] == 0 and report["n_generation_failure_pairs"] == 1
    assert report["n_failed_generations"] == 2
    assert make_plan([], MODEL)["expected_completion_requests"] == 0


@pytest.mark.parametrize("failure_policy,report,match", [
    ("silently_skip", {}, "failure_policy"),
    ("report", None, "selection_report"),
])
def test_generation_failure_reporting_requires_explicit_valid_policy(tmp_path, failure_policy, report, match):
    path = tmp_path / "rewrites.jsonl"
    write_rewrites(path, rewriting_rows())
    with pytest.raises(ValueError, match=match):
        load_pairs([path], "a", "b", failure_policy=failure_policy, selection_report=report)


def test_reporting_does_not_change_all_ok_pairs_or_frozen_plan(tmp_path):
    path = tmp_path / "rewrites.jsonl"
    write_rewrites(path, rewriting_rows())
    strict = load_pairs([path], "a", "b")
    report = {}
    reported = load_pairs([path], "a", "b", failure_policy="report", selection_report=report)
    assert strict == reported
    assert make_plan(strict, MODEL) == make_plan(reported, MODEL)
    assert report["n_selected_sources"] == report["n_eligible_pairs"] == 3
    assert report["n_generation_failure_pairs"] == 0 and report["failures"] == []


@pytest.mark.parametrize("missing_condition", [True, False])
def test_reporting_keeps_missing_conditions_and_source_mismatches_as_errors(tmp_path, missing_condition):
    path = tmp_path / "rewrites.jsonl"
    rows = rewriting_rows(("source-1",))
    if missing_condition:
        rows.pop()
    else:
        rows[1]["original"] = "Different original."
    write_rewrites(path, rows)
    with pytest.raises(ValueError, match="Missing one rewriting condition|same original"):
        load_pairs([path], "a", "b", failure_policy="report", selection_report={})


def test_dry_plan_makes_no_api_call_and_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = FakeClient()
    manifest = run_plan(make_plan([pair()], MODEL), tmp_path, client=client)
    assert manifest["status"] == "planned"
    assert client.gets == 0 and client.posts == []
    assert not (tmp_path / "judgments.jsonl").exists()


def test_unknown_model_is_rejected_before_any_generation(tmp_path):
    client = FakeClient(models=["some-other-model"])
    with pytest.raises(ValueError, match="absent.*no fallback"):
        run_plan(make_plan([pair()], MODEL), tmp_path, execute=True, client=client)
    assert client.gets == 1 and not client.posts


def test_resume_and_global_budget_keep_completed_requests_and_order_disagreement(tmp_path):
    plan = make_plan([pair()], MODEL)
    client = FakeClient(responses=[response("A"), response("A")])
    first = run_plan(plan, tmp_path, execute=True, client=client, max_requests=1)
    assert first["status"] == "partial" and first["generation_post_attempts"] == 1
    second = run_plan(plan, tmp_path, execute=True, client=client, max_requests=2)
    assert second["generation_post_attempts"] == 2 and len(client.posts) == 2
    summary = json.loads((tmp_path / "pairwise_summary.json").read_text())
    assert not summary[0]["order_consistent"] and summary[0]["preference"] is None
    run_plan(plan, tmp_path, execute=True, client=client, max_requests=2)
    assert len(client.posts) == 2
    with pytest.raises(ValueError, match="plan differs"):
        run_plan(make_plan([pair("changed")], MODEL), tmp_path)


def test_rate_limit_retries_honor_delay_and_count_toward_budget(tmp_path):
    client = FakeClient(responses=[APIError(429, "rate limited", "2"), response("A")])
    sleeps = []
    result = run_plan(make_plan([pair()], MODEL), tmp_path, execute=True, client=client,
                      max_requests=2, sleep=sleeps.append)
    assert sleeps == [2.0]
    assert result["generation_post_attempts"] == 2 and result["completed_valid_orders"] == 1
    assert result["blocked_reason"] == "request_budget_exhausted"


def test_long_retry_after_defers_without_sleep_or_unsupported_reasoning_fallback(tmp_path):
    client = FakeClient(responses=[APIError(429, "rate limited", "120")])
    result = run_plan(make_plan([pair()], MODEL), tmp_path, execute=True, client=client,
                      max_requests=6, sleep=lambda _: pytest.fail("Do not block for two minutes"))
    assert result["status"] == "partial" and len(client.posts) == 1
    assert result["blocked_reason"] == "provider_http_429"
    other = FakeClient(responses=[APIError(400, "unsupported no-thinking option")])
    result = run_plan(make_plan([pair()], MODEL), tmp_path / "unsupported", execute=True, client=other)
    assert len(other.posts) == 1
    assert other.posts[0]["reasoning_effort"] == "none"
    assert result["blocked_reason"] == "provider_http_400"


def test_invalid_model_reasoning_or_truncation_is_not_an_accepted_judgment():
    with pytest.raises(ValueError, match="different model"):
        parse_judgment(response(model="fallback"), MODEL)
    item = response()
    item["choices"][0]["message"]["reasoning"] = "Hidden thoughts"
    with pytest.raises(ValueError, match="reasoning"):
        parse_judgment(item, MODEL)
    item = response()
    item["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="truncated"):
        parse_judgment(item, MODEL)


def test_rating_validation_distinguishes_assistant_proxy_and_empty_human_forms(tmp_path):
    ratings = tmp_path / "ratings.csv"
    key = tmp_path / "key.csv"
    write_csv(ratings, [dict(review_id="N1", naturalness_1_to_5=""),
                        dict(review_id="N2", naturalness_1_to_5="")])
    write_csv(key, [dict(review_id="N1", source_id="source-1", role="qwen_humanize"),
                    dict(review_id="N2", source_id="source-1", role="dipper")])
    plan = make_plan([pair()], MODEL)
    rows = [dict(source_id="source-1", order="AB", status="ok", judgment={"preference": "A"}),
            dict(source_id="source-1", order="BA", status="ok", judgment={"preference": "B"})]
    empty = validate_ratings(plan, rows, ratings, key, label_source="human")
    assert empty["n_compared"] == 0 and empty["agreement"] is None
    write_csv(ratings, [dict(review_id="N1", naturalness_1_to_5="5"),
                        dict(review_id="N2", naturalness_1_to_5="3")])
    proxy = validate_ratings(plan, rows, ratings, key, label_source="assistant_proxy")
    assert proxy["n_compared"] == 1 and proxy["agreement"] == 1
    assert not proxy["reference_is_human"] and proxy["cohen_kappa"] is None


def test_catalog_403_reports_provider_reason_without_key_or_inference_options():
    secret = 'gsk_test_credential_for_unit_test'

    class Response:
        status_code = 403
        headers = {'Retry-After': '2'}
        text = json.dumps({'error': {'code': 'access_denied', 'message': f'Account denied for {secret}'}})

    class Session:
        def request(self, method, url, **kwargs):
            assert method == 'GET' and url.endswith('/models') and kwargs['json'] is None
            assert kwargs['headers']['Authorization'] == 'Bearer ' + secret
            assert kwargs['allow_redirects'] is False
            return Response()

    with pytest.raises(APIError) as caught:
        GroqClient(api_key=' ' + secret + '\n', session=Session()).check_model(MODEL)
    message = str(caught.value)
    assert caught.value.status == 403 and caught.value.retry_after == '2'
    assert 'access_denied' in message and 'catalog' in message
    assert 'no inference options submitted' in message and secret not in message


def test_plain_waf_403_remains_diagnosable_with_no_redirect_or_secret_echo():
    class Response:
        status_code = 403
        headers = {}
        text = '<html><body>error code: 1010; Bearer gsk_some_other_secret</body></html>'

    class Session:
        def request(self, *args, **kwargs):
            return Response()

    with pytest.raises(APIError) as caught:
        GroqClient(api_key='test-key', session=Session()).check_model(MODEL)
    assert '1010' in str(caught.value) and 'gsk_' not in str(caught.value)
