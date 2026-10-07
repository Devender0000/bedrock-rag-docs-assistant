from pathlib import Path

import pytest

from eval.metrics import (
    ANSWERABLE,
    UNANSWERABLE,
    Question,
    format_summary,
    format_value,
    keywords_match,
    load_questions,
    percentile,
    render_comparison,
    retrieval_rank,
    source_matches,
    summarize,
    validate_questions,
)

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"


def rec(record_id, record_type=ANSWERABLE, **overrides):
    record = {
        "id": record_id,
        "type": record_type,
        "question": f"Question {record_id}",
        "retrieved_urls": [],
        "rank": None,
        "covered": None,
        "blocked": None,
        "grounded": None,
        "answer": None,
        "cited_urls": [],
        "keyword_ok": None,
        "citation_hit": None,
        "judge_verdict": None,
        "judge_reason": None,
        "latency_s": None,
        "error": None,
    }
    record.update(overrides)
    return record


def sample_records():
    return [
        rec("a1", rank=1, covered=True, grounded=True, keyword_ok=True, citation_hit=True, latency_s=1.0),
        rec("a2", rank=3, covered=True, grounded=True, keyword_ok=False, citation_hit=False, latency_s=2.0),
        rec("a3", rank=None, covered=False, grounded=False, latency_s=3.0),
        rec("a4", rank=2, covered=True, grounded=False, keyword_ok=True, citation_hit=False, latency_s=4.0),
        rec("u1", UNANSWERABLE, covered=False, grounded=False, latency_s=1.0),
        rec("u2", UNANSWERABLE, covered=True, grounded=True, latency_s=2.0),
        rec("e1", error="ClientError: boom"),
    ]


# ---------------------------------------------------------------- loading
def test_load_questions_parses_fields_and_normalizes_keyword_groups(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"id": "a1", "type": "answerable", "question": "Q?", "expected_sources": ["x/"],'
        ' "answer_keywords": ["single", ["alt1", "alt2"]], "reference": "R"}\n\n'
        '{"id": "u1", "type": "unanswerable", "question": "Off topic?"}\n'
    )
    first, second = load_questions(path)
    assert first.answer_keywords == (("single",), ("alt1", "alt2"))
    assert first.expected_sources == ("x/",)
    assert second.type == UNANSWERABLE and second.expected_sources == ()


def test_load_questions_reports_line_number_for_bad_json(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"id": "a1"\n')
    with pytest.raises(ValueError, match=":1: invalid JSON"):
        load_questions(path)


def test_load_questions_reports_missing_field(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"id": "a1", "type": "answerable"}\n')
    with pytest.raises(ValueError, match="missing field"):
        load_questions(path)


def test_validate_questions_flags_problems():
    questions = [
        Question("a1", "Q", ANSWERABLE),
        Question("a1", "Q", "weird"),
        Question("u1", "Q", UNANSWERABLE, expected_sources=("x/",)),
    ]
    problems = validate_questions(questions)
    assert any("duplicate id" in p for p in problems)
    assert any("needs expected_sources" in p for p in problems)
    assert any("needs answer_keywords" in p for p in problems)
    assert any("needs a reference" in p for p in problems)
    assert any("type must be" in p for p in problems)
    assert any("must not list sources" in p for p in problems)


def test_committed_question_set_is_valid_and_large_enough():
    questions = load_questions(QUESTIONS_FILE)
    assert validate_questions(questions) == []
    assert sum(q.type == ANSWERABLE for q in questions) >= 40
    assert sum(q.type == UNANSWERABLE for q in questions) >= 10


# ---------------------------------------------------------------- scoring helpers
def test_source_matches_and_retrieval_rank():
    urls = [
        "https://fastapi.tiangolo.com/tutorial/body/",
        "https://fastapi.tiangolo.com/tutorial/first-steps/",
    ]
    assert source_matches(urls[1], ["tutorial/first-steps/"])
    assert not source_matches(urls[0], ["tutorial/first-steps/"])
    assert retrieval_rank(urls, ["tutorial/first-steps/"]) == 2
    assert retrieval_rank(urls, ["tutorial/missing/"]) is None
    assert retrieval_rank([], ["x"]) is None


def test_keywords_match_is_case_insensitive_and_requires_every_group():
    groups = (("FastAPI()",), ("@app.get", "app.get"))
    assert keywords_match("Use fastapi() and then @APP.GET('/')", groups)
    assert not keywords_match("Use FastAPI() only", groups)
    assert keywords_match("anything", ())


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([], 50) is None


# ---------------------------------------------------------------- summarize
def test_summarize_computes_every_headline_metric():
    summary = summarize(sample_records())
    m = summary["metrics"]
    assert (m["n_answerable"], m["n_unanswerable"], m["n_errors"]) == (4, 2, 1)
    assert m["retrieval_hit_rate"] == pytest.approx(0.75)
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 2) / 4)
    assert m["false_refusal_rate"] == pytest.approx(0.25)
    assert m["keyword_accuracy"] == pytest.approx(0.5)
    assert m["judge_accuracy"] is None
    assert m["citation_hit_rate"] == pytest.approx(1 / 3)
    assert m["grounded_rate"] == pytest.approx(0.75)
    assert m["refusal_accuracy"] == pytest.approx(0.5)
    assert m["hallucination_rate"] == pytest.approx(0.5)
    assert m["latency_p50_s"] == pytest.approx(2.0)
    assert m["latency_p95_s"] == pytest.approx(3.75)


def test_summarize_lists_failing_question_ids():
    failed = summarize(sample_records())["failed_ids"]
    assert failed == {
        "retrieval_miss": ["a3"],
        "false_refusal": ["a3"],
        "wrong_answer": ["a2"],
        "no_citation_hit": ["a2", "a4"],
        "should_have_refused": ["u2"],
        "errored": ["e1"],
    }


def test_summarize_prefers_judge_verdict_over_keywords():
    records = sample_records()
    verdicts = {"a1": "INCORRECT", "a2": "CORRECT", "a4": "CORRECT"}
    for record in records:
        if record["id"] in verdicts:
            record["judge_verdict"] = verdicts[record["id"]]
    summary = summarize(records)
    assert summary["metrics"]["judge_accuracy"] == pytest.approx(2 / 4)
    assert summary["failed_ids"]["wrong_answer"] == ["a1"]


def test_summarize_retrieval_only_leaves_answer_metrics_empty():
    records = [rec("a1", rank=1), rec("a2", rank=None)]
    m = summarize(records)["metrics"]
    assert m["retrieval_hit_rate"] == pytest.approx(0.5)
    assert m["keyword_accuracy"] is None
    assert m["false_refusal_rate"] is None
    assert m["refusal_accuracy"] is None


def test_summarize_handles_no_records():
    m = summarize([])["metrics"]
    assert m["n_answerable"] == 0
    assert m["retrieval_hit_rate"] is None and m["mrr"] is None


# ---------------------------------------------------------------- formatting
@pytest.mark.parametrize(
    "value, kind, expected",
    [
        (None, "pct", "n/a"),
        (0.9333, "pct", "93.3%"),
        (0.4583, "float", "0.46"),
        (2.5, "seconds", "2.50s"),
        (7, "int", "7"),
    ],
)
def test_format_value(value, kind, expected):
    assert format_value(value, kind) == expected


def test_format_summary_lists_all_metrics():
    text = format_summary(summarize(sample_records())["metrics"])
    assert "Retrieval hit rate" in text and "75.0%" in text
    assert "Latency p95" in text


def test_render_comparison_builds_markdown_table():
    runs = [
        {"label": "fixed-300", "metrics": summarize(sample_records())["metrics"]},
        {"label": "fixed-500", "metrics": summarize([rec("a1", rank=1)])["metrics"]},
    ]
    table = render_comparison(runs).splitlines()
    assert table[0] == "| Metric | fixed-300 | fixed-500 |"
    assert table[1] == "|---|---|---|"
    retrieval_row = next(line for line in table if line.startswith("| Retrieval hit rate"))
    assert "75.0%" in retrieval_row and "100.0%" in retrieval_row
