import json
from pathlib import Path

import pytest

from eval.metrics import ANSWERABLE, UNANSWERABLE, Question, load_questions, summarize
from eval.run_eval import (
    call_with_retries,
    check_questions,
    evaluate_question,
    main,
    render_report,
    run_evaluation,
    write_results,
)

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"
FIRST_STEPS = "https://fastapi.tiangolo.com/tutorial/first-steps/"

ANSWERABLE_Q = Question(
    "a1",
    "How do I create an app?",
    ANSWERABLE,
    expected_sources=("tutorial/first-steps/",),
    answer_keywords=(("FastAPI()",),),
    reference="Use app = FastAPI().",
)
UNANSWERABLE_Q = Question("u1", "What is the capital of Australia?", UNANSWERABLE)


def good_result(**overrides):
    result = {
        "answer": "Create it with FastAPI().",
        "covered": True,
        "blocked": False,
        "grounded": True,
        "sources": [{"title": "First Steps", "url": FIRST_STEPS, "snippet": "..."}],
        "session_id": None,
    }
    result.update(overrides)
    return result


def retrieve_ok(query):
    return [{"url": "https://fastapi.tiangolo.com/tutorial/body/"}, {"url": FIRST_STEPS}]


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 1.5
        return self.now


# ---------------------------------------------------------------- evaluate_question
def test_answerable_question_is_fully_scored():
    record = evaluate_question(ANSWERABLE_Q, retrieve_ok, lambda q: good_result(), clock=_Clock())
    assert record["rank"] == 2
    assert record["covered"] is True and record["keyword_ok"] is True and record["citation_hit"] is True
    assert record["latency_s"] == pytest.approx(1.5)
    assert record["cited_urls"] == [FIRST_STEPS]
    assert record["error"] is None


def test_unanswerable_question_skips_retrieval_and_records_refusal():
    def retrieve_fail(query):
        raise AssertionError("retrieval should not run for unanswerable questions")

    refusal = good_result(answer="I couldn't find that.", covered=False, grounded=False, sources=[])
    record = evaluate_question(UNANSWERABLE_Q, retrieve_fail, lambda q: refusal)
    assert record["covered"] is False and record["rank"] is None and record["error"] is None


def test_retrieval_only_never_calls_the_answer_function():
    def answer_fail(query):
        raise AssertionError("answer generation should be skipped")

    record = evaluate_question(ANSWERABLE_Q, retrieve_ok, answer_fail, retrieval_only=True)
    assert record["rank"] == 2 and record["covered"] is None and record["latency_s"] is None


def test_errors_are_captured_instead_of_crashing_the_run():
    def boom(query):
        raise RuntimeError("bedrock exploded")

    record = evaluate_question(ANSWERABLE_Q, retrieve_ok, boom)
    assert record["error"] == "RuntimeError: bedrock exploded"


def test_judge_verdict_is_recorded_for_covered_answerable_questions():
    def judge(question, reference, answer):
        return {"verdict": "CORRECT", "reason": "Matches reference."}

    record = evaluate_question(ANSWERABLE_Q, retrieve_ok, lambda q: good_result(), judge)
    assert record["judge_verdict"] == "CORRECT" and record["judge_reason"] == "Matches reference."


def test_judge_failure_becomes_error_verdict_not_a_crashed_question():
    def bad_judge(question, reference, answer):
        raise TimeoutError("judge timed out")

    record = evaluate_question(ANSWERABLE_Q, retrieve_ok, lambda q: good_result(), bad_judge)
    assert record["judge_verdict"] == "ERROR" and record["error"] is None


def test_judge_is_not_called_when_the_assistant_refused():
    def judge(*args):
        pytest.fail("judge should not run on refusals")

    refusal = good_result(covered=False, grounded=False, sources=[])
    record = evaluate_question(ANSWERABLE_Q, retrieve_ok, lambda q: refusal, judge)
    assert record["judge_verdict"] is None and record["keyword_ok"] is None


# ---------------------------------------------------------------- run_evaluation
def test_run_evaluation_sleeps_between_questions_and_reports_progress():
    sleeps, progress = [], []
    records = run_evaluation(
        [ANSWERABLE_Q, UNANSWERABLE_Q],
        retrieve_ok,
        lambda q: good_result(),
        delay=0.5,
        sleep=sleeps.append,
        on_record=lambda i, n, r: progress.append((i, n, r["id"])),
    )
    assert [r["id"] for r in records] == ["a1", "u1"]
    assert sleeps == [0.5]
    assert progress == [(1, 2, "a1"), (2, 2, "u1")]


# ---------------------------------------------------------------- retries
class _AwsError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


def test_call_with_retries_backs_off_on_throttling_then_succeeds():
    outcomes = [_AwsError("ThrottlingException"), _AwsError("ThrottlingException"), "ok"]
    sleeps = []

    def flaky():
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert call_with_retries(flaky, base_delay=1.0, sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]


def test_call_with_retries_does_not_retry_other_errors():
    calls = []

    def denied():
        calls.append(1)
        raise _AwsError("AccessDeniedException")

    with pytest.raises(_AwsError):
        call_with_retries(denied, sleep=lambda s: None)
    assert len(calls) == 1


def test_call_with_retries_gives_up_after_the_last_attempt():
    calls = []

    def always_throttled():
        calls.append(1)
        raise _AwsError("ThrottlingException")

    with pytest.raises(_AwsError):
        call_with_retries(always_throttled, attempts=3, sleep=lambda s: None)
    assert len(calls) == 3


# ---------------------------------------------------------------- question checks
def _make_corpus(root: Path, fragments):
    for fragment in fragments:
        page = root / "docs" / (fragment.strip("/") + ".md")
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("# Page\n")


def test_check_questions_flags_expected_sources_missing_from_corpus(tmp_path):
    _make_corpus(tmp_path, ["tutorial/first-steps/"])
    missing_q = Question("a2", "Q", ANSWERABLE, ("tutorial/missing/",), (("x",),), "ref")
    problems = check_questions([ANSWERABLE_Q, missing_q], tmp_path)
    assert problems == ["a2: expected source 'tutorial/missing/' matches no page in the corpus"]


def test_check_questions_flags_an_empty_corpus(tmp_path):
    (tmp_path / "docs").mkdir()
    problems = check_questions([ANSWERABLE_Q], tmp_path)
    assert any("No prepared documents" in p for p in problems)


def test_cli_check_questions_passes_for_the_committed_question_set(tmp_path, capsys):
    fragments = {s for q in load_questions(QUESTIONS_FILE) for s in q.expected_sources}
    _make_corpus(tmp_path, fragments)
    assert main(["--check-questions", "--docs-dir", str(tmp_path)]) == 0
    assert "OK:" in capsys.readouterr().out


def test_cli_check_questions_fails_when_the_corpus_is_missing_pages(tmp_path, capsys):
    _make_corpus(tmp_path, ["tutorial/first-steps/"])
    assert main(["--check-questions", "--docs-dir", str(tmp_path)]) == 1
    assert "matches no page in the corpus" in capsys.readouterr().out


# ---------------------------------------------------------------- reports
def _doc_and_records():
    records = [
        evaluate_question(ANSWERABLE_Q, retrieve_ok, lambda q: good_result()),
        evaluate_question(UNANSWERABLE_Q, retrieve_ok, lambda q: good_result()),
    ]
    summary = summarize(records)
    doc = {
        "label": "fixed-300-20",
        "created_at": "2026-10-07T12:00:00+00:00",
        "k": 5,
        "model_arn": "arn:model",
        "judge": False,
        **summary,
    }
    return doc, records


def test_report_lists_metrics_and_failures():
    doc, records = _doc_and_records()
    report = render_report(doc, records)
    assert "# Evaluation report: fixed-300-20" in report
    assert "| Retrieval hit rate (answerable) | 100.0% |" in report
    assert "### Answered an unanswerable question" in report
    assert "- u1: What is the capital of Australia?" in report


def test_report_says_when_there_are_no_failures():
    records = [evaluate_question(ANSWERABLE_Q, retrieve_ok, lambda q: good_result())]
    doc = {
        "label": "clean",
        "created_at": "now",
        "k": 5,
        "model_arn": "arn",
        "judge": True,
        **summarize(records),
    }
    assert "No failures." in render_report(doc, records)


def test_write_results_creates_all_three_files(tmp_path):
    doc, records = _doc_and_records()
    out_dir = tmp_path / "run"
    write_results(out_dir, doc, records)
    assert json.loads((out_dir / "summary.json").read_text())["label"] == "fixed-300-20"
    lines = (out_dir / "records.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["a1", "u1"]
    assert (out_dir / "report.md").read_text().startswith("# Evaluation report")
