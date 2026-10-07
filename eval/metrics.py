"""Pure scoring helpers for the evaluation harness (no AWS, no network)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ANSWERABLE = "answerable"
UNANSWERABLE = "unanswerable"

# (summary key, label, value kind) in display order.
METRIC_ROWS = [
    ("n_answerable", "Answerable questions", "int"),
    ("n_unanswerable", "Unanswerable questions", "int"),
    ("n_errors", "Questions that errored", "int"),
    ("retrieval_hit_rate", "Retrieval hit rate (answerable)", "pct"),
    ("mrr", "Mean reciprocal rank", "float"),
    ("keyword_accuracy", "Answer accuracy (keywords)", "pct"),
    ("judge_accuracy", "Answer accuracy (LLM judge)", "pct"),
    ("citation_hit_rate", "Citation hit rate (answered)", "pct"),
    ("grounded_rate", "Answers with sources", "pct"),
    ("false_refusal_rate", "False refusals (answerable)", "pct"),
    ("refusal_accuracy", "Correct refusals (unanswerable)", "pct"),
    ("hallucination_rate", "Answered when it should refuse", "pct"),
    ("latency_p50_s", "Latency p50", "seconds"),
    ("latency_p95_s", "Latency p95", "seconds"),
]


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    type: str
    expected_sources: tuple[str, ...] = ()
    # Each inner tuple is a group of alternatives; every group must match.
    answer_keywords: tuple[tuple[str, ...], ...] = ()
    reference: str = ""


def _normalize_groups(raw) -> tuple[tuple[str, ...], ...]:
    groups = []
    for item in raw:
        groups.append((item,) if isinstance(item, str) else tuple(item))
    return tuple(groups)


def load_questions(path) -> list[Question]:
    questions: list[Question] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            questions.append(
                Question(
                    id=item["id"],
                    question=item["question"],
                    type=item["type"],
                    expected_sources=tuple(item.get("expected_sources", ())),
                    answer_keywords=_normalize_groups(item.get("answer_keywords", ())),
                    reference=item.get("reference", ""),
                )
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from None
        except KeyError as exc:
            raise ValueError(f"{path}:{line_no}: missing field {exc}") from None
    return questions


def validate_questions(questions: list[Question]) -> list[str]:
    """Return a list of problems with the question set (empty means valid)."""
    problems: list[str] = []
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.type not in (ANSWERABLE, UNANSWERABLE):
            problems.append(f"{q.id}: type must be '{ANSWERABLE}' or '{UNANSWERABLE}'")
        elif q.type == ANSWERABLE:
            if not q.expected_sources:
                problems.append(f"{q.id}: answerable question needs expected_sources")
            if not q.answer_keywords:
                problems.append(f"{q.id}: answerable question needs answer_keywords")
            if not q.reference:
                problems.append(f"{q.id}: answerable question needs a reference answer")
        elif q.expected_sources or q.answer_keywords:
            problems.append(f"{q.id}: unanswerable question must not list sources or keywords")
    return problems


def source_matches(url: str, fragments) -> bool:
    return any(fragment in url for fragment in fragments)


def retrieval_rank(urls: list[str], fragments) -> int | None:
    """1-based rank of the first retrieved URL that matches an expected source."""
    for rank, url in enumerate(urls, start=1):
        if source_matches(url, fragments):
            return rank
    return None


def keywords_match(answer: str, groups) -> bool:
    """True if every keyword group has at least one alternative in the answer."""
    lowered = answer.lower()
    return all(any(alt.lower() in lowered for alt in group) for group in groups)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def summarize(records: list[dict]) -> dict:
    """Compute headline metrics and the ids of failing questions."""
    valid = [r for r in records if not r.get("error")]
    answerable = [r for r in valid if r["type"] == ANSWERABLE]
    unanswerable = [r for r in valid if r["type"] == UNANSWERABLE]

    hits = [r for r in answerable if r["rank"] is not None]
    mrr = (
        sum(1 / r["rank"] for r in hits) / len(answerable) if answerable else None
    )

    generated = [r for r in answerable if r["covered"] is not None]
    answered = [r for r in generated if r["covered"]]
    keyword_ok = [r for r in answered if r["keyword_ok"]]
    judged_correct = [r for r in answered if r.get("judge_verdict") == "CORRECT"]
    judged_any = [r for r in answered if r.get("judge_verdict")]
    citation_hits = [r for r in answered if r["citation_hit"]]

    all_covered = [r for r in valid if r["covered"]]
    grounded = [r for r in all_covered if r["grounded"]]

    unanswerable_generated = [r for r in unanswerable if r["covered"] is not None]
    refused = [r for r in unanswerable_generated if not r["covered"]]

    latencies = [r["latency_s"] for r in valid if r.get("latency_s") is not None]

    answer_rate = _rate(len(answered), len(generated))
    refusal_accuracy = _rate(len(refused), len(unanswerable_generated))

    def is_wrong(record: dict) -> bool:
        if record.get("judge_verdict"):
            return record["judge_verdict"] == "INCORRECT"
        return not record["keyword_ok"]

    metrics = {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_errors": len(records) - len(valid),
        "retrieval_hit_rate": _rate(len(hits), len(answerable)),
        "mrr": mrr,
        "keyword_accuracy": _rate(len(keyword_ok), len(generated)),
        "judge_accuracy": _rate(len(judged_correct), len(generated)) if judged_any else None,
        "citation_hit_rate": _rate(len(citation_hits), len(answered)),
        "grounded_rate": _rate(len(grounded), len(all_covered)),
        "false_refusal_rate": None if answer_rate is None else 1 - answer_rate,
        "refusal_accuracy": refusal_accuracy,
        "hallucination_rate": None if refusal_accuracy is None else 1 - refusal_accuracy,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
    }
    failed_ids = {
        "retrieval_miss": [r["id"] for r in answerable if r["rank"] is None],
        "false_refusal": [r["id"] for r in generated if not r["covered"]],
        "wrong_answer": [r["id"] for r in answered if is_wrong(r)],
        "no_citation_hit": [r["id"] for r in answered if not r["citation_hit"]],
        "should_have_refused": [r["id"] for r in unanswerable_generated if r["covered"]],
        "errored": [r["id"] for r in records if r.get("error")],
    }
    return {"metrics": metrics, "failed_ids": failed_ids}


def format_value(value, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "float":
        return f"{value:.2f}"
    if kind == "seconds":
        return f"{value:.2f}s"
    return str(int(value))


def format_summary(metrics: dict) -> str:
    width = max(len(label) for _, label, _ in METRIC_ROWS)
    return "\n".join(
        f"{label:<{width}}  {format_value(metrics.get(key), kind)}"
        for key, label, kind in METRIC_ROWS
    )


def render_comparison(summaries: list[dict]) -> str:
    """Markdown table comparing several runs (each has 'label' and 'metrics')."""
    header = "| Metric | " + " | ".join(s["label"] for s in summaries) + " |"
    divider = "|---|" + "---|" * len(summaries)
    rows = [
        f"| {label} | "
        + " | ".join(format_value(s["metrics"].get(key), kind) for s in summaries)
        + " |"
        for key, label, kind in METRIC_ROWS
    ]
    return "\n".join([header, divider, *rows])
