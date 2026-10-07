"""Evaluate the deployed assistant against the question set.

Run from the project root:

    python -m eval.run_eval --check-questions        # validate the question set against your docs
    python -m eval.run_eval --label fixed-300-20     # full run (retrieval + answers)
    python -m eval.run_eval --judge --label with-judge
    python -m eval.run_eval --retrieval-only         # cheap: skips answer generation

Each run writes summary.json, records.jsonl and report.md under eval/results/.
Use `python -m eval.compare` to put several runs side by side.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eval.judge import make_judge
from eval.metrics import (
    ANSWERABLE,
    METRIC_ROWS,
    Question,
    format_summary,
    format_value,
    keywords_match,
    load_questions,
    retrieval_rank,
    source_matches,
    summarize,
    validate_questions,
)
from ingestion.docs_prep import source_url
from lambda_api import rag

DEFAULT_STACK = "BedrockRagDocsAssistant"
THROTTLE_CODES = {"ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"}

FAILURE_TITLES = {
    "retrieval_miss": "Retrieval missed the expected page",
    "false_refusal": "Refused an answerable question",
    "wrong_answer": "Answered incorrectly",
    "no_citation_hit": "Answered without citing the expected page",
    "should_have_refused": "Answered an unanswerable question",
    "errored": "Errored",
}


# ----------------------------------------------------------------------
# Running the evaluation (clients are injected so this is easy to test)
# ----------------------------------------------------------------------
def call_with_retries(fn, *, attempts: int = 5, base_delay: float = 1.0, sleep=time.sleep):
    """Call fn(), retrying with exponential backoff only on AWS throttling errors."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code not in THROTTLE_CODES or attempt == attempts - 1:
                raise
            sleep(base_delay * 2**attempt)


def evaluate_question(
    question: Question,
    retrieve_fn,
    answer_fn,
    judge_fn=None,
    *,
    retrieval_only: bool = False,
    clock=time.perf_counter,
) -> dict:
    record = {
        "id": question.id,
        "type": question.type,
        "question": question.question,
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
    try:
        if question.type == ANSWERABLE:
            urls = [item["url"] for item in retrieve_fn(question.question)]
            record["retrieved_urls"] = urls
            record["rank"] = retrieval_rank(urls, question.expected_sources)

        if retrieval_only:
            return record

        start = clock()
        result = answer_fn(question.question)
        record["latency_s"] = clock() - start
        record["covered"] = result["covered"]
        record["blocked"] = result["blocked"]
        record["grounded"] = result["grounded"]
        record["answer"] = result["answer"]
        record["cited_urls"] = [s["url"] for s in result["sources"] if s.get("url")]

        if question.type == ANSWERABLE and result["covered"]:
            record["keyword_ok"] = keywords_match(result["answer"], question.answer_keywords)
            record["citation_hit"] = any(
                source_matches(url, question.expected_sources) for url in record["cited_urls"]
            )
            if judge_fn is not None and question.reference:
                try:
                    verdict = judge_fn(question.question, question.reference, result["answer"])
                except Exception as exc:
                    verdict = {"verdict": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}
                record["judge_verdict"] = verdict["verdict"]
                record["judge_reason"] = verdict["reason"]
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def run_evaluation(
    questions: list[Question],
    retrieve_fn,
    answer_fn,
    judge_fn=None,
    *,
    retrieval_only: bool = False,
    delay: float = 0.0,
    sleep=time.sleep,
    on_record=None,
) -> list[dict]:
    records = []
    for index, question in enumerate(questions, start=1):
        record = evaluate_question(
            question, retrieve_fn, answer_fn, judge_fn, retrieval_only=retrieval_only
        )
        records.append(record)
        if on_record is not None:
            on_record(index, len(questions), record)
        if delay and index < len(questions):
            sleep(delay)
    return records


# ----------------------------------------------------------------------
# Question-set validation against the ingested corpus
# ----------------------------------------------------------------------
def corpus_urls(docs_dir: Path) -> list[str]:
    root = Path(docs_dir) / "docs"
    return [source_url(path.relative_to(root).as_posix()) for path in sorted(root.rglob("*.md"))]


def check_questions(questions: list[Question], docs_dir: Path) -> list[str]:
    """Problems with the question set, including expected sources missing from the corpus."""
    problems = validate_questions(questions)
    urls = corpus_urls(docs_dir)
    if not urls:
        problems.append(f"No prepared documents found under {docs_dir}/docs. Run the ingestion script first.")
        return problems
    for q in questions:
        for fragment in q.expected_sources:
            if not any(fragment in url for url in urls):
                problems.append(f"{q.id}: expected source '{fragment}' matches no page in the corpus")
    return problems


# ----------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------
def render_report(doc: dict, records: list[dict]) -> str:
    lines = [
        f"# Evaluation report: {doc['label']}",
        "",
        f"- Run at: {doc['created_at']}",
        f"- Retrieval depth (k): {doc['k']}",
        f"- Model: {doc['model_arn']}",
        f"- LLM judge: {'yes' if doc['judge'] else 'no'}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, label, kind in METRIC_ROWS:
        lines.append(f"| {label} | {format_value(doc['metrics'].get(key), kind)} |")

    by_id = {r["id"]: r for r in records}
    lines += ["", "## Failures", ""]
    any_failure = False
    for category, ids in doc["failed_ids"].items():
        if not ids:
            continue
        any_failure = True
        lines.append(f"### {FAILURE_TITLES[category]}")
        lines.append("")
        for question_id in ids:
            record = by_id[question_id]
            detail = f" ({record['error']})" if record["error"] else ""
            lines.append(f"- {question_id}: {record['question']}{detail}")
        lines.append("")
    if not any_failure:
        lines.append("No failures.")
    return "\n".join(lines).rstrip() + "\n"


def write_results(out_dir: Path, doc: dict, records: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    (out_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_report(doc, records), encoding="utf-8")


# ----------------------------------------------------------------------
# AWS wiring
# ----------------------------------------------------------------------
def load_config(session, stack: str, num_results: int) -> rag.RagConfig:
    """Use KNOWLEDGE_BASE_ID/MODEL_ARN/GUARDRAIL_* env vars if set, else the stack outputs."""
    env_keys = ("KNOWLEDGE_BASE_ID", "MODEL_ARN", "GUARDRAIL_ID", "GUARDRAIL_VERSION")
    if all(os.environ.get(key) for key in env_keys):
        return rag.RagConfig.from_env({**os.environ, "NUM_RESULTS": str(num_results)})

    stacks = session.client("cloudformation").describe_stacks(StackName=stack)["Stacks"]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}
    try:
        return rag.RagConfig(
            knowledge_base_id=outputs["KnowledgeBaseId"],
            model_arn=outputs["GenerationModelArn"],
            guardrail_id=outputs["GuardrailId"],
            guardrail_version=outputs["GuardrailVersion"],
            num_results=num_results,
        )
    except KeyError as missing:
        raise SystemExit(
            f"Stack {stack} has no output {missing}. Deploy the latest version of the stack."
        )


def build_aws_functions(session, config: rag.RagConfig, k: int):
    runtime = session.client("bedrock-agent-runtime")

    def retrieve_fn(query: str) -> list[dict]:
        response = call_with_retries(
            lambda: runtime.retrieve(
                knowledgeBaseId=config.knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": k}},
            )
        )
        results = []
        for item in response.get("retrievalResults", []):
            metadata = item.get("metadata") or {}
            uri = (item.get("location") or {}).get("s3Location", {}).get("uri", "")
            results.append({"url": metadata.get("source_url") or uri, "score": item.get("score")})
        return results

    def answer_fn(question: str) -> dict:
        return call_with_retries(lambda: rag.answer_question(runtime, config, question))

    return retrieve_fn, answer_fn


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument(
        "--check-questions",
        action="store_true",
        help="Validate questions against the prepared docs; no AWS calls",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("data/prepared"),
        help="Prepared docs folder (from the ingestion script)",
    )
    parser.add_argument("--label", default="run", help="Name for this run, e.g. the chunking settings used")
    parser.add_argument("--k", type=int, default=5, help="Chunks retrieved per question (default: 5)")
    parser.add_argument("--retrieval-only", action="store_true", help="Only measure retrieval; skip answer generation")
    parser.add_argument("--judge", action="store_true", help="Grade answers with an LLM judge in addition to keywords")
    parser.add_argument("--judge-model", help="Model ID for the judge (default: the answering model)")
    parser.add_argument("--limit", type=int, help="Only run the first N questions")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to wait between questions (default: 0.5)")
    parser.add_argument("--out", type=Path, default=Path("eval/results"))
    parser.add_argument("--stack", default=DEFAULT_STACK)
    parser.add_argument("--region", help="AWS region (default: from your AWS configuration)")
    return parser.parse_args(argv)


def _print_progress(index: int, total: int, record: dict) -> None:
    if record["error"]:
        status = f"ERROR {record['error']}"
    else:
        parts = []
        if record["type"] == ANSWERABLE:
            parts.append(f"rank={record['rank']}")
        if record["covered"] is not None:
            parts.append(f"covered={record['covered']}")
        if record["latency_s"] is not None:
            parts.append(f"{record['latency_s']:.1f}s")
        status = " ".join(parts)
    print(f"[{index:>3}/{total}] {record['id']} {status}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    if args.check_questions:
        problems = check_questions(questions, args.docs_dir)
        if problems:
            print("Question set problems:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print(f"OK: {len(questions)} questions are valid and all expected sources exist in the corpus.")
        return 0

    problems = validate_questions(questions)
    if problems:
        print("Question set problems:", *problems, sep="\n  - ")
        return 2

    import boto3

    session = boto3.Session(region_name=args.region)
    config = load_config(session, args.stack, args.k)
    retrieve_fn, answer_fn = build_aws_functions(session, config, args.k)

    judge_fn = None
    if args.judge and not args.retrieval_only:
        judge_model = args.judge_model or config.model_arn.rsplit("/", 1)[-1]
        judge_fn = make_judge(session.client("bedrock-runtime"), judge_model)

    print(f"Evaluating {len(questions)} questions (k={args.k}, label={args.label}) ...")
    records = run_evaluation(
        questions,
        retrieve_fn,
        answer_fn,
        judge_fn,
        retrieval_only=args.retrieval_only,
        delay=args.delay,
        on_record=_print_progress,
    )

    summary = summarize(records)
    created_at = datetime.now(timezone.utc)
    doc = {
        "label": args.label,
        "created_at": created_at.isoformat(timespec="seconds"),
        "k": args.k,
        "model_arn": config.model_arn,
        "knowledge_base_id": config.knowledge_base_id,
        "questions_file": str(args.questions),
        "judge": judge_fn is not None,
        **summary,
    }
    out_dir = args.out / f"{created_at.strftime('%Y%m%d-%H%M%S')}-{args.label}"
    write_results(out_dir, doc, records)

    print()
    print(format_summary(summary["metrics"]))
    print(f"\nResults written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
