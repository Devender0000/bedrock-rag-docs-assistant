"""Optional LLM-as-judge grading of answers against a reference.

Keyword checks are fast but crude. A judge model reads the reference facts and
the assistant's answer and decides whether the answer is correct.
"""

from __future__ import annotations

import json
import re

JUDGE_PROMPT = """You are grading an answer from a documentation assistant.

Question: {question}

Reference facts (what a correct answer must convey): {reference}

Assistant answer: {answer}

Decide whether the assistant's answer is correct. It must convey the reference facts and must not
contradict them. Extra correct detail is fine.
Reply with only JSON in this form: {{"verdict": "CORRECT" or "INCORRECT", "reason": "<one short sentence>"}}"""

_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def build_judge_prompt(question: str, reference: str, answer: str) -> str:
    return JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)


def parse_verdict(text: str) -> dict:
    """Parse the judge's reply. Returns {'verdict': CORRECT|INCORRECT|ERROR, 'reason': str}."""
    for match in _JSON_OBJECT.finditer(text):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        verdict = str(data.get("verdict", "")).strip().upper()
        if verdict in ("CORRECT", "INCORRECT"):
            return {"verdict": verdict, "reason": str(data.get("reason", "")).strip()}
    return {"verdict": "ERROR", "reason": f"Unparseable judge reply: {text[:120]!r}"}


def make_judge(client, model_id: str):
    """Build judge(question, reference, answer) using the Bedrock Converse API."""

    def judge(question: str, reference: str, answer: str) -> dict:
        response = client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": build_judge_prompt(question, reference, answer)}],
                }
            ],
            inferenceConfig={"temperature": 0.0, "maxTokens": 300},
        )
        text = response["output"]["message"]["content"][0]["text"]
        return parse_verdict(text)

    return judge
