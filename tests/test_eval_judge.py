import pytest

from eval.judge import build_judge_prompt, make_judge, parse_verdict


def test_prompt_contains_question_reference_and_answer():
    prompt = build_judge_prompt("How do I X?", "Use Y.", "You use Z.")
    assert "How do I X?" in prompt and "Use Y." in prompt and "You use Z." in prompt
    assert '"verdict"' in prompt


@pytest.mark.parametrize(
    "text, verdict",
    [
        ('{"verdict": "CORRECT", "reason": "Matches."}', "CORRECT"),
        ('{"verdict": "INCORRECT", "reason": "Wrong function."}', "INCORRECT"),
        ('Sure! ```json\n{"verdict": "correct", "reason": "ok"}\n```', "CORRECT"),
        ('Thinking... {"note": 1} then {"verdict": "INCORRECT", "reason": "no"}', "INCORRECT"),
    ],
)
def test_parse_verdict_accepts_common_formats(text, verdict):
    assert parse_verdict(text)["verdict"] == verdict


@pytest.mark.parametrize("text", ["", "I think it is fine.", '{"verdict": "MAYBE"}', "{broken json"])
def test_parse_verdict_flags_unparseable_replies(text):
    assert parse_verdict(text)["verdict"] == "ERROR"


def test_parse_verdict_keeps_reason():
    assert parse_verdict('{"verdict": "CORRECT", "reason": " Good. "}')["reason"] == "Good."


def test_make_judge_calls_converse_and_parses_reply():
    calls = []

    class FakeBedrock:
        def converse(self, **kwargs):
            calls.append(kwargs)
            text = '{"verdict": "CORRECT", "reason": "Matches."}'
            return {"output": {"message": {"content": [{"text": text}]}}}

    judge = make_judge(FakeBedrock(), "us.amazon.nova-lite-v1:0")
    result = judge("Q?", "Reference", "Answer")

    assert result == {"verdict": "CORRECT", "reason": "Matches."}
    assert calls[0]["modelId"] == "us.amazon.nova-lite-v1:0"
    assert calls[0]["inferenceConfig"]["temperature"] == 0.0
    assert "Reference" in calls[0]["messages"][0]["content"][0]["text"]
