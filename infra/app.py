#!/usr/bin/env python3
"""CDK app entry point.

Optional context overrides, e.g.:
    cdk deploy -c generation_model_id=us.amazon.nova-pro-v1:0 -c chunk_max_tokens=500
    cdk deploy -c grounding_threshold=0.5 -c relevance_threshold=0.3   # turn the grounding checks on
"""

import os

import aws_cdk as cdk

from rag_stack import RagStack

app = cdk.App()
get = app.node.try_get_context


def threshold(name: str) -> float | None:
    """Read a guardrail threshold from CDK context. Unset or "off" leaves that check disabled."""
    value = get(name)
    if value is None or str(value).lower() in ("off", "none", "false"):
        return None
    return float(value)


RagStack(
    app,
    "BedrockRagDocsAssistant",
    embedding_model_id=get("embedding_model_id") or "amazon.titan-embed-text-v2:0",
    generation_model_id=get("generation_model_id") or "us.amazon.nova-lite-v1:0",
    chunk_max_tokens=int(get("chunk_max_tokens") or 300),
    chunk_overlap_pct=int(get("chunk_overlap_pct") or 20),
    grounding_threshold=threshold("grounding_threshold"),
    relevance_threshold=threshold("relevance_threshold"),
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=os.getenv("CDK_DEFAULT_REGION") or "us-east-1",
    ),
)

app.synth()
