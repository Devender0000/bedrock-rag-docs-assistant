"""Infrastructure tests: synthesize the CDK stack and assert on the template.

Skipped automatically if aws-cdk-lib is not installed
(install with: pip install -r infra/requirements.txt).
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("aws_cdk")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infra"))

import aws_cdk as cdk  # noqa: E402
from aws_cdk.assertions import Match, Template  # noqa: E402

from rag_stack import RagStack, split_model_id  # noqa: E402


@pytest.fixture(scope="module")
def template():
    app = cdk.App()
    stack = RagStack(
        app,
        "TestStack",
        embedding_model_id="amazon.titan-embed-text-v2:0",
        generation_model_id="us.amazon.nova-lite-v1:0",
        chunk_max_tokens=300,
        chunk_overlap_pct=20,
        env=cdk.Environment(account="123456789012", region="us-east-1"),
    )
    return Template.from_stack(stack)


def test_knowledge_base_uses_s3_vectors(template):
    template.has_resource_properties(
        "AWS::Bedrock::KnowledgeBase",
        {"StorageConfiguration": Match.object_like({"Type": "S3_VECTORS"})},
    )


def test_vector_index_matches_embedding_model(template):
    template.has_resource_properties(
        "AWS::S3Vectors::Index",
        {
            "Dimension": 1024,
            "DistanceMetric": "cosine",
            "DataType": "float32",
            "MetadataConfiguration": {
                "NonFilterableMetadataKeys": [
                    "AMAZON_BEDROCK_TEXT",
                    "AMAZON_BEDROCK_METADATA",
                ]
            },
        },
    )


def test_data_source_chunking_config(template):
    template.has_resource_properties(
        "AWS::Bedrock::DataSource",
        {
            "VectorIngestionConfiguration": {
                "ChunkingConfiguration": {
                    "ChunkingStrategy": "FIXED_SIZE",
                    "FixedSizeChunkingConfiguration": {
                        "MaxTokens": 300,
                        "OverlapPercentage": 20,
                    },
                }
            }
        },
    )


def test_guardrail_has_prompt_attack_filter_and_no_grounding_check_by_default(template):
    template.has_resource_properties(
        "AWS::Bedrock::Guardrail",
        {
            "ContextualGroundingPolicyConfig": Match.absent(),
            "ContentPolicyConfig": {
                "FiltersConfig": Match.array_with(
                    [Match.object_like({"Type": "PROMPT_ATTACK", "OutputStrength": "NONE"})]
                )
            },
        },
    )


def _guardrail_template(**kwargs):
    stack = RagStack(
        cdk.App(),
        "GuardrailVariant",
        embedding_model_id="amazon.titan-embed-text-v2:0",
        generation_model_id="us.amazon.nova-lite-v1:0",
        env=cdk.Environment(account="123456789012", region="us-east-1"),
        **kwargs,
    )
    return Template.from_stack(stack)


def test_grounding_checks_can_be_switched_on():
    variant = _guardrail_template(grounding_threshold=0.5, relevance_threshold=0.3)
    variant.has_resource_properties(
        "AWS::Bedrock::Guardrail",
        {
            "ContextualGroundingPolicyConfig": {
                "FiltersConfig": Match.array_with(
                    [
                        Match.object_like({"Type": "GROUNDING", "Threshold": 0.5}),
                        Match.object_like({"Type": "RELEVANCE", "Threshold": 0.3}),
                    ]
                )
            }
        },
    )


def test_changing_thresholds_changes_the_guardrail_version_description():
    variant = _guardrail_template(grounding_threshold=0.8, relevance_threshold=None)
    variant.has_resource_properties(
        "AWS::Bedrock::Guardrail",
        {
            "ContextualGroundingPolicyConfig": {
                "FiltersConfig": [{"Type": "GROUNDING", "Threshold": 0.8}]
            }
        },
    )
    variant.has_resource_properties(
        "AWS::Bedrock::GuardrailVersion",
        {"Description": "grounding=0.8, relevance=None"},
    )


def test_ask_endpoint_requires_api_key(template):
    template.has_resource_properties(
        "AWS::ApiGateway::Method",
        {"HttpMethod": "POST", "ApiKeyRequired": True},
    )


def test_usage_plan_has_daily_quota(template):
    template.has_resource_properties(
        "AWS::ApiGateway::UsagePlan",
        {"Quota": {"Limit": 1000, "Period": "DAY"}},
    )


def test_lambda_has_required_environment(template):
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {
            "Environment": {
                "Variables": Match.object_like(
                    {
                        "KNOWLEDGE_BASE_ID": Match.any_value(),
                        "MODEL_ARN": Match.any_value(),
                        "GUARDRAIL_ID": Match.any_value(),
                        "GUARDRAIL_VERSION": Match.any_value(),
                        "NUM_RESULTS": "5",
                    }
                )
            }
        },
    )


@pytest.mark.parametrize(
    "output_name",
    [
        "DocsBucketName",
        "KnowledgeBaseId",
        "DataSourceId",
        "GenerationModelArn",
        "GuardrailId",
        "GuardrailVersion",
        "ApiUrl",
        "ApiKeyId",
    ],
)
def test_stack_exposes_outputs_needed_by_scripts(template, output_name):
    assert output_name in template.to_json()["Outputs"]


def test_docs_bucket_blocks_public_access(template):
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            }
        },
    )


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("us.amazon.nova-lite-v1:0", (True, "amazon.nova-lite-v1:0")),
        ("eu.amazon.nova-pro-v1:0", (True, "amazon.nova-pro-v1:0")),
        ("amazon.titan-text-express-v1", (False, "amazon.titan-text-express-v1")),
    ],
)
def test_split_model_id(model_id, expected):
    assert split_model_id(model_id) == expected
