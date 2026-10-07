"""CDK stack for the Bedrock RAG docs assistant.

Resources created:
  - S3 bucket for the source documents
  - S3 Vectors bucket + index (low-cost vector store, no idle charges)
  - Bedrock Knowledge Base + S3 data source (chunking, embedding, indexing)
  - Bedrock Guardrail (content filters + contextual grounding checks)
  - Lambda function that answers questions (RetrieveAndGenerate with citations)
  - API Gateway REST API with an API key and usage plan (throttling + daily quota)
"""

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigateway as apigw,
    aws_bedrock as bedrock,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3vectors as s3vectors,
)
from constructs import Construct

# Titan Text Embeddings V2 supports 256 / 512 / 1024 dimensions.
EMBEDDING_DIMENSIONS = 1024

# Cross-region inference profile IDs start with one of these prefixes.
_PROFILE_PREFIXES = ("us.", "eu.", "apac.", "us-gov.", "global.")


def split_model_id(model_id: str) -> tuple[bool, str]:
    """Return (is_inference_profile, base_model_id) for a Bedrock model ID."""
    for prefix in _PROFILE_PREFIXES:
        if model_id.startswith(prefix):
            return True, model_id[len(prefix):]
    return False, model_id


class RagStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        embedding_model_id: str,
        generation_model_id: str,
        chunk_max_tokens: int = 300,
        chunk_overlap_pct: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ------------------------------------------------------------------
        # Source documents
        # ------------------------------------------------------------------
        docs_bucket = s3.Bucket(
            self,
            "DocsBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # ------------------------------------------------------------------
        # Vector store (S3 Vectors)
        # ------------------------------------------------------------------
        vector_bucket = s3vectors.CfnVectorBucket(self, "VectorBucket")

        vector_index = s3vectors.CfnIndex(
            self,
            "VectorIndex",
            vector_bucket_arn=vector_bucket.attr_vector_bucket_arn,
            index_name="docs-index",
            data_type="float32",
            dimension=EMBEDDING_DIMENSIONS,
            distance_metric="cosine",
            # Bedrock stores chunk text and metadata as vector metadata. Mark
            # them non-filterable so they don't hit the filterable-metadata
            # size limit.
            metadata_configuration=s3vectors.CfnIndex.MetadataConfigurationProperty(
                non_filterable_metadata_keys=[
                    "AMAZON_BEDROCK_TEXT",
                    "AMAZON_BEDROCK_METADATA",
                ]
            ),
        )
        vector_index.add_resource_dependency(vector_bucket)

        # ------------------------------------------------------------------
        # Knowledge Base
        # ------------------------------------------------------------------
        embedding_model_arn = (
            f"arn:aws:bedrock:{self.region}::foundation-model/{embedding_model_id}"
        )

        kb_role = iam.Role(
            self,
            "KnowledgeBaseRole",
            assumed_by=iam.ServicePrincipal(
                "bedrock.amazonaws.com",
                conditions={
                    "StringEquals": {"aws:SourceAccount": self.account},
                    "ArnLike": {
                        "aws:SourceArn": (
                            f"arn:aws:bedrock:{self.region}:{self.account}:knowledge-base/*"
                        )
                    },
                },
            ),
        )
        docs_bucket.grant_read(kb_role)
        kb_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"], resources=[embedding_model_arn]
            )
        )
        kb_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3vectors:GetIndex",
                    "s3vectors:QueryVectors",
                    "s3vectors:PutVectors",
                    "s3vectors:GetVectors",
                    "s3vectors:DeleteVectors",
                ],
                resources=[vector_index.attr_index_arn],
            )
        )

        knowledge_base = bedrock.CfnKnowledgeBase(
            self,
            "KnowledgeBase",
            name="docs-assistant-kb",
            description="FastAPI documentation knowledge base",
            role_arn=kb_role.role_arn,
            knowledge_base_configuration=bedrock.CfnKnowledgeBase.KnowledgeBaseConfigurationProperty(
                type="VECTOR",
                vector_knowledge_base_configuration=bedrock.CfnKnowledgeBase.VectorKnowledgeBaseConfigurationProperty(
                    embedding_model_arn=embedding_model_arn,
                    embedding_model_configuration=bedrock.CfnKnowledgeBase.EmbeddingModelConfigurationProperty(
                        bedrock_embedding_model_configuration=bedrock.CfnKnowledgeBase.BedrockEmbeddingModelConfigurationProperty(
                            dimensions=EMBEDDING_DIMENSIONS,
                            embedding_data_type="FLOAT32",
                        )
                    ),
                ),
            ),
            storage_configuration=bedrock.CfnKnowledgeBase.StorageConfigurationProperty(
                type="S3_VECTORS",
                s3_vectors_configuration=bedrock.CfnKnowledgeBase.S3VectorsConfigurationProperty(
                    index_arn=vector_index.attr_index_arn
                ),
            ),
        )
        # Make sure the role's policies and the index exist before the KB is created.
        knowledge_base.node.add_dependency(kb_role)
        knowledge_base.add_resource_dependency(vector_index)

        data_source = bedrock.CfnDataSource(
            self,
            "DocsDataSource",
            knowledge_base_id=knowledge_base.attr_knowledge_base_id,
            name="fastapi-docs",
            data_source_configuration=bedrock.CfnDataSource.DataSourceConfigurationProperty(
                type="S3",
                s3_configuration=bedrock.CfnDataSource.S3DataSourceConfigurationProperty(
                    bucket_arn=docs_bucket.bucket_arn,
                    inclusion_prefixes=["docs/"],
                ),
            ),
            vector_ingestion_configuration=bedrock.CfnDataSource.VectorIngestionConfigurationProperty(
                chunking_configuration=bedrock.CfnDataSource.ChunkingConfigurationProperty(
                    chunking_strategy="FIXED_SIZE",
                    fixed_size_chunking_configuration=bedrock.CfnDataSource.FixedSizeChunkingConfigurationProperty(
                        max_tokens=chunk_max_tokens,
                        overlap_percentage=chunk_overlap_pct,
                    ),
                )
            ),
        )

        # ------------------------------------------------------------------
        # Guardrail
        # ------------------------------------------------------------------
        content_filters = [
            bedrock.CfnGuardrail.ContentFilterConfigProperty(
                type=filter_type, input_strength="HIGH", output_strength="HIGH"
            )
            for filter_type in ("HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT")
        ]
        # Prompt-attack filtering applies to inputs only.
        content_filters.append(
            bedrock.CfnGuardrail.ContentFilterConfigProperty(
                type="PROMPT_ATTACK", input_strength="HIGH", output_strength="NONE"
            )
        )

        guardrail = bedrock.CfnGuardrail(
            self,
            "Guardrail",
            name="docs-assistant-guardrail",
            description="Content filters and grounding checks for the docs assistant",
            blocked_input_messaging="Sorry, I can't help with that request.",
            blocked_outputs_messaging=(
                "Sorry, I couldn't produce a reliable answer from the documentation."
            ),
            content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(
                filters_config=content_filters
            ),
            contextual_grounding_policy_config=bedrock.CfnGuardrail.ContextualGroundingPolicyConfigProperty(
                filters_config=[
                    bedrock.CfnGuardrail.ContextualGroundingFilterConfigProperty(
                        type="GROUNDING", threshold=0.7
                    ),
                    bedrock.CfnGuardrail.ContextualGroundingFilterConfigProperty(
                        type="RELEVANCE", threshold=0.5
                    ),
                ]
            ),
        )
        guardrail_version = bedrock.CfnGuardrailVersion(
            self,
            "GuardrailVersionResource",
            guardrail_identifier=guardrail.attr_guardrail_id,
            description="Initial version",
        )

        # ------------------------------------------------------------------
        # Query Lambda
        # ------------------------------------------------------------------
        is_profile, base_model_id = split_model_id(generation_model_id)
        if is_profile:
            generation_model_arn = (
                f"arn:aws:bedrock:{self.region}:{self.account}:"
                f"inference-profile/{generation_model_id}"
            )
        else:
            generation_model_arn = (
                f"arn:aws:bedrock:{self.region}::foundation-model/{generation_model_id}"
            )

        ask_function = _lambda.Function(
            self,
            "AskFunction",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=_lambda.Code.from_asset(
                str(Path(__file__).resolve().parent.parent / "lambda_api")
            ),
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "KNOWLEDGE_BASE_ID": knowledge_base.attr_knowledge_base_id,
                "MODEL_ARN": generation_model_arn,
                "GUARDRAIL_ID": guardrail.attr_guardrail_id,
                "GUARDRAIL_VERSION": guardrail_version.attr_version,
                "NUM_RESULTS": "5",
            },
            log_group=logs.LogGroup(
                self,
                "AskFunctionLogs",
                retention=logs.RetentionDays.ONE_MONTH,
                removal_policy=RemovalPolicy.DESTROY,
            ),
        )

        ask_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:Retrieve", "bedrock:RetrieveAndGenerate"],
                resources=[knowledge_base.attr_knowledge_base_arn],
            )
        )
        model_resources = [
            f"arn:aws:bedrock:*::foundation-model/{base_model_id}",
        ]
        model_actions = ["bedrock:InvokeModel"]
        if is_profile:
            model_resources.append(generation_model_arn)
            model_actions.append("bedrock:GetInferenceProfile")
        ask_function.add_to_role_policy(
            iam.PolicyStatement(actions=model_actions, resources=model_resources)
        )
        ask_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:ApplyGuardrail"],
                resources=[guardrail.attr_guardrail_arn],
            )
        )

        # ------------------------------------------------------------------
        # API Gateway: API key + usage plan keep the public endpoint cheap and safe
        # ------------------------------------------------------------------
        api = apigw.RestApi(
            self,
            "Api",
            rest_api_name="docs-assistant-api",
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=5,
                throttling_burst_limit=10,
            ),
        )
        ask_resource = api.root.add_resource("ask")
        ask_resource.add_method(
            "POST", apigw.LambdaIntegration(ask_function), api_key_required=True
        )

        api_key = api.add_api_key("ApiKey")
        usage_plan = api.add_usage_plan(
            "UsagePlan",
            throttle=apigw.ThrottleSettings(rate_limit=5, burst_limit=10),
            quota=apigw.QuotaSettings(limit=1000, period=apigw.Period.DAY),
        )
        usage_plan.add_api_key(api_key)
        usage_plan.add_api_stage(stage=api.deployment_stage)

        # ------------------------------------------------------------------
        # Outputs used by the ingestion script, UI, and evaluation harness
        # ------------------------------------------------------------------
        CfnOutput(self, "DocsBucketName", value=docs_bucket.bucket_name)
        CfnOutput(self, "KnowledgeBaseId", value=knowledge_base.attr_knowledge_base_id)
        CfnOutput(self, "DataSourceId", value=data_source.attr_data_source_id)
        CfnOutput(self, "GenerationModelArn", value=generation_model_arn)
        CfnOutput(self, "GuardrailId", value=guardrail.attr_guardrail_id)
        CfnOutput(self, "GuardrailVersion", value=guardrail_version.attr_version)
        CfnOutput(self, "ApiUrl", value=api.url_for_path("/ask"))
        CfnOutput(self, "ApiKeyId", value=api_key.key_id)
