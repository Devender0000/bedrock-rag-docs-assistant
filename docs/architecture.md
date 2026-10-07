# Architecture notes

## How a question is answered

1. The Streamlit page sends `POST /ask` with the question and an optional `session_id` to API Gateway, with the API
   key in the `x-api-key` header.
2. API Gateway enforces the key, the throttle (5 requests per second, burst 10) and the daily quota (1,000), then
   invokes the Lambda function.
3. The Lambda validates the input and calls Bedrock `RetrieveAndGenerate` against the Knowledge Base. The shared
   code is in `lambda_api/rag.py`.
4. The Knowledge Base embeds the question, fetches the top 5 chunks from the S3 Vectors index, and passes them to the
   LLM with a prompt that says to answer only from those chunks, or reply `NOT_COVERED`.
5. The Guardrail checks the input (content filters and prompt-attack detection) and the output (content filters and
   contextual grounding against the retrieved chunks).
6. The Lambda turns the reply into `{answer, covered, blocked, grounded, sources, session_id}`. Sources come from the
   citations and use the page title and URL stored as document metadata.

## How documents get indexed

1. `ingestion/ingest.py` downloads the FastAPI repository archive and reads the English docs and code samples in
   memory (nothing is extracted to disk).
2. FastAPI docs pull code in from separate files using include markers. These are replaced with the actual code so
   the index contains it.
3. Noise pages (release notes, contributor lists) are dropped. Each page gets a metadata file with its public URL,
   title and section, which Bedrock returns with citations.
4. Only changed files are uploaded to S3 (compared by MD5), then a Knowledge Base ingestion job chunks, embeds and
   indexes them.

## Design decisions

| Decision | Why | Trade-off |
|---|---|---|
| Managed Bedrock Knowledge Base | Fewer moving parts; chunking, embedding and search are handled for you | Less control over chunking strategy, no reranking step |
| S3 Vectors as the vector store | No always-on cost, which suits a portfolio project | Newer service; tuned for cost rather than lowest query latency |
| Fixed-size chunks (300 tokens, 20% overlap) | A simple, well-understood baseline that can be tuned through CDK context | May split ideas across chunks; the evaluation shows whether it matters |
| Inline code samples during ingestion | The docs reference code in separate files; without this the code would never be indexed | Include options such as highlighted line ranges are ignored, so the whole file is inlined |
| Two layers of refusal | The prompt's `NOT_COVERED` instruction handles most off-topic questions; the Guardrail's grounding check catches answers the chunks don't support | Both add latency and can occasionally refuse a valid question, which the evaluation measures as false refusals |
| API key, throttle and quota on API Gateway | A public demo cannot run up an unexpected bill | A shared key is not per-user authentication |
| `lambda_api/rag.py` shared with the evaluation | The evaluation measures the exact prompt and parsing code that production uses | The Lambda package contains a module the handler is the only user of at runtime |
| Infrastructure as code (CDK) | Repeatable deploys and a clean teardown | Requires the CDK toolchain |

## Security notes

- The docs bucket blocks all public access and requires TLS.
- IAM permissions are scoped: the Lambda can call only this Knowledge Base, the answering model and this Guardrail;
  the Knowledge Base role can read only the docs bucket and use only the embedding model and this vector index.
- The Lambda logs question length, not question text. Error details from AWS are not returned to callers.
- Secrets (the API key value) are never committed; `.env` and `.streamlit/secrets.toml` are git-ignored.

## Limitations and next steps

- Only the English FastAPI docs are indexed.
- No reranking, hybrid (keyword plus vector) search, or query rewriting. A reranker is the most likely next
  improvement if retrieval misses show up in the evaluation.
- The evaluation set has 60 questions. That is enough to compare settings but too small for precise percentages.
  Keyword matching approximates correctness; the optional LLM judge is better but uses a model that may favour
  its own style of answer unless `--judge-model` points to a different one.
- The API key is a shared secret. Per-user authentication (for example Amazon Cognito) would be the next step for
  a real deployment.
- Responses are not streamed.
- Redeploying with different chunk settings requires re-running ingestion so the index matches.
