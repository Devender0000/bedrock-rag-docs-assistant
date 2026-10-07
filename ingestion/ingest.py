"""Fetch the FastAPI docs, prepare them, upload to S3, and sync the Knowledge Base.

Run from the project root:

    python -m ingestion.ingest --dry-run          # download + prepare only, no AWS calls
    python -m ingestion.ingest                    # full run, using the CDK stack outputs
    python -m ingestion.ingest --prune --ref 0.115.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ingestion.docs_prep import PreparedDoc, metadata_json, prepare_corpus

REPO = "fastapi/fastapi"
DOCS_PREFIX = "docs/en/docs/"
SOURCES_PREFIX = "docs_src/"
SOURCE_SUFFIXES = {".py", ".js", ".json", ".toml", ".html", ".txt", ".md", ".yml", ".yaml", ".sh"}
MAX_FILE_BYTES = 1_000_000
S3_PREFIX = "docs/"
DEFAULT_STACK = "BedrockRagDocsAssistant"
TERMINAL_JOB_STATUSES = {"COMPLETE", "FAILED", "STOPPED"}


# ----------------------------------------------------------------------
# Fetching and reading the repository
# ----------------------------------------------------------------------
def download_tarball(ref: str, dest: Path) -> None:
    import requests

    url = f"https://codeload.github.com/{REPO}/tar.gz/{ref}"
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)


def load_from_tarball(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Read English docs and code samples straight from the tarball, in memory.

    Nothing is extracted to disk, so archive paths can never escape a directory.
    Returns (docs, sources): docs keyed by path under docs/en/docs/, sources
    keyed by their docs_src/... path.
    """
    docs: dict[str, str] = {}
    sources: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.size > MAX_FILE_BYTES:
                continue
            parts = member.name.split("/", 1)
            if len(parts) < 2:
                continue
            rel = parts[1]
            if rel.startswith(DOCS_PREFIX) and rel.endswith(".md"):
                target, key = docs, rel[len(DOCS_PREFIX):]
            elif rel.startswith(SOURCES_PREFIX) and PurePosixPath(rel).suffix in SOURCE_SUFFIXES:
                target, key = sources, rel
            else:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            try:
                target[key] = handle.read().decode("utf-8")
            except UnicodeDecodeError:
                continue
    return docs, sources


# ----------------------------------------------------------------------
# Writing prepared documents to disk
# ----------------------------------------------------------------------
def write_prepared(docs: list[PreparedDoc], out_dir: Path, ref: str) -> None:
    """Write each document and its metadata sidecar under out_dir/docs/."""
    docs_root = (out_dir / "docs").resolve()
    if docs_root.exists():
        shutil.rmtree(docs_root)
    docs_root.mkdir(parents=True)

    for doc in docs:
        dest = (docs_root / doc.rel_path).resolve()
        if not dest.is_relative_to(docs_root):
            raise ValueError(f"Refusing to write outside output directory: {doc.rel_path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(doc.text, encoding="utf-8")
        Path(f"{dest}.metadata.json").write_text(
            metadata_json(doc.rel_path, doc.title), encoding="utf-8"
        )

    manifest = {
        "source_repo": REPO,
        "ref": ref,
        "document_count": len(docs),
        "unresolved_includes": sum(len(d.unresolved_includes) for d in docs),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# S3 upload (only changed files)
# ----------------------------------------------------------------------
@dataclass
class UploadPlan:
    to_upload: list[str]
    to_delete: list[str]
    unchanged: list[str]


def plan_upload(local: Mapping[str, str], remote: Mapping[str, str]) -> UploadPlan:
    """Compare local MD5s with remote ETags and decide what to upload or delete."""
    to_upload = sorted(k for k, md5 in local.items() if remote.get(k) != md5)
    unchanged = sorted(k for k, md5 in local.items() if remote.get(k) == md5)
    to_delete = sorted(k for k in remote if k not in local)
    return UploadPlan(to_upload=to_upload, to_delete=to_delete, unchanged=unchanged)


def _local_files(out_dir: Path) -> dict[str, Path]:
    docs_root = out_dir / "docs"
    return {
        path.relative_to(out_dir).as_posix(): path
        for path in sorted(docs_root.rglob("*"))
        if path.is_file()
    }


def _remote_etags(s3, bucket: str) -> dict[str, str]:
    etags: dict[str, str] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            etags[obj["Key"]] = obj["ETag"].strip('"')
    return etags


def upload_directory(s3, bucket: str, out_dir: Path, *, prune: bool) -> UploadPlan:
    files = _local_files(out_dir)
    local_md5 = {key: hashlib.md5(path.read_bytes()).hexdigest() for key, path in files.items()}
    plan = plan_upload(local_md5, _remote_etags(s3, bucket))

    for key in plan.to_upload:
        content_type = "application/json" if key.endswith(".json") else "text/markdown"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=files[key].read_bytes(),
            ContentType=content_type,
        )
    if prune:
        for key in plan.to_delete:
            s3.delete_object(Bucket=bucket, Key=key)
    return plan


# ----------------------------------------------------------------------
# Knowledge Base sync
# ----------------------------------------------------------------------
def run_ingestion_job(
    agent_client, kb_id: str, ds_id: str, *, poll_seconds: int = 5, timeout_seconds: int = 1200
) -> dict:
    job = agent_client.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
        description="FastAPI docs sync",
    )["ingestionJob"]
    job_id = job["ingestionJobId"]

    deadline = time.monotonic() + timeout_seconds
    while True:
        job = agent_client.get_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id
        )["ingestionJob"]
        if job["status"] in TERMINAL_JOB_STATUSES:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"Ingestion job {job_id} still {job['status']} after {timeout_seconds}s")
        time.sleep(poll_seconds)

    if job["status"] != "COMPLETE":
        reasons = "; ".join(job.get("failureReasons", [])) or "no reason given"
        raise RuntimeError(f"Ingestion job {job_id} ended as {job['status']}: {reasons}")
    return job.get("statistics", {})


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def resolve_resources(args: argparse.Namespace, session) -> tuple[str, str, str]:
    """Return (bucket, kb_id, ds_id) from flags, falling back to CloudFormation outputs."""
    if args.bucket and args.kb_id and args.ds_id:
        return args.bucket, args.kb_id, args.ds_id

    cfn = session.client("cloudformation")
    stacks = cfn.describe_stacks(StackName=args.stack)["Stacks"]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}
    try:
        return (
            args.bucket or outputs["DocsBucketName"],
            args.kb_id or outputs["KnowledgeBaseId"],
            args.ds_id or outputs["DataSourceId"],
        )
    except KeyError as missing:
        raise SystemExit(f"Stack {args.stack} has no output {missing}. Deploy the stack first.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="master", help="Git branch or tag of the FastAPI repo (default: master)")
    parser.add_argument("--tarball", type=Path, help="Use a local tarball instead of downloading")
    parser.add_argument("--out", type=Path, default=Path("data/prepared"), help="Where prepared docs are written")
    parser.add_argument("--dry-run", action="store_true", help="Prepare docs locally; make no AWS calls")
    parser.add_argument("--skip-sync", action="store_true", help="Upload to S3 but don't start an ingestion job")
    parser.add_argument("--prune", action="store_true", help="Delete S3 objects that no longer exist locally")
    parser.add_argument("--stack", default=DEFAULT_STACK, help="CloudFormation stack name for resource lookup")
    parser.add_argument("--bucket", help="Docs bucket name (overrides stack output)")
    parser.add_argument("--kb-id", help="Knowledge Base ID (overrides stack output)")
    parser.add_argument("--ds-id", help="Data source ID (overrides stack output)")
    parser.add_argument("--region", help="AWS region (default: from your AWS configuration)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        tarball = args.tarball
        if tarball is None:
            tarball = Path(tmp) / "repo.tar.gz"
            print(f"Downloading {REPO}@{args.ref} ...")
            download_tarball(args.ref, tarball)
        raw_docs, sources = load_from_tarball(tarball)

    docs = prepare_corpus(raw_docs, sources)
    if not docs:
        print("No documents found. Check the --ref value.", file=sys.stderr)
        return 1

    write_prepared(docs, args.out, args.ref)
    unresolved = sorted({inc for d in docs for inc in d.unresolved_includes})
    print(f"Prepared {len(docs)} documents in {args.out}/docs")
    if unresolved:
        print(f"Warning: {len(unresolved)} code includes could not be resolved (e.g. {unresolved[0]})")

    if args.dry_run:
        print("Dry run: skipping AWS upload and sync.")
        return 0

    import boto3

    session = boto3.Session(region_name=args.region)
    bucket, kb_id, ds_id = resolve_resources(args, session)

    plan = upload_directory(session.client("s3"), bucket, args.out, prune=args.prune)
    print(
        f"S3 upload: {len(plan.to_upload)} uploaded, {len(plan.unchanged)} unchanged, "
        f"{len(plan.to_delete)} {'deleted' if args.prune else 'stale (use --prune to delete)'}"
    )

    if args.skip_sync:
        return 0

    print("Starting Knowledge Base ingestion job ...")
    stats = run_ingestion_job(session.client("bedrock-agent"), kb_id, ds_id)
    print("Ingestion complete:", json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
