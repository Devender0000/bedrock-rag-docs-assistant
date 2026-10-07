"""Turn the FastAPI docs repository into clean, citable Markdown documents.

Everything here is pure (no network, no AWS), so it is easy to unit test.

The FastAPI docs pull code examples in from a separate ``docs_src`` folder using
include markers. Left alone, a retrieval system would index the marker text
instead of the code, so we inline the code before indexing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

SITE_BASE_URL = "https://fastapi.tiangolo.com"

# Pages that add noise but no answerable documentation.
DEFAULT_EXCLUDES = frozenset(
    {
        "release-notes.md",
        "fastapi-people.md",
        "external-links.md",
        "newsletter.md",
        "management.md",
        "management-tasks.md",
    }
)

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".json": "json",
    ".toml": "toml",
    ".html": "html",
    ".txt": "text",
    ".md": "markdown",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".sh": "bash",
}

# Newer include syntax, alone on a line:  {* ../../docs_src/x/y.py hl[1,2] *}
_NEW_INCLUDE = re.compile(r"^[ \t]*\{\*\s*(?P<path>\S+).*?\*\}[ \t]*$", re.MULTILINE)
# Older include syntax, usually inside a code fence:  {!../../docs_src/x/y.py!}
_OLD_INCLUDE = re.compile(r"\{!\s*(?P<path>\S+?)\s*!\}")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_BLANK_RUNS = re.compile(r"\n{3,}")
_TITLE = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
_ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}\s*$")


@dataclass(frozen=True)
class PreparedDoc:
    rel_path: str
    text: str
    title: str
    url: str
    unresolved_includes: tuple[str, ...] = ()


def _lookup_key(path: str) -> str:
    """Normalize an include path to its ``docs_src/...`` key."""
    idx = path.find("docs_src/")
    return path[idx:] if idx != -1 else path


def inline_code_includes(
    text: str, sources: Mapping[str, str]
) -> tuple[str, list[str]]:
    """Replace include markers with the referenced code.

    Returns the new text and the list of include paths that could not be found.
    Unresolvable markers are removed so they don't pollute the index.
    """
    unresolved: list[str] = []

    def fenced(match: re.Match) -> str:
        key = _lookup_key(match.group("path"))
        code = sources.get(key)
        if code is None:
            unresolved.append(key)
            return ""
        lang = LANGUAGE_BY_SUFFIX.get(PurePosixPath(key).suffix, "")
        return f"```{lang}\n{code.rstrip()}\n```"

    def raw(match: re.Match) -> str:
        key = _lookup_key(match.group("path"))
        code = sources.get(key)
        if code is None:
            unresolved.append(key)
            return ""
        return code.rstrip("\n")

    text = _NEW_INCLUDE.sub(fenced, text)
    text = _OLD_INCLUDE.sub(raw, text)
    return text, unresolved


def clean_markdown(text: str) -> str:
    """Strip front matter and HTML comments, and collapse runs of blank lines."""
    text = _FRONTMATTER.sub("", text)
    text = _HTML_COMMENT.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip() + "\n"


def extract_title(text: str, fallback: str) -> str:
    match = _TITLE.search(text)
    if not match:
        return fallback
    title = _ANCHOR.sub("", match.group("title")).strip()
    return title or fallback


def source_url(rel_path: str) -> str:
    """Map a docs path to its public URL on the FastAPI site."""
    parts = list(PurePosixPath(rel_path).with_suffix("").parts)
    if parts and parts[-1] == "index":
        parts = parts[:-1]
    path = "/".join(parts)
    return f"{SITE_BASE_URL}/{path}/" if path else f"{SITE_BASE_URL}/"


def should_include(rel_path: str, excludes: Iterable[str] = DEFAULT_EXCLUDES) -> bool:
    if not rel_path.endswith(".md"):
        return False
    excludes = set(excludes)
    return rel_path not in excludes and PurePosixPath(rel_path).name not in excludes


def build_metadata(rel_path: str, title: str) -> dict:
    """Bedrock Knowledge Base metadata sidecar content for one document."""
    parts = PurePosixPath(rel_path).parts
    return {
        "metadataAttributes": {
            "source_url": source_url(rel_path),
            "title": title,
            "section": parts[0] if len(parts) > 1 else "overview",
        }
    }


def metadata_json(rel_path: str, title: str) -> str:
    return json.dumps(build_metadata(rel_path, title), indent=2)


def prepare_document(
    rel_path: str, raw_text: str, sources: Mapping[str, str]
) -> PreparedDoc:
    text, unresolved = inline_code_includes(raw_text, sources)
    text = clean_markdown(text)
    fallback = PurePosixPath(rel_path).stem.replace("-", " ").title()
    return PreparedDoc(
        rel_path=rel_path,
        text=text,
        title=extract_title(text, fallback),
        url=source_url(rel_path),
        unresolved_includes=tuple(unresolved),
    )


def prepare_corpus(
    raw_docs: Mapping[str, str],
    sources: Mapping[str, str],
    excludes: Iterable[str] = DEFAULT_EXCLUDES,
) -> list[PreparedDoc]:
    excludes = set(excludes)
    return [
        prepare_document(rel_path, raw_docs[rel_path], sources)
        for rel_path in sorted(raw_docs)
        if should_include(rel_path, excludes)
    ]
