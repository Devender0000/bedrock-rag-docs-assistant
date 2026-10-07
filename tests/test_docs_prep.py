import io
import json
import tarfile
from pathlib import Path

import pytest

from ingestion.docs_prep import (
    build_metadata,
    clean_markdown,
    extract_title,
    inline_code_includes,
    prepare_corpus,
    prepare_document,
    should_include,
    source_url,
)
from ingestion.ingest import load_from_tarball, plan_upload, write_prepared

SOURCES = {"docs_src/first_steps/tutorial001.py": "from fastapi import FastAPI\n\napp = FastAPI()\n"}


def test_new_style_include_becomes_fenced_code():
    text = "Intro\n\n{* ../../docs_src/first_steps/tutorial001.py hl[3] *}\n\nOutro"
    result, unresolved = inline_code_includes(text, SOURCES)
    assert "```python\nfrom fastapi import FastAPI" in result
    assert "{*" not in result
    assert unresolved == []


def test_old_style_include_inside_existing_fence():
    text = "```Python\n{!../../docs_src/first_steps/tutorial001.py!}\n```"
    result, unresolved = inline_code_includes(text, SOURCES)
    assert result.startswith("```Python\nfrom fastapi import FastAPI")
    assert result.count("```") == 2
    assert unresolved == []


def test_unresolved_include_is_removed_and_reported():
    result, unresolved = inline_code_includes("{* ../../docs_src/missing/file.py *}", SOURCES)
    assert "{*" not in result
    assert unresolved == ["docs_src/missing/file.py"]


def test_clean_markdown_strips_comments_and_extra_blank_lines():
    text = "---\ntitle: x\n---\n# Hi\n\n\n\n<!-- hidden -->\nBody\n"
    assert clean_markdown(text) == "# Hi\n\nBody\n"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("# First Steps\n\nbody", "First Steps"),
        ("# Path Parameters { #path-parameters }\n", "Path Parameters"),
        ("no heading here", "Fallback"),
    ],
)
def test_extract_title(text, expected):
    assert extract_title(text, "Fallback") == expected


@pytest.mark.parametrize(
    "rel_path, expected",
    [
        ("index.md", "https://fastapi.tiangolo.com/"),
        ("tutorial/first-steps.md", "https://fastapi.tiangolo.com/tutorial/first-steps/"),
        ("tutorial/security/index.md", "https://fastapi.tiangolo.com/tutorial/security/"),
    ],
)
def test_source_url(rel_path, expected):
    assert source_url(rel_path) == expected


def test_should_include_filters_noise_and_non_markdown():
    assert should_include("tutorial/first-steps.md")
    assert not should_include("release-notes.md")
    assert not should_include("img/diagram.png")


def test_build_metadata_has_citation_fields():
    metadata = build_metadata("tutorial/first-steps.md", "First Steps")["metadataAttributes"]
    assert metadata == {
        "source_url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "title": "First Steps",
        "section": "tutorial",
    }


def test_prepare_document_end_to_end():
    raw = "# First Steps\n\n{* ../../docs_src/first_steps/tutorial001.py *}\n"
    doc = prepare_document("tutorial/first-steps.md", raw, SOURCES)
    assert doc.title == "First Steps"
    assert doc.url.endswith("/tutorial/first-steps/")
    assert "app = FastAPI()" in doc.text


def test_plan_upload_detects_new_changed_unchanged_and_stale():
    local = {"docs/a.md": "111", "docs/b.md": "222", "docs/c.md": "333"}
    remote = {"docs/a.md": "111", "docs/b.md": "OLD", "docs/stale.md": "999"}
    plan = plan_upload(local, remote)
    assert plan.unchanged == ["docs/a.md"]
    assert plan.to_upload == ["docs/b.md", "docs/c.md"]
    assert plan.to_delete == ["docs/stale.md"]


def _make_tarball(path: Path, files: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_tarball_to_prepared_files(tmp_path):
    tarball = tmp_path / "repo.tar.gz"
    _make_tarball(
        tarball,
        {
            "fastapi-master/docs/en/docs/tutorial/first-steps.md": (
                "# First Steps\n\n{* ../../docs_src/first_steps/tutorial001.py *}\n"
            ),
            "fastapi-master/docs/en/docs/release-notes.md": "# Release notes\n",
            "fastapi-master/docs/es/docs/index.md": "# Spanish page\n",
            "fastapi-master/docs_src/first_steps/tutorial001.py": "app = FastAPI()\n",
        },
    )
    raw_docs, sources = load_from_tarball(tarball)
    assert set(raw_docs) == {"tutorial/first-steps.md", "release-notes.md"}
    assert "docs_src/first_steps/tutorial001.py" in sources

    docs = prepare_corpus(raw_docs, sources)
    assert [d.rel_path for d in docs] == ["tutorial/first-steps.md"]

    out_dir = tmp_path / "out"
    write_prepared(docs, out_dir, "master")
    written = out_dir / "docs" / "tutorial" / "first-steps.md"
    assert "app = FastAPI()" in written.read_text()
    sidecar = json.loads(Path(f"{written}.metadata.json").read_text())
    assert sidecar["metadataAttributes"]["title"] == "First Steps"
    assert json.loads((out_dir / "manifest.json").read_text())["document_count"] == 1


def test_write_prepared_refuses_path_traversal(tmp_path):
    from ingestion.docs_prep import PreparedDoc

    evil = PreparedDoc(rel_path="../../escape.md", text="x", title="x", url="x")
    with pytest.raises(ValueError):
        write_prepared([evil], tmp_path / "out", "master")
