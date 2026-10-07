"""Compare evaluation runs side by side as a Markdown table.

    python -m eval.compare eval/results/<run-a>/summary.json eval/results/<run-b>/summary.json

Useful for comparing chunking settings: deploy with different
`-c chunk_max_tokens=...`, re-ingest, run the evaluation with a matching --label,
then compare the runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import render_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries", nargs="+", type=Path, help="summary.json files to compare")
    args = parser.parse_args(argv)

    docs = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    print(render_comparison(docs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
