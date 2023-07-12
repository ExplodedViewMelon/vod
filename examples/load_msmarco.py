from __future__ import annotations

from typing import Optional

import rich

from src import vod_datasets
from src.vod_tools import arguantic


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    language: str = "en"
    local_source_path: Optional[str] = None  # "/Users/vlievin/data/remote_data/msmarco"
    invalidate_cache: int = 0
    sample_idx: int = 42


if __name__ == "__main__":
    args = Args.parse()
    marco = vod_datasets.load_msmarco(
        language=args.language,
        invalidate_cache=bool(args.invalidate_cache),
        local_source_path=args.local_source_path,
    )
    rich.print(marco)

    rich.print("==== train ====")
    question = marco.qa_splits["train"][args.sample_idx]
    section = marco.sections[question["section_ids"][0]]
    rich.print(
        {
            "question": question,
            "section": section,
        }
    )

    rich.print("==== validation ====")
    question = marco.qa_splits["validation"][args.sample_idx]
    section = marco.sections[question["section_ids"][0]]
    rich.print(
        {
            "question": question,
            "section": section,
        }
    )
