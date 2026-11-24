"""Canonical dataset format: JSON Lines, one Example per line.

Each public dataset (DocILE, MIDD, FATURA) gets a converter that writes this format, so the
harness, validators and calibrator only ever see one shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from invoice_agent.schema import Example, Split


def read_examples(path: str | Path, split: Split | None = None) -> Iterator[Example]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                ex = Example.model_validate_json(line)
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
            if split is None or ex.split == split:
                yield ex


def write_examples(path: str | Path, examples: Iterable[Example]) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
            n += 1
    return n
