"""Local OCR via Apple Vision (tools/ocr/vision_ocr), plus layout-preserving text reconstruction.

Vision returns text boxes; we rebuild reading order row by row so table columns stay on one line,
separated by wide gaps. That keeps more layout than MIDD's flat token stream.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

VISION_OCR = Path(__file__).resolve().parents[3] / "tools" / "ocr" / "vision_ocr"


@dataclass(frozen=True)
class Box:
    text: str
    conf: float
    x: float
    y: float
    w: float
    h: float

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def run_vision_ocr(paths: Iterable[Path], binary: Path = VISION_OCR) -> Iterator[tuple[str, list[Box], str | None]]:
    if not binary.exists():
        raise FileNotFoundError(f"{binary} missing; build it: swiftc -O tools/ocr/vision_ocr.swift -o {binary}")
    proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert proc.stdin and proc.stdout
    proc.stdin.write("".join(f"{p}\n" for p in paths))
    proc.stdin.close()
    for line in proc.stdout:
        rec = json.loads(line)
        yield rec["path"], [Box(**b) for b in rec["lines"]], rec.get("error")
    if proc.wait() != 0:
        raise RuntimeError(f"vision_ocr exited with {proc.returncode}")


def boxes_to_text(boxes: list[Box], gap: str = "   ") -> str:
    """Group boxes into rows whose vertical centres fall within half a line height, then sort by x."""
    rows: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: (b.cy, b.x)):
        if rows:
            row = rows[-1]
            ref_h = sum(r.h for r in row) / len(row)
            if abs(b.cy - sum(r.cy for r in row) / len(row)) <= ref_h / 2:
                row.append(b)
                continue
        rows.append([b])
    return "\n".join(gap.join(b.text for b in sorted(row, key=lambda b: b.x)) for row in rows)
