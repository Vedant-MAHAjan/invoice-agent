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


def group_rows(boxes: list[Box]) -> list[list[Box]]:
    """Group boxes into rows whose vertical centres fall within half a line height, sorted by x."""
    rows: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: (b.cy, b.x)):
        if rows:
            row = rows[-1]
            ref_h = sum(r.h for r in row) / len(row)
            if abs(b.cy - sum(r.cy for r in row) / len(row)) <= ref_h / 2:
                row.append(b)
                continue
        rows.append([b])
    return [sorted(row, key=lambda b: b.x) for row in rows]


def boxes_to_text(boxes: list[Box], gap: str = "   ", columns: bool = False) -> str:
    rows = group_rows(boxes)
    if not columns:
        return "\n".join(gap.join(b.text for b in row) for row in rows)
    return "\n".join(_emit_with_columns(rows, gap))


# Column detection: side-by-side blocks (supplier | buyer) otherwise get interleaved line by line.
SEGMENT_GAP = 0.04  # horizontal gap (fraction of page width) that separates two segments in a row
MIN_TWO_COLUMN_ROWS = 2


def _segments(row: list[Box]) -> list[list[Box]]:
    segs = [[row[0]]]
    for b in row[1:]:
        prev = segs[-1][-1]
        if b.x - (prev.x + prev.w) > SEGMENT_GAP:
            segs.append([b])
        else:
            segs[-1].append(b)
    return segs


def _span(seg: list[Box]) -> tuple[float, float]:
    return seg[0].x, max(b.x + b.w for b in seg)


def _emit_with_columns(rows: list[list[Box]], gap: str) -> list[str]:
    """Rows splitting into exactly two segments around a shared gutter form a two-column region;
    single-segment rows that sit wholly on one side extend it. Tables (3+ segments) and full-width
    lines end it. A region is emitted as its left block, then its right block."""
    out: list[str] = []
    text = lambda seg: gap.join(b.text for b in seg)  # noqa: E731
    i = 0
    while i < len(rows):
        segs = _segments(rows[i])
        if len(segs) != 2:
            out.append(gap.join(text(s) for s in segs))
            i += 1
            continue
        gutter = (_span(segs[0])[1], _span(segs[1])[0])
        left, right, two_col, j = [text(segs[0])], [text(segs[1])], 1, i + 1
        left_x0, right_x0 = [_span(segs[0])[0]], [_span(segs[1])[0]]
        while j < len(rows):
            nxt = _segments(rows[j])
            if len(nxt) == 2:
                lo, hi = max(gutter[0], _span(nxt[0])[1]), min(gutter[1], _span(nxt[1])[0])
                if lo >= hi:
                    break
                gutter = (lo, hi)
                left.append(text(nxt[0]))
                right.append(text(nxt[1]))
                left_x0.append(_span(nxt[0])[0])
                right_x0.append(_span(nxt[1])[0])
                two_col += 1
            elif len(nxt) == 1:
                x0, x1 = _span(nxt[0])
                if x0 < gutter[0] and x1 > gutter[1]:
                    break  # full-width line
                # Side = whichever block's typical left edge this line starts nearest to.
                to_left = abs(x0 - sorted(left_x0)[len(left_x0) // 2]) <= abs(x0 - sorted(right_x0)[len(right_x0) // 2])
                (left if to_left else right).append(text(nxt[0]))
                (left_x0 if to_left else right_x0).append(x0)
            else:
                break
            j += 1
        if two_col >= MIN_TWO_COLUMN_ROWS:
            out += left + [""] + right
            i = j
        else:
            out.append(gap.join(text(s) for s in segs))
            i += 1
    return out
