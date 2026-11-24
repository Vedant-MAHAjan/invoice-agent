"""Field-level P/R/F1 and whole-document all-fields-correct rate.

Per field, per document:
  gold empty, pred empty   -> TN (not counted)
  gold empty, pred value   -> FP (hallucination)
  gold value, pred empty   -> FN (miss)
  gold value, pred match   -> TP
  gold value, pred differs -> FP + FN (wrong value is both a miss and a false claim)
Only fields in Example.labeled_fields are scored.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from invoice_agent.eval.normalize import normalize, values_match
from invoice_agent.schema import (
    HEADER_FIELDS,
    LINE_ITEM_FIELDS,
    LINE_ITEMS,
    Example,
    FieldKind,
    Invoice,
    LineItem,
)


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def compare_value(kind: FieldKind, gold: Any, pred: Any) -> Counts:
    g_empty = normalize(kind, gold) is None
    p_empty = normalize(kind, pred) is None
    if g_empty and p_empty:
        return Counts()
    if g_empty:
        return Counts(fp=1)
    if p_empty:
        return Counts(fn=1)
    return Counts(tp=1) if values_match(kind, gold, pred) else Counts(fp=1, fn=1)


def _line_similarity(g: LineItem, p: LineItem) -> int:
    return sum(
        1
        for name, kind in LINE_ITEM_FIELDS.items()
        if normalize(kind, getattr(g, name)) is not None
        and values_match(kind, getattr(g, name), getattr(p, name))
    )


def align_line_items(
    gold: tuple[LineItem, ...], pred: tuple[LineItem, ...]
) -> list[tuple[LineItem | None, LineItem | None]]:
    # Greedy best-pair matching; invoices rarely have enough rows for this to differ from Hungarian.
    pairs = sorted(
        ((_line_similarity(g, p), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    aligned: list[tuple[LineItem | None, LineItem | None]] = []
    for score, gi, pi in pairs:
        if score == 0 or gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        aligned.append((gold[gi], pred[pi]))
    aligned += [(g, None) for i, g in enumerate(gold) if i not in used_g]
    aligned += [(None, p) for i, p in enumerate(pred) if i not in used_p]
    return aligned


@dataclass
class DocResult:
    doc_id: str
    source: str
    fields: dict[str, Counts] = field(default_factory=dict)

    @property
    def all_correct(self) -> bool:
        return all(c.fp == 0 and c.fn == 0 for c in self.fields.values())

    @property
    def errors(self) -> list[str]:
        return sorted(k for k, c in self.fields.items() if c.fp or c.fn)


def score_document(example: Example, pred: Invoice) -> DocResult:
    result = DocResult(doc_id=example.doc_id, source=example.source)
    for name, kind in HEADER_FIELDS.items():
        if name in example.labeled_fields:
            result.fields[name] = compare_value(kind, getattr(example.gold, name), getattr(pred, name))

    if LINE_ITEMS in example.labeled_fields:
        per_field = {name: Counts() for name in LINE_ITEM_FIELDS}
        for g, p in align_line_items(example.gold.line_items, pred.line_items):
            for name, kind in LINE_ITEM_FIELDS.items():
                gv = getattr(g, name) if g else None
                pv = getattr(p, name) if p else None
                per_field[name].add(compare_value(kind, gv, pv))
        for name, c in per_field.items():
            result.fields[f"{LINE_ITEMS}.{name}"] = c
    return result


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(results: list[DocResult]) -> dict[str, Any]:
    per_field: dict[str, Counts] = defaultdict(Counts)
    header, lines = Counts(), Counts()
    for r in results:
        for name, c in r.fields.items():
            per_field[name].add(c)
            (lines if name.startswith(f"{LINE_ITEMS}.") else header).add(c)
    n = len(results)
    correct = sum(r.all_correct for r in results)
    lo, hi = wilson_interval(correct, n)
    return {
        "n_docs": n,
        "doc_all_fields_correct": {
            "rate": round(correct / n, 4) if n else None,
            "correct": correct,
            "ci95": [round(lo, 4), round(hi, 4)],
        },
        "micro_header": header.as_dict(),
        "micro_line_items": lines.as_dict(),
        "per_field": {k: per_field[k].as_dict() for k in sorted(per_field)},
    }
