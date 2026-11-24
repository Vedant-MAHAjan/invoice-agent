"""Convert MIDD (Zenodo 5113009, CC-BY-4.0) IOB CSVs to the canonical JSONL format.

    python -m invoice_agent.data.midd data/raw/midd/extracted/IOB data/canonical/midd.jsonl

MIDD ships token text only (no page images), 7 value fields, no line items. Each CSV is one
document as the authors annotated it. Exact-content duplicates (e.g. "594 (1)" vs "594") are
dropped. Dates on these Indian GST invoices are day-first. A field with no span in a document
is excluded from that document's labeled_fields, so annotation gaps aren't scored as hallucinations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from invoice_agent.data.loader import write_examples
from invoice_agent.eval.normalize import to_date, to_decimal
from invoice_agent.schema import Example, Invoice, Split

TAG_TO_FIELD = {
    "SUPP_N": "vendor_name",
    "SUPP_G": "vendor_tax_id",
    "BUY_N": "buyer_name",
    "BUY_G": "buyer_tax_id",
    "INV_NO": "invoice_number",
    "INV_DT": "invoice_date",
    "GT_AMT": "total",
}
_PAREN_COPY = re.compile(r"\s*\(\d+\)")


@dataclass
class Stats:
    files: int = 0
    duplicates: int = 0
    written: int = 0
    multi_span: Counter = field(default_factory=Counter)
    missing: Counter = field(default_factory=Counter)
    unparseable: Counter = field(default_factory=Counter)


def read_tokens(path: Path) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        out = []
        for row in reader:
            if not row or not row[0].strip():
                continue
            tag = row[1].strip() if len(row) > 1 else "O"
            out.append((row[0].strip(), tag if tag[:2] in ("B-", "I-") else "O"))
        return out


def spans(tokens: list[tuple[str, str]]) -> dict[str, list[str]]:
    """IOB -> {entity: [span text, ...]}. A stray I- with no open span starts a new one."""
    found: dict[str, list[str]] = {}
    cur_ent, cur_toks = None, []

    def close():
        if cur_ent and cur_toks:
            found.setdefault(cur_ent, []).append(" ".join(cur_toks))

    for text, tag in tokens:
        prefix, _, ent = tag.partition("-")
        if tag == "O" or (prefix == "I" and ent != cur_ent):
            close()
            cur_ent, cur_toks = (ent, [text]) if prefix == "I" else (None, [])
        elif prefix == "B":
            close()
            cur_ent, cur_toks = ent, [text]
        else:
            cur_toks.append(text)
    close()
    return found


def split_for(key: str) -> Split:
    return Split.DEV if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 2 == 0 else Split.TEST


def convert(root: Path, stats: Stats) -> list[Example]:
    files = sorted(root.glob("Layout*/*.csv"), key=lambda p: (p.parent.name, len(p.name), p.name))
    seen: set[str] = set()
    examples = []
    for path in files:
        stats.files += 1
        tokens = read_tokens(path)
        content = hashlib.sha1("\x1f".join(t for t, _ in tokens).encode()).hexdigest()
        if content in seen:
            stats.duplicates += 1
            continue
        seen.add(content)

        values: dict = {}
        for ent, field_name in TAG_TO_FIELD.items():
            found = spans(tokens).get(ent, [])
            if not found:
                stats.missing[field_name] += 1
                continue
            if len(set(found)) > 1:
                stats.multi_span[field_name] += 1
            raw = found[0]
            if field_name == "invoice_date":
                parsed = to_date(raw, dayfirst=True)
            elif field_name == "total":
                parsed = to_decimal(raw)
            else:
                parsed = raw
            if parsed is None:
                stats.unparseable[field_name] += 1
                continue
            values[field_name] = parsed

        layout = path.parent.name.lower()
        stem = _PAREN_COPY.sub("", path.stem)
        doc_id = f"midd/{layout}/{stem}"
        examples.append(
            Example(
                doc_id=doc_id,
                source=f"midd-{layout}",
                split=split_for(content),
                ocr_text=" ".join(t for t, _ in tokens),
                # Unlabeled fields are annotation gaps on GST invoices, not absent fields.
                labeled_fields=frozenset(values),
                gold=Invoice(**values),
            )
        )
    ids = Counter(e.doc_id for e in examples)
    examples = [
        e.model_copy(update={"doc_id": f"{e.doc_id}#{i}"}) if ids[e.doc_id] > 1 else e
        for i, e in enumerate(examples)
    ]
    stats.written = len(examples)
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="extracted IOB/ directory")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    stats = Stats()
    write_examples(args.out, convert(args.root, stats))
    print(json.dumps({k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.__dict__.items()}, indent=2))


if __name__ == "__main__":
    main()
