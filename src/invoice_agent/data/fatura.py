"""Convert FATURA (Zenodo 10371464, CC-BY-4.0) to the canonical JSONL format.

    python -m invoice_agent.data.fatura data/raw/fatura/extracted/invoices_dataset_final \
        data/canonical/fatura.jsonl --per-template 4

FATURA has 10,000 synthetic invoice images from 50 templates. Annotation text includes the printed
caption ("TOTAL : 1113.61 EUR"), so each value is parsed out of it. The table is annotated only as a
box, so there are no line-item labels. Splits follow the authors' strategy 1 (per-instance).

Running a local LLM on all 10,000 isn't practical, so we take `--per-template` documents per
template from each of dev and test, and OCR only those images with Apple Vision.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from invoice_agent.data.loader import write_examples
from invoice_agent.data.ocr import boxes_to_text, run_vision_ocr
from invoice_agent.eval.normalize import to_date, to_decimal
from invoice_agent.schema import Example, Invoice, Split

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_CURRENCY = re.compile(r"\b(USD|EUR|GBP|INR|JPY|CAD|AUD|CHF)\b")
_DATE = re.compile(r"\d{1,2}-[A-Za-z]{3}-\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}")
_NUMBER_CAPTION = re.compile(r"^\s*(invoice\s*(id|number|no\.?|#)|inv\.?\s*no\.?|number|no\.?)\s*[:#.]?\s*", re.I)
_GSTIN_CAPTION = re.compile(r"^\s*\(?\s*GSTIN(/UIN)?(\s*Code)?\s*[:|]?\s*", re.I)
_BUYER_CAPTION = re.compile(r"^\s*(bill(ed)?[ _]?to|buyer)\s*:?\s*", re.I)
_ACCOUNT = re.compile(r"Account\s*(Number|No\.?)\s*:?\s*([A-Z0-9-]+)", re.I)


def _text(node: Any) -> str | None:
    if isinstance(node, dict):
        return node.get("text") or None
    if isinstance(node, list) and node and isinstance(node[0], dict):
        return node[0].get("text") or None
    return None


def after_colon(t: str) -> str:
    return t.split(":", 1)[1].strip() if ":" in t else t.strip()


def parse_amount(t: str) -> Any:
    nums = _NUM.findall(after_colon(t).replace(" ", ""))
    return to_decimal(nums[-1]) if nums else None


def parse_date(t: str) -> Any:
    m = _DATE.search(t)
    return to_date(m.group(0)) if m else None


def parse_number(t: str) -> str | None:
    v = _NUMBER_CAPTION.sub("", t.strip()).strip(" :#")
    return v or None


def parse_buyer(t: str) -> str | None:
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return None
    first = _BUYER_CAPTION.sub("", lines[0]).strip()
    return first or (lines[1] if len(lines) > 1 else None)


def parse_gstin(t: str) -> str | None:
    v = _GSTIN_CAPTION.sub("", t.replace("\n", " ")).strip(" ,:")
    return v or None


def parse_address(t: str) -> str | None:
    v = re.sub(r"^\s*Address\s*:\s*", "", t, flags=re.I).replace("\n", " ").strip()
    return v or None


def parse_account(t: str) -> str | None:
    m = _ACCOUNT.search(t)
    return m.group(2) if m else None


FIELDS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "NUMBER": ("invoice_number", parse_number),
    "DATE": ("invoice_date", parse_date),
    "DUE_DATE": ("due_date", parse_date),
    "SELLER_NAME": ("vendor_name", lambda t: t.strip() or None),
    "SELLER_ADDRESS": ("vendor_address", parse_address),
    "GSTIN_SELLER": ("vendor_tax_id", parse_gstin),
    "GSTIN_BUYER": ("buyer_tax_id", parse_gstin),
    "PO_NUMBER": ("po_number", after_colon),
    "SUB_TOTAL": ("subtotal", parse_amount),
    "TAX": ("tax_total", parse_amount),
    "TOTAL": ("total", parse_amount),
    "AMOUNT_DUE": ("amount_due", parse_amount),
    "PAYMENT_DETAILS": ("bank_account_number", parse_account),
}
BUYER_KEYS = ("BUYER", "BILL_TO")


def parse_annotation(ann: dict[str, Any], stats: Counter) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, (field, parser) in FIELDS.items():
        t = _text(ann.get(key))
        if not t:
            continue
        v = parser(t)
        if v in (None, ""):
            stats[f"unparsed:{field}"] += 1
            continue
        values[field] = v
    for key in BUYER_KEYS:
        t = _text(ann.get(key))
        if t and (name := parse_buyer(t)):
            values["buyer_name"] = name
            break
    total_text = _text(ann.get("TOTAL")) or _text(ann.get("SUB_TOTAL")) or ""
    if m := _CURRENCY.search(total_text):
        values["currency"] = m.group(1)
    return values


def read_split_files(root: Path) -> dict[str, Split]:
    splits: dict[str, Split] = {}
    for name, split in (("strat1_train.csv", Split.TRAIN), ("strat1_dev.csv", Split.DEV), ("strat1_test.csv", Split.TEST)):
        with open(root / name, newline="") as f:
            for row in csv.DictReader(f):
                splits[Path(row["img_path"]).stem] = split
    return splits


def template_of(stem: str) -> str:
    return stem.split("_", 1)[0]


def sample(stems: list[str], splits: dict[str, Split], per_template: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    groups: dict[tuple[str, Split], list[str]] = {}
    for s in sorted(stems):
        if splits.get(s) in (Split.DEV, Split.TEST):
            groups.setdefault((template_of(s), splits[s]), []).append(s)
    picked = []
    for key in sorted(groups):
        picked += rng.sample(groups[key], min(per_template, len(groups[key])))
    return picked


def convert(root: Path, per_template: int, seed: int, ocr: bool = True) -> tuple[list[Example], Counter]:
    stats: Counter = Counter()
    splits = read_split_files(root)
    ann_dir = root / "Annotations" / "Original_Format"
    stems = sample([p.stem for p in ann_dir.glob("*.json")], splits, per_template, seed)

    ocr_text: dict[str, str] = {}
    if ocr:
        paths = [root / "images" / f"{s}.jpg" for s in stems]
        for path, boxes, error in run_vision_ocr(paths):
            if error:
                stats["ocr_error"] += 1
            ocr_text[Path(path).stem] = boxes_to_text(boxes)

    examples = []
    for s in stems:
        values = parse_annotation(json.loads((ann_dir / f"{s}.json").read_text()), stats)
        for f in values:
            stats[f"labeled:{f}"] += 1
        examples.append(
            Example(
                doc_id=f"fatura/{s}",
                source="fatura",
                split=splits[s],
                document_path=str(root / "images" / f"{s}.jpg"),
                ocr_text=ocr_text.get(s),
                labeled_fields=frozenset(values),
                gold=Invoice(**values),
            )
        )
    stats["written"] = len(examples)
    return examples, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="extracted invoices_dataset_final/ directory")
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-template", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args()
    examples, stats = convert(args.root, args.per_template, args.seed, ocr=not args.no_ocr)
    write_examples(args.out, examples)
    print(json.dumps(dict(sorted(stats.items())), indent=2))


if __name__ == "__main__":
    main()
