"""Gold-free signals about whether an extraction is safe to post.

Everything here looks only at the prediction, the document text and the repair notes, never at
gold labels, so the same code runs in production.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal

from rapidfuzz import fuzz

from invoice_agent.eval.normalize import normalize
from invoice_agent.schema import FieldKind, Invoice
from invoice_agent.validate.amount_words import amount_in_words, printed_amounts
from invoice_agent.validate.gstin import is_valid_gstin, normalize_gstin, owner_matches

# What an ERP purchase invoice needs to be posted. Currency is critical for correctness, but when the
# document doesn't print one it comes from the company default, so its absence isn't a gate.
CRITICAL_FIELDS = ("vendor_name", "invoice_number", "invoice_date", "total", "currency")
REQUIRED_PRINTED = ("vendor_name", "invoice_number", "invoice_date", "total")
GROUNDED_TEXT_FIELDS = ("vendor_name", "invoice_number", "buyer_name", "vendor_tax_id", "buyer_tax_id")
GROUND_THRESHOLD = 90.0
MIN_DATE = dt.date(1990, 1, 1)
ARITH_TOLERANCE = Decimal("0.02")


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return "".join(ch for ch in s if ch.isalnum())


def grounded(value: str | None, text: str) -> bool | None:
    """Does the value literally appear in the document (ignoring case, spacing, punctuation)?"""
    if not value:
        return None
    v, t = _norm_text(value), _norm_text(text)
    if not v:
        return None
    return v in t or fuzz.partial_ratio(v, t) >= GROUND_THRESHOLD


def amount_grounded(value: Decimal | None, text: str) -> bool | None:
    if value is None:
        return None
    return any(abs(p - value) <= Decimal("0.01") for p in printed_amounts(text))


def looks_like_gstin(value: str | None) -> bool:
    g = normalize_gstin(value)
    return bool(g and len(g) == 15 and g[:2].isdigit())


@dataclass
class Signals:
    features: dict[str, float] = field(default_factory=dict)
    gates: list[str] = field(default_factory=list)  # reasons that force review regardless of score


def compute_signals(inv: Invoice, text: str | None, repairs: list[str], suspicious: bool, today: dt.date) -> Signals:
    text = text or ""
    s = Signals()
    f = s.features

    missing = [c for c in REQUIRED_PRINTED if getattr(inv, c) in (None, "")]
    f["missing_critical"] = len(missing)
    if missing:
        s.gates.append("missing:" + ",".join(missing))

    ungrounded = [name for name in GROUNDED_TEXT_FIELDS if grounded(getattr(inv, name), text) is False]
    f["ungrounded_text_fields"] = len(ungrounded)
    f["critical_ungrounded"] = sum(n in CRITICAL_FIELDS for n in ungrounded)
    if f["critical_ungrounded"]:
        s.gates.append("ungrounded:" + ",".join(n for n in ungrounded if n in CRITICAL_FIELDS))

    total_grounded = amount_grounded(inv.total, text)
    f["total_not_printed"] = float(total_grounded is False)

    words = amount_in_words(text)
    f["words_present"] = float(words is not None)
    f["words_disagree"] = float(words is not None and inv.total is not None and abs(inv.total - words) > 1)
    if f["words_disagree"]:
        s.gates.append("total_disagrees_with_words")
    f["total_repaired"] = float("total_from_words" in repairs)
    f["gstin_repaired"] = float(any(r.startswith("gstin_") for r in repairs))

    bad_gstin = 0
    owner_mismatch = 0
    for id_field, name_field in (("vendor_tax_id", "vendor_name"), ("buyer_tax_id", "buyer_name")):
        tax_id = getattr(inv, id_field)
        if looks_like_gstin(tax_id):
            bad_gstin += not is_valid_gstin(tax_id)
            owner_mismatch += owner_matches(tax_id, getattr(inv, name_field)) is False
    f["gstin_bad_checksum"] = bad_gstin
    f["gstin_owner_mismatch"] = owner_mismatch

    if inv.subtotal is not None and inv.total is not None:
        expected = inv.subtotal + (inv.tax_total or 0)
        f["arith_mismatch"] = float(abs(expected - inv.total) > ARITH_TOLERANCE)
    else:
        f["arith_mismatch"] = 0.0

    insane = inv.invoice_date is not None and not (MIN_DATE <= inv.invoice_date <= today)
    insane |= inv.due_date is not None and inv.invoice_date is not None and inv.due_date < inv.invoice_date
    f["date_insane"] = float(insane)
    if inv.invoice_date is not None and not (MIN_DATE <= inv.invoice_date <= today):
        s.gates.append("invoice_date_out_of_range")

    # Kept as a feature only: on clean MIDD/FATURA dev docs the model sets this on ~50% of invoices
    # (payment-terms boilerplate reads as "instructions"), so it can't gate anything.
    f["suspicious_instructions"] = float(suspicious)

    f["invoice_number_len"] = float(len(normalize(FieldKind.ID, inv.invoice_number) or ""))
    f["caption_in_value"] = float(
        bool(_CAPTION.search(inv.invoice_number or "")) or bool(_CAPTION.search(inv.vendor_name or ""))
    )
    return s


# Printed labels that leak into extracted values ("INVOICE ID2603-148", "e-Way Bill No. SM/..").
_CAPTION = re.compile(r"(invoice|inv\s*no|bill\s*no|dated|e-?way|gstin|address|tel\b|phone)", re.I)


def agreement(primary: Invoice, other: Invoice) -> dict[str, float]:
    """Field-level agreement with an independent second extraction, using the scorer's matching."""
    from invoice_agent.eval.normalize import values_match
    from invoice_agent.schema import HEADER_FIELDS

    out = {}
    disagree = 0
    for name in REQUIRED_PRINTED:
        a, b = getattr(primary, name), getattr(other, name)
        same = (a is None and b is None) or (a is not None and b is not None and values_match(HEADER_FIELDS[name], a, b))
        out[f"agree_{name}"] = float(same)
        disagree += not same
    out["critical_disagreements"] = float(disagree)
    return out
