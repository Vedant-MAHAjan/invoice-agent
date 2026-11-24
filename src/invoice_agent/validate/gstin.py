"""Indian GSTIN checks: format, mod-36 check digit, and party ownership via the embedded PAN.

GSTIN = 2-digit state code + 10-char PAN + entity number + 'Z' + check digit.
The PAN's 5th character is the first letter of the holder's name (surname for individuals),
which lets us tell whose tax ID is whose without trusting the model's party assignment.
"""

from __future__ import annotations

import re

from invoice_agent.schema import Invoice

_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_FORMAT = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_WORD = re.compile(r"[A-Z0-9]+")
_STOPWORDS = {"M", "S", "MS", "THE", "SHRI", "SRI", "MR", "MRS", "AND", "CO", "OF"}


def normalize_gstin(value: str | None) -> str | None:
    if not value:
        return None
    s = "".join(ch for ch in value.upper() if ch.isalnum())
    return s or None


def check_digit(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14):
        product = _ALPHABET.index(ch) * (2 if i % 2 else 1)
        total += product // 36 + product % 36
    return _ALPHABET[(36 - total % 36) % 36]


def is_valid_gstin(value: str | None) -> bool:
    g = normalize_gstin(value)
    return bool(g and _FORMAT.match(g) and check_digit(g[:14]) == g[14])


def name_initials(name: str | None) -> set[str]:
    words = [w for w in _WORD.findall((name or "").upper()) if w not in _STOPWORDS]
    return {w[0] for w in words if w[0].isalpha()}


def owner_matches(gstin: str | None, name: str | None) -> bool | None:
    """True/False if the GSTIN's PAN initial does/doesn't match the name; None if undecidable."""
    g = normalize_gstin(gstin)
    initials = name_initials(name)
    if not g or len(g) != 15 or not initials:
        return None
    return g[6] in initials


def repair_party_tax_ids(inv: Invoice) -> tuple[Invoice, list[str]]:
    """Move tax IDs to the party whose name they belong to. Only acts when the evidence is one-sided,
    so two parties sharing an initial leave the extraction untouched."""
    v_id, b_id = inv.vendor_tax_id, inv.buyer_tax_id
    v_name, b_name = inv.vendor_name, inv.buyer_name

    def belongs_to_buyer_not_vendor(tax_id: str | None) -> bool:
        return owner_matches(tax_id, b_name) is True and owner_matches(tax_id, v_name) is False

    def belongs_to_vendor_not_buyer(tax_id: str | None) -> bool:
        return owner_matches(tax_id, v_name) is True and owner_matches(tax_id, b_name) is False

    if v_id and b_id and belongs_to_buyer_not_vendor(v_id) and belongs_to_vendor_not_buyer(b_id):
        return inv.model_copy(update={"vendor_tax_id": b_id, "buyer_tax_id": v_id}), ["gstin_swapped"]
    if v_id and not b_id and belongs_to_buyer_not_vendor(v_id):
        return inv.model_copy(update={"vendor_tax_id": None, "buyer_tax_id": v_id}), ["gstin_moved_to_buyer"]
    if b_id and not v_id and belongs_to_vendor_not_buyer(b_id):
        return inv.model_copy(update={"vendor_tax_id": b_id, "buyer_tax_id": None}), ["gstin_moved_to_vendor"]
    return inv, []
