"""Cross-check the extracted grand total against the amount printed in words."""

from __future__ import annotations

from decimal import Decimal

from invoice_agent.schema import Invoice
from invoice_agent.validate.amount_words import amount_in_words, printed_amounts

TOLERANCE = Decimal("1")  # words usually drop paise/rounding


def repair_total_from_words(inv: Invoice, text: str | None) -> tuple[Invoice, list[str]]:
    """Replace the total only when the words amount is also printed as a number in the document,
    so two independent renderings agree; a garbled words line can never overwrite a good total."""
    words = amount_in_words(text)
    if words is None:
        return inv, []
    if inv.total is not None and abs(inv.total - words) <= TOLERANCE:
        return inv, ["total_matches_words"]
    if any(abs(p - words) <= TOLERANCE for p in printed_amounts(text or "")):
        return inv.model_copy(update={"total": words}), ["total_from_words"]
    return inv, ["total_disagrees_words"]
