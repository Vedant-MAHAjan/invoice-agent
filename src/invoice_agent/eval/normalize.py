"""Turn field values into comparable forms so scoring measures extraction, not formatting."""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

from dateutil import parser as date_parser
from rapidfuzz import fuzz

from invoice_agent.schema import FieldKind

TEXT_MATCH_THRESHOLD = 90.0
MONEY_TOLERANCE = Decimal("0.005")
NUMBER_TOLERANCE = Decimal("0.0001")

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
_NON_ALNUM = re.compile(r"[^0-9A-Za-z]")
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def to_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    s = str(value).strip()
    if not s:
        return None
    negative = s.startswith("(") and s.endswith(")")  # accounting negatives
    s = re.sub(r"[^\d,.\-]", "", s)
    if "," in s and "." in s:
        # whichever separator comes last is the decimal point
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = f"{head.replace(',', '')}.{tail}" if len(tail) in (1, 2) else s.replace(",", "")
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    return -d if negative else d


def to_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return date_parser.parse(str(value), dayfirst=False, fuzzy=True).date()
    except (ValueError, OverflowError):
        return None


def normalize(kind: FieldKind, value: Any) -> Any:
    """Return a canonical comparable value, or None if the value is empty/unparseable."""
    if value is None:
        return None
    if kind is FieldKind.TEXT:
        s = unicodedata.normalize("NFKC", str(value)).casefold()
        s = _SPACES.sub(" ", _PUNCT.sub(" ", s)).strip()
        return s or None
    if kind is FieldKind.ID:
        s = _NON_ALNUM.sub("", unicodedata.normalize("NFKC", str(value))).upper()
        return s or None
    if kind is FieldKind.CURRENCY:
        s = str(value).strip()
        return _CURRENCY_SYMBOLS.get(s, s.upper()) or None
    if kind is FieldKind.DATE:
        return to_date(value)
    if kind in (FieldKind.MONEY, FieldKind.NUMBER):
        return to_decimal(value)
    raise ValueError(f"unknown kind {kind}")


def values_match(kind: FieldKind, gold: Any, pred: Any) -> bool:
    g, p = normalize(kind, gold), normalize(kind, pred)
    if g is None or p is None:
        return g is None and p is None
    if kind is FieldKind.TEXT:
        return g == p or fuzz.token_sort_ratio(g, p) >= TEXT_MATCH_THRESHOLD
    if kind is FieldKind.MONEY:
        return abs(g - p) <= MONEY_TOLERANCE
    if kind is FieldKind.NUMBER:
        return abs(g - p) <= NUMBER_TOLERANCE * max(Decimal(1), abs(g))
    return g == p
