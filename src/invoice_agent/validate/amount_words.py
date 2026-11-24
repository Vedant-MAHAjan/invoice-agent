"""Parse the amount-in-words line many invoices print ("Indian Rupees Two Lakh Eighty Thousand
Two Hundred Fifty Only") into a number, as an independent check on the extracted total.
Handles Indian (lakh, crore) and international (thousand, million, billion) scales and paise/cents."""

from __future__ import annotations

import re
from decimal import Decimal

UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
SCALES = {"thousand": 10**3, "lakh": 10**5, "lakhs": 10**5, "lac": 10**5, "lacs": 10**5,
          "million": 10**6, "crore": 10**7, "crores": 10**7, "billion": 10**9}
FRACTION_WORDS = {"paise", "paisa", "cents", "cent"}
_NUMBER_WORDS = set(UNITS) | set(TENS) | set(SCALES) | FRACTION_WORDS | {"hundred", "and"}
_WORD_ALT = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
# A run of number words ending in "only"; the text just before it is the label.
_PHRASE = re.compile(rf"\b((?:(?:{_WORD_ALT})[\s\-,]+)+)only\b", re.I)
_TAX_LABEL = re.compile(r"\b(tax|gst|cgst|sgst|igst|vat)\b", re.I)
_LABEL_CHARS = 40


def words_to_int(words: str) -> int | None:
    tokens = [t for t in re.split(r"[\s\-,]+", words.lower()) if t and t not in ("and", "&")]
    if not tokens:
        return None
    total = current = 0
    for t in tokens:
        if t in UNITS:
            current += UNITS[t]
        elif t in TENS:
            current += TENS[t]
        elif t == "hundred":
            current = (current or 1) * 100
        elif t in SCALES:
            total += (current or 1) * SCALES[t]
            current = 0
        else:
            return None
    return total + current


def words_to_amount(words: str) -> Decimal | None:
    parts = re.split(r"\b(?:and\s+)?(?:" + "|".join(FRACTION_WORDS) + r")\b", words.lower())
    main = parts[0]
    frac = None
    m = re.search(r"\band\s+([a-z\s\-]+?)\s+(?:" + "|".join(FRACTION_WORDS) + r")\b", words.lower())
    if m:
        main = words.lower()[: m.start()]
        frac = words_to_int(m.group(1))
        if frac is None or frac >= 100:
            return None
    whole = words_to_int(main)
    if whole is None:
        return None
    return Decimal(whole) + (Decimal(frac) / 100 if frac else Decimal(0))


_NUMBER_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def printed_amounts(text: str) -> set[Decimal]:
    out = set()
    for tok in _NUMBER_TOKEN.findall(text):
        try:
            out.add(Decimal(tok.replace(",", "")))
        except ArithmeticError:
            continue
    return out


def amount_in_words(text: str | None) -> Decimal | None:
    """Return the grand total stated in words, or None if absent or unparseable.
    Phrases labelled as tax ("Tax Amount (in words)", "Total GST :") are skipped; of the rest, the
    largest is taken, since a document's grand total is never below its other stated amounts."""
    if not text:
        return None
    found = []
    for m in _PHRASE.finditer(text):
        label = text[max(0, m.start() - _LABEL_CHARS) : m.start()]
        if _TAX_LABEL.search(label):
            continue
        amount = words_to_amount(m.group(1))
        if amount is not None and amount > 0:
            found.append(amount)
    return max(found) if found else None
