from decimal import Decimal

import pytest

from invoice_agent.schema import Invoice
from invoice_agent.validate.amount_words import amount_in_words, words_to_amount, words_to_int
from invoice_agent.validate.totals import repair_total_from_words


@pytest.mark.parametrize(
    "words,expected",
    [
        ("Seventy Seven", 77),
        ("Two Lakh Eighty Thousand Two Hundred Fifty", 280250),
        ("Twenty Eight Thousand Five Hundred Forty", 28540),
        ("One Crore Five Lakh", 10500000),
        ("Three Million Two Hundred Thousand", 3200000),
        ("Hundred", 100),
    ],
)
def test_words_to_int(words, expected):
    assert words_to_int(words) == expected


def test_paise_and_garbage():
    assert words_to_amount("Eleven and Seventy paise") == Decimal("11.70")
    assert words_to_int("Seventy Banana") is None


def test_picks_total_not_tax():
    text = (
        "Total 30 Nos . 77 E. & O.E Amount Chargeable ( in words ) Indian Rupees Seventy Seven Only "
        "Total Tax Amount ( in words ) : Indian Rupees Eleven and Seventy paise Only"
    )
    assert amount_in_words(text) == Decimal(77)
    gst_first = "Total GST : Forty Two Thousand Seven Hundred Fifty Only Bill Amount : Two Lakh Eighty Thousand Two Hundred Fifty Only"
    assert amount_in_words(gst_first) == Decimal(280250)
    assert amount_in_words("Subject to PUNE Jurisdiction only") is None


TEXT = "Total 30 Nos . 77 E. & O.E Amount Chargeable ( in words ) Indian Rupees Seventy Seven Only"


def test_repair_replaces_only_when_corroborated():
    fixed, notes = repair_total_from_words(Invoice(total=Decimal("88.7")), TEXT)
    assert fixed.total == Decimal(77) and notes == ["total_from_words"]
    same, notes = repair_total_from_words(Invoice(total=Decimal("77.00")), TEXT)
    assert same.total == Decimal("77.00") and notes == ["total_matches_words"]
    uncorroborated = "Amount ( in words ) Indian Rupees Nine Only"  # 9 not printed as a number
    kept, notes = repair_total_from_words(Invoice(total=Decimal("231929")), uncorroborated)
    assert kept.total == Decimal("231929") and notes == ["total_disagrees_words"]
    assert repair_total_from_words(Invoice(total=Decimal(5)), "no words here") == (Invoice(total=Decimal(5)), [])
