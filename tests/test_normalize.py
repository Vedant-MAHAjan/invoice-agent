import datetime as dt
from decimal import Decimal

import pytest

from invoice_agent.eval.normalize import to_date, to_decimal, values_match
from invoice_agent.schema import FieldKind


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1,234.56", "1234.56"),
        ("1.234,56", "1234.56"),
        ("$ 1,234", "1234"),
        ("12,5", "12.5"),
        ("(45.00)", "-45.00"),
        ("€99", "99"),
        (12.3, "12.3"),
    ],
)
def test_to_decimal(raw, expected):
    assert to_decimal(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", [None, "", "n/a", True])
def test_to_decimal_empty(raw):
    assert to_decimal(raw) is None


def test_to_date():
    assert to_date("March 5, 2025") == dt.date(2025, 3, 5)
    assert to_date("2025-03-05") == dt.date(2025, 3, 5)
    assert to_date("not a date") is None


def test_id_ignores_separators_and_case():
    assert values_match(FieldKind.ID, "INV-2025/0042", "inv 2025 0042")
    assert not values_match(FieldKind.ID, "INV-0042", "INV-42")


def test_text_is_fuzzy_but_bounded():
    assert values_match(FieldKind.TEXT, "Northwind Traders Ltd.", "NORTHWIND TRADERS LTD")
    assert not values_match(FieldKind.TEXT, "Northwind Traders", "Contoso Office")


def test_money_tolerance():
    assert values_match(FieldKind.MONEY, Decimal("100.00"), "100.004")
    assert not values_match(FieldKind.MONEY, Decimal("100.00"), "100.01")


def test_currency_symbol():
    assert values_match(FieldKind.CURRENCY, "USD", "$")
